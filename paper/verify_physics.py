"""
物理正确性核验 (P2) —— 生成论文附录用的数值证据。

检查项：
  A. H(q) 对称正定 + 条件数分布（学到的 H vs MuJoCo 真值 H，看是不是模型的锅）
  B. 科里奥利项能量恒等式  cᵀq̇ = ½ q̇ᵀ Ḣ q̇
  C. JVP 求出的 Ḣ·q̇ 与有限差分 / 显式 ∂H/∂q 收缩的一致性
  D. 学到的 H(q)、g(q) 与 MuJoCo 真值的偏差（只有 Franka 有解析真值）
  E. 自适应 λ 的分布（是否塌缩到 0 ⇒ 物理分支被关掉）
  F. 物理分支单独的预测能力（τ_rigid 对 τ 的 R²）

    python paper/verify_physics.py --ckpt results_v2/full_franka_seed0.pth
    python paper/verify_physics.py --all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paper import core
from paper.core import ARMS, Normalizer, load_arm, metrics, provenance, save_json
from paper.models import ModelCfg, build_model

ROOT = core.ROOT
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_EVAL = 2000


def load_ckpt(path):
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    cfg = ModelCfg(**ck['cfg'])
    arm = ck['arm']
    data = load_arm(arm)
    norm = Normalizer(*data['train'])
    model = build_model(cfg, 7, norm, ARMS[arm].axes).to(DEVICE)
    model.load_state_dict(ck['state_dict'])
    model.eval()
    return model, cfg, arm, data, norm, ck


# ---------------------------------------------------------------- MuJoCo 真值

def mujoco_ground_truth(q, dq, ddq):
    """返回 (H_true, g_true, tau_true_id)。只对 tool/panda.xml（Franka）可用。"""
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(ROOT / 'tool' / 'panda.xml'))
    d = mujoco.MjData(m)
    n = m.nv
    H = np.zeros((len(q), n, n), dtype=np.float64)
    g = np.zeros((len(q), n), dtype=np.float64)
    tau = np.zeros((len(q), n), dtype=np.float64)
    M = np.zeros((n, n), dtype=np.float64)
    for i in range(len(q)):
        d.qpos[:] = q[i]
        d.qvel[:] = 0.0
        d.qacc[:] = 0.0
        mujoco.mj_forward(m, d)
        mujoco.mj_fullM(m, M, d.qM)
        H[i] = M
        mujoco.mj_inverse(m, d)
        g[i] = d.qfrc_inverse.copy()          # q̇=q̈=0 ⇒ 只剩重力
        d.qvel[:] = dq[i]
        d.qacc[:] = ddq[i]
        mujoco.mj_inverse(m, d)
        tau[i] = d.qfrc_inverse.copy()
    return H, g, tau


# ---------------------------------------------------------------- 各项检查

def check_H(model, q_t):
    H = model.inertia_phys(q_t)
    Hs = H.detach()
    asym = (Hs - Hs.transpose(1, 2)).abs().max().item()
    ev = torch.linalg.eigvalsh(Hs.double())
    cond = (ev[:, -1] / ev[:, 0].clamp_min(1e-30)).cpu().numpy()
    return {
        'max_asymmetry': asym,
        'min_eigenvalue': float(ev.min()),
        'frac_positive_definite': float((ev.min(dim=1).values > 0).float().mean()),
        'cond_mean': float(np.mean(cond)), 'cond_median': float(np.median(cond)),
        'cond_p95': float(np.percentile(cond, 95)), 'cond_max': float(cond.max()),
        'cond_values': cond[:500].tolist(),
    }


def check_energy_identity(model, q_t, dq_t, ddq_t):
    """c ≡ Ḣq̇ − ½∂(q̇ᵀHq̇)/∂q 必须满足 cᵀq̇ = ½ q̇ᵀ Ḣ q̇（Ḣ−2C 反对称的标量形式）。"""
    q = q_t.clone().requires_grad_(True)
    phys = model.phys
    qm, qs = model.q_mean, model.q_std

    def h_fn(qq):
        Hq = phys.construct_H(qq, qm, qs)
        return Hq, torch.bmm(Hq, dq_t.unsqueeze(2)).squeeze(2)

    (H, _), (Hdot, Hdot_dq) = torch.func.jvp(h_fn, (q,), (dq_t,))
    quad = torch.bmm(torch.bmm(dq_t.unsqueeze(1), H), dq_t.unsqueeze(2)).squeeze(-1).squeeze(-1)
    grad_quad = torch.autograd.grad(quad.sum(), q, retain_graph=True)[0]
    c = Hdot_dq - 0.5 * grad_quad
    lhs = (c * dq_t).sum(1)
    rhs = 0.5 * torch.bmm(torch.bmm(dq_t.unsqueeze(1), Hdot), dq_t.unsqueeze(2)).squeeze(-1).squeeze(-1)
    err = (lhs - rhs).abs()
    scale = torch.maximum(lhs.abs(), rhs.abs()).clamp_min(1e-12)
    return {
        'abs_err_mean': float(err.mean()), 'abs_err_max': float(err.max()),
        'rel_err_mean': float((err / scale).mean()), 'rel_err_max': float((err / scale).max()),
        'lhs_scale_mean': float(lhs.abs().mean()),
        'lhs_sample': lhs[:200].detach().cpu().tolist(),
        'rhs_sample': rhs[:200].detach().cpu().tolist(),
    }


def check_jvp_vs_explicit(model, q_t, dq_t, n=64):
    """JVP 的 Ḣq̇ 与逐分量 reverse-mode 显式求 ∂H/∂q_i 再收缩的结果对比。"""
    phys, qm, qs = model.phys, model.q_mean, model.q_std
    q = q_t[:n].clone().requires_grad_(True)
    dq = dq_t[:n]

    def h_fn(qq):
        Hq = phys.construct_H(qq, qm, qs)
        return torch.bmm(Hq, dq.unsqueeze(2)).squeeze(2)

    _, jvp_out = torch.func.jvp(h_fn, (q,), (dq,))

    # 显式: (Ḣq̇)_j = Σ_k Σ_i ∂H_jk/∂q_i · q̇_i · q̇_k
    H = phys.construct_H(q, qm, qs)
    Hdq = torch.bmm(H, dq.unsqueeze(2)).squeeze(2)     # (n, 7)
    explicit = torch.zeros_like(Hdq)
    for j in range(7):
        gj = torch.autograd.grad(Hdq[:, j].sum(), q, retain_graph=True, create_graph=False)[0]
        explicit[:, j] = (gj * dq).sum(1)
    diff = (jvp_out - explicit).abs()
    denom = explicit.abs().mean().clamp_min(1e-12)
    return {'abs_err_mean': float(diff.mean()), 'abs_err_max': float(diff.max()),
            'rel_err_mean': float(diff.mean() / denom), 'magnitude': float(denom)}


def check_vs_mujoco(model, arm, q, dq, ddq):
    if not arm.startswith('franka'):
        return {'available': False, 'reason': 'Baxter 无解析真值模型（数据来自实机）'}
    q_t = torch.tensor(q, dtype=torch.float32, device=DEVICE).requires_grad_(True)
    H_p = model.inertia_phys(q_t).detach().double().cpu().numpy()
    g_p = model.gravity_phys(q_t).detach().double().cpu().numpy()

    # 逐点真值：H 用 mj_fullM；g 用 q̇=q̈=0 时的 mj_inverse
    import mujoco
    mj = mujoco.MjModel.from_xml_path(str(ROOT / 'tool' / 'panda.xml'))
    md = mujoco.MjData(mj)
    n = mj.nv
    M = np.zeros((n, n), dtype=np.float64)
    H_t = np.zeros((len(q), n, n)); g_t = np.zeros((len(q), n))
    for i in range(len(q)):
        md.qpos[:] = q[i]; md.qvel[:] = 0; md.qacc[:] = 0
        mujoco.mj_forward(mj, md); mujoco.mj_fullM(mj, M, md.qM); H_t[i] = M
        # mj_forward 会覆写 qacc；重设 0 再求逆，否则重力项被当作「给定了加速度」而消失
        md.qacc[:] = 0
        mujoco.mj_inverse(mj, md); g_t[i] = md.qfrc_inverse

    def rel(a, b):
        return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))

    Hq_p = np.einsum('bij,bj->bi', H_p, ddq)
    Hq_t = np.einsum('bij,bj->bi', H_t, ddq)
    ev_t = np.linalg.eigvalsh(H_t)

    # g 逐关节：报告相对误差与相关系数，只在真值有量级的关节上报告相对误差
    g_corr, g_rel_joint, g_mag = [], [], []
    for i in range(n):
        rms = float(np.sqrt((g_t[:, i] ** 2).mean()))
        g_corr.append(float(np.corrcoef(g_p[:, i], g_t[:, i])[0, 1]) if g_t[:, i].std() > 1e-9
                      else float('nan'))
        g_rel_joint.append(float(np.linalg.norm(g_p[:, i] - g_t[:, i]) /
                                 np.linalg.norm(g_t[:, i])) if rms > 1e-2 else float('nan'))
        g_mag.append(rms)
    act = np.array([r > 1e-2 for r in g_mag])
    return {
        'available': True,
        'H_rel_fro_err': rel(H_p, H_t),
        'Hddq_rel_err': rel(Hq_p, Hq_t),
        'H_diag_pred_mean': H_p.diagonal(axis1=1, axis2=2).mean(0).tolist(),
        'H_diag_true_mean': H_t.diagonal(axis1=1, axis2=2).mean(0).tolist(),
        'g_rel_err': float(np.linalg.norm(g_p[:, act] - g_t[:, act]) /
                           max(np.linalg.norm(g_t[:, act]), 1e-12)),
        'g_rel_err_active_joints': [bool(x) for x in act],
        'g_corr_per_joint': g_corr,
        'g_rel_err_per_joint': g_rel_joint,
        'g_true_rms_per_joint': g_mag,
        'g_pred_rms_per_joint': [float(np.sqrt((g_p[:, i] ** 2).mean())) for i in range(n)],
        'true_cond_mean': float(np.mean(ev_t[:, -1] / ev_t[:, 0])),
        'true_min_eig_mean': float(ev_t[:, 0].mean()),
    }


def check_lambda(model, q_t, dq_t, ddq_t):
    if not getattr(model.cfg, 'adaptive_lambda', False):
        return {'available': False}
    with torch.no_grad():
        z = model.zscore(q_t, dq_t, ddq_t)
        lam = model.lam_net(*z.chunk(3, dim=-1)).cpu().numpy()
    return {'available': True, 'mean': float(lam.mean()), 'std': float(lam.std()),
            'min': float(lam.min()), 'max': float(lam.max()),
            'frac_below_1e-3': float((lam < 1e-3).mean()),
            'ceiling': 0.1, 'hist': np.histogram(lam, bins=20, range=(0, 0.1))[0].tolist()}


def check_branch_split(model, cfg, data, norm, arm):
    """物理分支单独 / 不确定性分支单独 对真实 τ 的解释力。"""
    from paper.train import predict, to_t
    q, dq, ddq, tau = data['test']
    q_t, dq_t, ddq_t = to_t(q), to_t(dq), to_t(ddq)
    names = ARMS[arm].joint_names
    out = {}
    preds = []
    with torch.enable_grad():
        for i in range(0, len(q), 2048):
            p, c = model(q_t[i:i + 2048].clone().requires_grad_(True),
                         dq_t[i:i + 2048], ddq_t[i:i + 2048], want_terms=False)
            preds.append({k: v.detach() for k, v in c.items() if torch.is_tensor(v) and v.dim() == 2})
            preds[-1]['total_n'] = p.detach()
    cat = {k: torch.cat([p[k] for p in preds]).cpu().numpy() for k in preds[0]}
    out['total'] = metrics(tau, norm.denorm_tau(cat['total_n']), names)
    if cfg.physics:
        out['physics_only'] = metrics(tau, cat['tau_rigid_phys'], names)
        out['physics_share'] = float(np.abs(cat['tau_rigid_phys']).mean() /
                                     max(np.abs(norm.denorm_tau(cat['total_n'])).mean(), 1e-12))
    if cfg.uncertainty:
        eps_phys = cat['epsilon'] * norm.tau_scale
        out['uncertainty_rms_Nm'] = float(np.sqrt((eps_phys ** 2).mean()))
        out['tau_rms_Nm'] = float(np.sqrt((tau ** 2).mean()))
        out['uncertainty_share'] = out['uncertainty_rms_Nm'] / max(out['tau_rms_Nm'], 1e-12)
    return out


# ---------------------------------------------------------------- 主流程

def verify(ckpt_path, out_dir='experimental_results/v2_physics'):
    model, cfg, arm, data, norm, ck = load_ckpt(ckpt_path)
    q, dq, ddq, tau = data['test']
    idx = np.linspace(0, len(q) - 1, min(N_EVAL, len(q))).astype(int)
    qs, dqs, ddqs = q[idx], dq[idx], ddq[idx]
    q_t = torch.tensor(qs, device=DEVICE).requires_grad_(True)
    dq_t = torch.tensor(dqs, device=DEVICE)
    ddq_t = torch.tensor(ddqs, device=DEVICE)

    rep = {'checkpoint': str(ckpt_path), 'variant': ck['variant'], 'arm': arm,
           'seed': ck['seed'], 'cfg': ck['cfg'], 'n_eval': len(idx),
           'provenance': provenance()}

    if cfg.physics:
        rep['A_inertia'] = check_H(model, q_t)
        rep['B_energy_identity'] = check_energy_identity(model, q_t, dq_t, ddq_t)
        rep['C_jvp_consistency'] = check_jvp_vs_explicit(model, q_t, dq_t)
        rep['D_vs_mujoco'] = check_vs_mujoco(model, arm, qs, dqs, ddqs)
    rep['E_lambda'] = check_lambda(model, q_t, dq_t, ddq_t)
    rep['F_branch_split'] = check_branch_split(model, cfg, data, norm, arm)

    name = Path(ckpt_path).stem
    p = save_json(ROOT / out_dir / f'{name}_physics.json', rep)
    print(f'[saved] {p}')
    return rep


def summarize(rep):
    print(f"\n=== {rep['variant']} / {rep['arm']} seed{rep['seed']} ===")
    if 'A_inertia' in rep:
        a = rep['A_inertia']
        print(f"  A H(q): 对称误差={a['max_asymmetry']:.2e}  最小特征值={a['min_eigenvalue']:.3e}  "
              f"正定比例={a['frac_positive_definite']:.3f}")
        print(f"    条件数 中位数={a['cond_median']:.3g} 均值={a['cond_mean']:.3g} "
              f"p95={a['cond_p95']:.3g} max={a['cond_max']:.3g}")
        b = rep['B_energy_identity']
        print(f"  B 能量恒等式 cᵀq̇=½q̇ᵀḢq̇: 相对误差 mean={b['rel_err_mean']:.3e} "
              f"max={b['rel_err_max']:.3e}")
        c = rep['C_jvp_consistency']
        print(f"  C JVP vs 显式 ∂H/∂q: 相对误差 mean={c['rel_err_mean']:.3e}")
        d = rep['D_vs_mujoco']
        if d.get('available'):
            print(f"  D vs MuJoCo 真值: ‖ΔH‖/‖H‖={d['H_rel_fro_err']:.3f}  "
                  f"‖ΔHq̈‖/‖Hq̈‖={d['Hddq_rel_err']:.3f}  ‖Δg‖/‖g‖={d['g_rel_err']:.3f}")
            print(f"    真值 H 条件数均值={d['true_cond_mean']:.3g}（学到的={rep['A_inertia']['cond_mean']:.3g}）")
        else:
            print(f"  D vs MuJoCo: {d.get('reason')}")
    e = rep['E_lambda']
    if e.get('available'):
        print(f"  E λ: mean={e['mean']:.4f} std={e['std']:.4f} 区间[{e['min']:.4f},{e['max']:.4f}] "
              f"上限0.1  塌缩到<1e-3 的比例={e['frac_below_1e-3']:.3f}")
    f = rep['F_branch_split']
    print(f"  F 分支贡献: 整体 R²(global)={f['total']['r2_global']:.4f}")
    if 'physics_only' in f:
        print(f"    仅物理分支 R²(global)={f['physics_only']['r2_global']:.4f}  "
              f"逐关节={[round(v,3) for v in f['physics_only']['r2_per_joint'].values()]}")
    if 'uncertainty_share' in f:
        print(f"    不确定性分支 RMS={f['uncertainty_rms_Nm']:.4f} N·m "
              f"(占 τ RMS {f['uncertainty_share']*100:.1f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--pattern', default='*_seed0.pth')
    a = ap.parse_args()
    paths = ([Path(a.ckpt)] if a.ckpt else
             sorted((ROOT / 'results_v2').glob(a.pattern)))
    for p in paths:
        try:
            summarize(verify(p))
        except Exception as ex:
            import traceback
            print(f'[FAIL] {p}: {ex}')
            traceback.print_exc()


if __name__ == '__main__':
    main()
