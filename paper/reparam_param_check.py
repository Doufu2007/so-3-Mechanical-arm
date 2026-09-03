"""可辨识子空间重参数化分支的恢复核验 + 「重参数化前后」对照表。

回答的问题：把 SE(3) 分支从 K=80 全重心参数压到 43 维可辨识子空间（π=B·θ）后，
「学回真参数」从不可能（有零空间）变成可能。本脚本产出：

  1. --selfcheck : 把 θ 置为 Bᵀ·π_true，验证 mass_matrix 与 MuJoCo 一致
     （证明 B 与几何/数学都对，且 M(π_null)=0 ⇒ π_id 就给出正确的 M）
  2. --ckpt      : 载入训练好的 se3_reparam checkpoint，报告
       a. θ 恢复误差 ‖θ̂−θ_true‖ / ‖θ_true‖
       b. 逐连杆 (m, c, I) 恢复误差（重心参数转回物理量后 vs MuJoCo 真值）
       c. 「重参数化前后」对照：naive se3（有零空间）vs reparam（无零空间）

用法（须在项目根目录下跑，MuJoCo 中文路径会 ValueError）：
    python paper/reparam_param_check.py --selfcheck
    python paper/reparam_param_check.py --ckpt results_v2/se3_reparam_no_uncertainty_franka_ex_seed0.pth
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from se3_physics import extract_geometry                                   # noqa: E402
from base_reparam_branch import ReparamBranch                              # noqa: E402
from identifiability import true_pi                                        # noqa: E402
from se3_param_check import check_params, _body_names                       # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_basis(path):
    d = np.load(path)
    return d['B'], d['pi_true'], d['rank']


def theta_true_of(B, pi_true):
    """真 base 参数 θ_true = Bᵀ·π_true（可辨识分量的坐标）。"""
    return B.T @ pi_true


def selfcheck(xml, basis_path, n_random=64, seed=0, tol=1e-3):
    """θ=Bᵀ·π_true ⇒ π=B·θ=π_id，验证 mass_matrix(π_id) 与 MuJoCo mj_fullM 一致。

    M 对 π 线性，且 π_null 的 M 恒为 0（Y·π_null=0 对任意 q̈ 成立 ⇒ M(π_null)=0），
    故 M(π_id)=M(π_true)。若此自检通过，说明「压掉零空间不损失任何动力学」。
    """
    import mujoco
    m = mujoco.MjModel.from_xml_path(xml)
    d = mujoco.MjData(m)
    geom = extract_geometry(xml)
    B, pi_true, rank = load_basis(basis_path)

    branch = ReparamBranch(geom, B).to(DEVICE)
    with torch.no_grad():
        branch.theta.copy_(torch.as_tensor(theta_true_of(B, pi_true),
                                           dtype=torch.float32).to(DEVICE))
    branch.eval()

    rng = np.random.RandomState(seed)
    qlo = np.maximum(-1.4, m.jnt_range[:, 0])
    qhi = np.minimum(1.4, m.jnt_range[:, 1])
    err_M = []
    for _ in range(n_random):
        q = rng.uniform(qlo, qhi, m.nv)
        d.qpos[:] = q; d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        M_true = np.zeros((m.nv, m.nv))
        mujoco.mj_fullM(m, M_true, d.qM)
        qt = torch.tensor(q, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            M_pred = branch.mass_matrix(qt)[0].cpu().numpy()
        err_M.append(np.abs(M_pred - M_true).max())
    emax = float(np.max(err_M))
    return {'M_max_abs_err': emax, 'tol': tol, 'rank': int(rank),
            'n_random': n_random, 'theta_true_rank': int(rank)}


def forward_smoke(xml, basis_path, seed=0):
    """验证 forward() 的 jvp 通路在 θ 为可学习参数时正常，且梯度能回传到 θ。"""
    geom = extract_geometry(xml)
    B, pi_true, rank = load_basis(basis_path)
    branch = ReparamBranch(geom, B).to(DEVICE)
    with torch.no_grad():
        branch.theta.copy_(torch.as_tensor(theta_true_of(B, pi_true),
                                           dtype=torch.float32).to(DEVICE))
    branch.train()
    rng = np.random.RandomState(seed)
    q = torch.tensor(rng.uniform(-1.2, 1.2, (8, 7)), dtype=torch.float32, device=DEVICE)
    dq = torch.tensor(rng.uniform(-2, 2, (8, 7)), dtype=torch.float32, device=DEVICE)
    ddq = torch.tensor(rng.uniform(-8, 8, (8, 7)), dtype=torch.float32, device=DEVICE)
    tau, _ = branch.forward(q, dq, ddq)
    loss = tau.pow(2).mean()
    loss.backward()
    g = branch.theta.grad
    return {'tau_finite': bool(torch.isfinite(tau).all()),
            'tau_norm': float(tau.norm()),
            'theta_grad_norm': float(g.norm()) if g is not None else -1.0}


def load_reparam_ckpt(path):
    from paper import core
    from paper.core import ARMS, Normalizer, load_arm
    from paper.models import ModelCfg, build_model
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    cfg = ModelCfg(**ck['cfg'])
    if cfg.physics_kind != 'se3_reparam':
        raise ValueError(f'{path} 不是 se3_reparam 变体（physics_kind={cfg.physics_kind}）')
    arm = ck['arm']
    data = load_arm(arm)
    norm = Normalizer(*data['train'])
    model = build_model(cfg, 7, norm, ARMS[arm].axes).to(DEVICE)
    model.load_state_dict(ck['state_dict'])
    model.eval()
    return model, cfg, arm, ck


def report_recovery(path, xml, basis_path):
    """θ 恢复 + 逐连杆物理参数恢复 + 对照表。"""
    model, cfg, arm, ck = load_reparam_ckpt(path)
    geom = extract_geometry(xml)
    B, pi_true, rank = load_basis(basis_path)

    with torch.no_grad():
        theta = model.phys.theta.detach().cpu().numpy().astype(np.float64)
    theta_true = theta_true_of(B, pi_true)
    theta_err = float(np.linalg.norm(theta - theta_true) / np.linalg.norm(theta_true))
    # π 空间：学到的 π=B·θ vs 真 π 的可辨识投影 B·Bᵀ·π_true
    pi_learned = B @ theta
    pi_id = B @ theta_true
    pi_err = float(np.linalg.norm(pi_learned - pi_id) / np.linalg.norm(pi_id))

    # 逐连杆物理参数恢复（复用 se3_param_check 的 check_params，它调 inertial_params()）
    res = check_params(model, cfg)
    s = res['summary']

    # naive se3（有零空间）对照：读它已有的 param check json
    naive = {}
    naive_json = ROOT / 'experimental_results' / 'v2_physics' / \
        f'se3_param_se3_no_uncertainty_{arm}_seed{ck.get("seed", 0)}.json'
    if naive_json.exists():
        import json
        ns = json.loads(naive_json.read_text(encoding='utf-8'))['summary']
        naive = {'mass_rel_err_mean': ns['mass_rel_err_mean'],
                 'com_err_m_mean': ns['com_err_m_mean'],
                 'I_rel_err_mean': ns['I_rel_err_mean']}

    return {'theta_err': theta_err, 'pi_id_err': pi_err, 'rank': int(rank),
            'summary': s, 'per_link': res['per_link'], 'naive': naive}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selfcheck', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--xml', default='tool/panda.xml')
    ap.add_argument('--basis', default='base_basis.npz')
    a = ap.parse_args()

    if a.selfcheck:
        r = selfcheck(a.xml, a.basis)
        ok = r['M_max_abs_err'] < r['tol']
        print(f"[reparam selfcheck] rank={r['rank']}  M(q) 最大绝对误差 {r['M_max_abs_err']:.3e} / 阈值 {r['tol']:.0e}")
        print('[reparam selfcheck] ' + ('PASS ✓ 压掉零空间不损失动力学，B/几何正确' if ok
                                        else 'FAIL ✗ 有错，勿训练'))
        sys.exit(0 if ok else 1)

    if a.smoke:
        r = forward_smoke(a.xml, a.basis)
        print(f"[reparam smoke] tau_finite={r['tau_finite']}  |tau|={r['tau_norm']:.4f}  "
              f"|dθ|={r['theta_grad_norm']:.4f}")
        ok = r['tau_finite'] and r['theta_grad_norm'] > 0
        print('[reparam smoke] ' + ('PASS ✓ jvp 前向 + θ 梯度正常' if ok
                                    else 'FAIL ✗ forward 断梯度'))
        sys.exit(0 if ok else 1)

    if a.ckpt:
        rep = report_recovery(a.ckpt, a.xml, a.basis)
        s = rep['summary']
        print(f"\n=== 重参数化后参数恢复 / {Path(a.ckpt).name} ===")
        print(f"  rank = {rep['rank']}")
        print(f"  θ 恢复误差  ‖θ̂−θ_true‖/‖θ_true‖ = {rep['theta_err']:.6f}")
        print(f"  π 投影误差 ‖Bθ̂−π_id‖/‖π_id‖  = {rep['pi_id_err']:.6f}")
        print(f"  质量相对误差均值 = {s['mass_rel_err_mean']:.6f}")
        print(f"  质心误差均值     = {s['com_err_m_mean']:.6f} m")
        print(f"  惯量相对误差均值 = {s['I_rel_err_mean']:.6f}")
        if rep['naive']:
            n = rep['naive']
            print(f"\n  对照（naive se3，有零空间）:")
            print(f"    mass_rel_err_mean  {n['mass_rel_err_mean']:.6f}  →  reparam {s['mass_rel_err_mean']:.6f}")
            print(f"    com_err_m_mean     {n['com_err_m_mean']:.6f}  →  reparam {s['com_err_m_mean']:.6f}")
            print(f"    I_rel_err_mean     {n['I_rel_err_mean']:.6f}  →  reparam {s['I_rel_err_mean']:.6f}")
        print(f"\n  {'body':<12} {'m_true':>8} {'m_learned':>8} {'m_rel':>8} {'com_m':>8} {'I_rel':>8}")
        for p in rep['per_link']:
            print(f"  {p['body']:<12} {p['mass_true']:>8.3f} {p['mass_learned']:>8.3f} "
                  f"{p['mass_rel_err']:>8.4f} {p['com_err_m']:>8.4f} {p['I_rel_err']:>8.4f}")
        return

    ap.print_help()


if __name__ == '__main__':
    main()
