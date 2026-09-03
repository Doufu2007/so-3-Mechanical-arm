"""
单次训练 — 一个 (variant, arm, seed) 一个进程，结果落一个 json + 一个 checkpoint。

    python paper/train.py --arm baxter --variant full --seed 0

关键规范：
  * 选模只看**验证集**，测试集仅在训练结束后用最优权重评一次
  * 记录逐 epoch 训练损失与验证 R² 曲线（用于收敛性图）
  * 结果 json 带完整溯源（git commit / GPU / 耗时 / 复现命令）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paper import core
from paper.core import ARMS, Normalizer, load_arm, metrics, provenance, save_json
from paper.models import ModelCfg, build_model, compute_loss

ROOT = core.ROOT
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------------------------------------------- 变体登记表
# 「full」= 完整模型；其余每个变体只改动一个因素，构成单因素消融矩阵。
VARIANTS = {
    'full':            ModelCfg(encoding='rotmat'),
    'enc_sincos':      ModelCfg(encoding='sincos'),
    'enc_none':        ModelCfg(encoding='none'),
    # 去掉信息瓶颈的编码变体：编码特征原样进网络，不压回 7 维
    'enc_rotmat_wide': ModelCfg(encoding='rotmat', enc_bottleneck=False),
    'enc_sincos_wide': ModelCfg(encoding='sincos', enc_bottleneck=False),
    'no_physics':      ModelCfg(encoding='rotmat', physics=False),
    'no_uncertainty':  ModelCfg(encoding='rotmat', uncertainty=False),
    'no_adaptive_lam': ModelCfg(encoding='rotmat', adaptive_lambda=False),
    'no_multiscale':   ModelCfg(encoding='rotmat', multiscale_loss=False),
    # 对偶上升约束式 λ：与 full 同编码/瓶颈，只把 λ 机制换成对偶上升（单因素消融）
    'dual_lam':        ModelCfg(encoding='rotmat', dual_ascent=True, adaptive_lambda=False),
    # 对偶上升 + 最优编码（sin/cos、无瓶颈），作为「旗舰」配置
    'dual_lam_wide':   ModelCfg(encoding='sincos', enc_bottleneck=False,
                                dual_ascent=True, adaptive_lambda=False),
    # SE(3)/SO(3) 结构化惯量（本工作的核心）：M(q)=Σ JᵀGJ，只学 10·n_body 个物理惯量参数。
    # 'se3' 与 'dual_lam_wide' 唯一差别 = 物理分支结构（SE(3) vs 任意神经网络 H），单因素消融。
    # encoding 对 se3 分支不生效（SE(3) 结构直接用物理关节角做前向运动学）。
    'se3':              ModelCfg(encoding='none', physics_kind='se3',
                                 dual_ascent=True, adaptive_lambda=False),
    # SE(3) 物理分支单独（无残差）：证明结构化惯量本身就是对的归纳偏置（80 参数 vs DeLaN 十万参数）
    'se3_no_uncertainty': ModelCfg(encoding='none', physics_kind='se3',
                                   uncertainty=False, adaptive_lambda=False),
    # 可辨识子空间重参数化：π = B·θ，参数从 K 压到 43，无零空间 → 学回真 base 参数。
    # 与 'se3' / 'se3_no_uncertainty' 唯一差别 = 参数空间（可辨识子空间 vs 全重心空间），
    # 单因素消融，回答「压掉零空间能不能让参数恢复从 ~1.0 误差变成真恢复」。
    'se3_reparam': ModelCfg(encoding='none', physics_kind='se3_reparam',
                            dual_ascent=True, adaptive_lambda=False),
    'se3_reparam_no_uncertainty': ModelCfg(encoding='none', physics_kind='se3_reparam',
                                           uncertainty=False, adaptive_lambda=False),
    'mlp':             ModelCfg(kind='mlp'),
    'hnn':             ModelCfg(kind='hnn'),
    'lnn':             ModelCfg(kind='lnn'),
    'symoden':         ModelCfg(kind='symoden'),
}

DEFAULTS = dict(epochs=300, batch_size=1024, lr=1e-3, weight_decay=1e-5,
                grad_clip=1.0, eval_every=10)


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_t(x):
    return torch.from_numpy(np.ascontiguousarray(x)).float().to(DEVICE)


@torch.enable_grad()
def predict(model, q, dq, ddq, chunk=2048, want_terms=False):
    """分块前向（物理分支需要 autograd，不能 no_grad），返回归一化空间的预测。"""
    model.eval()
    outs, terms = [], []
    for i in range(0, len(q), chunk):
        p, c = model(q[i:i + chunk].clone().requires_grad_(True),
                     dq[i:i + chunk], ddq[i:i + chunk], want_terms=want_terms)
        outs.append(p.detach())
        if want_terms:
            terms.append({k: v.detach() for k, v in c.items() if torch.is_tensor(v)})
    pred = torch.cat(outs)
    if want_terms:
        merged = {k: torch.cat([t[k] for t in terms]) for k in terms[0]}
        return pred, merged
    del outs
    return pred, {}


def r2_norm(pred, true):
    """归一化空间的**全局** R²（残差平方和 / 总平方和，全部关节合并统计）。

    用作选模准则。不用逐关节平均：Franka 的 J1/J7 扭矩方差接近 0，逐关节 R²
    会在 ±10² 量级剧烈跳动，用它选模等于按噪声选模。
    """
    ss_res = ((true - pred) ** 2).sum()
    ss_tot = ((true - true.mean(0)) ** 2).sum()
    return float(1 - ss_res / ss_tot.clamp_min(1e-12))


def run(arm, variant, seed, epochs=None, frac=1.0, out_dir='experimental_results/v2',
        tag=None, batch_size=None, lr=None, hidden_scale=1.0, lambda_reg=None,
        dual_delta_frac=None, dual_lr=None, dual_lambda_init=None,
        save_ckpt=True):
    cfg = VARIANTS[variant]
    hp = dict(DEFAULTS)
    if epochs:
        hp['epochs'] = epochs
    if batch_size:
        hp['batch_size'] = batch_size
    if lr:
        hp['lr'] = lr
    overrides = {}
    if hidden_scale != 1.0:
        overrides['delan_hidden'] = tuple(int(h * hidden_scale) for h in cfg.delan_hidden)
        overrides['unc_hidden'] = tuple(int(h * hidden_scale) for h in cfg.unc_hidden)
    if lambda_reg is not None:
        overrides['lambda_reg'] = lambda_reg
    if dual_delta_frac is not None:
        overrides['dual_delta_frac'] = dual_delta_frac
    if dual_lr is not None:
        overrides['dual_lr'] = dual_lr
    if dual_lambda_init is not None:
        overrides['dual_lambda_init'] = dual_lambda_init
    if overrides:
        cfg = ModelCfg(**{**cfg.to_dict(), **overrides})

    spec = ARMS[arm]
    data = load_arm(arm)
    q_tr, dq_tr, ddq_tr, tau_tr = data['train']

    if frac < 1.0:
        rng = np.random.RandomState(seed)
        idx = np.sort(rng.choice(len(q_tr), max(64, int(len(q_tr) * frac)), replace=False))
        q_tr, dq_tr, ddq_tr, tau_tr = q_tr[idx], dq_tr[idx], ddq_tr[idx], tau_tr[idx]

    norm = Normalizer(q_tr, dq_tr, ddq_tr, tau_tr)
    set_seed(seed)

    def prep(split):
        """q/dq/ddq 保持物理单位喂进模型（模型内部自己做 z-score）；τ 归一化。"""
        q, dq, ddq, tau = split
        return to_t(q), to_t(dq), to_t(ddq), to_t(norm.norm_tau(tau))

    Xtr = prep((q_tr, dq_tr, ddq_tr, tau_tr))
    Xva = prep(data['val'])
    Xte = prep(data['test'])

    model = build_model(cfg, 7, norm, axes=spec.axes).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*Xtr), batch_size=hp['batch_size'], shuffle=True,
        drop_last=len(Xtr[0]) > hp['batch_size'])
    opt = torch.optim.AdamW(model.parameters(), lr=hp['lr'], weight_decay=hp['weight_decay'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=hp['epochs'])

    curve = {'epoch': [], 'train_loss': [], 'data_loss': [], 'reg_loss': [],
             'lambda_mean': [], 'residual_sq_mean': [], 'val_epoch': [], 'val_r2': [],
             'train_r2': []}
    best = {'val_r2': -1e9, 'epoch': -1, 'state': None}
    t0 = time.time()

    # 对偶上升：λ 是对偶变量，独立于 Adam 更新（Adam 只更新模型参数 θ）。
    # λ ← max(0, λ + η_λ·(‖ε‖² − δ))，沿约束违反方向做投影梯度上升 ⇒ 不会塌缩到 0。
    dual = cfg.dual_ascent
    lam_dual = float(cfg.dual_lambda_init) if dual else 0.0

    for ep in range(1, hp['epochs'] + 1):
        model.train()
        agg = {'total': 0.0, 'data_loss': 0.0, 'reg_loss': 0.0, 'lambda_mean': 0.0,
               'residual_sq_mean': 0.0}
        for qb, dqb, ddqb, taub in loader:
            opt.zero_grad(set_to_none=True)
            pred, comps = model(qb.clone().requires_grad_(True), dqb, ddqb)
            loss, parts = compute_loss(pred, taub, comps, cfg,
                                       lam_dual=lam_dual if dual else None)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), hp['grad_clip'])
            opt.step()
            if dual:
                lam_dual = max(0.0, lam_dual + cfg.dual_lr *
                               (parts['residual_sq_mean'] - parts['dual_delta']))
            agg['total'] += float(loss)
            for k in ['data_loss', 'reg_loss', 'lambda_mean', 'residual_sq_mean']:
                agg[k] += parts[k]
        sched.step()
        nb = len(loader)
        curve['epoch'].append(ep)
        curve['train_loss'].append(agg['total'] / nb)
        curve['data_loss'].append(agg['data_loss'] / nb)
        curve['reg_loss'].append(agg['reg_loss'] / nb)
        curve['lambda_mean'].append(agg['lambda_mean'] / nb)
        curve['residual_sq_mean'].append(agg['residual_sq_mean'] / nb)

        if ep % hp['eval_every'] == 0 or ep == hp['epochs'] or ep == 1:
            vp, _ = predict(model, Xva[0], Xva[1], Xva[2])
            vr2 = r2_norm(vp, Xva[3])
            tp, _ = predict(model, Xtr[0][:20000], Xtr[1][:20000], Xtr[2][:20000])
            tr2 = r2_norm(tp, Xtr[3][:20000])
            curve['val_epoch'].append(ep)
            curve['val_r2'].append(vr2)
            curve['train_r2'].append(tr2)
            if vr2 > best['val_r2']:
                best.update(val_r2=vr2, epoch=ep,
                            state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            print(f'  ep{ep:4d} loss={agg["total"]/nb:.5f} train_r2={tr2:.4f} '
                  f'val_r2={vr2:.4f} best={best["val_r2"]:.4f} {time.time()-t0:.0f}s', flush=True)

    train_time = time.time() - t0
    model.load_state_dict(best['state'])

    # -------- 最终评估：验证集 + 测试集，物理单位下的逐关节指标 --------
    def full_eval(X, raw_tau):
        pred_n, _ = predict(model, X[0], X[1], X[2])
        pred = norm.denorm_tau(pred_n.cpu().numpy())
        return metrics(raw_tau, pred, spec.joint_names)

    m_val = full_eval(Xva, data['val'][3])
    m_test = full_eval(Xte, data['test'][3])

    name = tag or f'{variant}_{arm}_seed{seed}' + (f'_frac{frac:g}' if frac < 1.0 else '')
    result = {
        'variant': variant, 'arm': arm, 'seed': seed, 'frac': frac,
        'cfg': cfg.to_dict(), 'hp': hp, 'n_params': n_params,
        'n_train': int(len(q_tr)), 'n_val': int(len(data['val'][0])), 'n_test': int(len(data['test'][0])),
        'split_manifest': data['manifest'],
        'best_epoch': best['epoch'], 'val_r2_norm': best['val_r2'],
        'val': m_val, 'test': m_test,
        'train_time_s': train_time, 'curve': curve,
        'normalizer': norm.state_dict(),
        'provenance': provenance(),
    }
    if dual:
        result['dual_final_lambda'] = lam_dual
        result['dual_delta'] = float(cfg.dual_delta_frac ** 2 *
                                     float((Xtr[3] ** 2).mean()))
        result['dual_note'] = ('对偶上升约束式 λ：残差预算 δ = dual_delta_frac²·mean(‖τ‖²)，'
                               'λ 收敛到一个非零值，说明物理分支保持活跃，未塌缩。')
    out_path = ROOT / out_dir / f'{name}.json'
    save_json(out_path, result)

    if save_ckpt:
        ck_dir = ROOT / 'results_v2'
        ck_dir.mkdir(exist_ok=True)
        torch.save({'state_dict': best['state'], 'cfg': cfg.to_dict(), 'variant': variant,
                    'arm': arm, 'seed': seed, 'frac': frac,
                    'normalizer': norm.state_dict(), 'val_r2': best['val_r2'],
                    'test_r2_all': m_test['r2_all_joints']}, ck_dir / f'{name}.pth')

    print(f'[done] {name}: val_r2={best["val_r2"]:.4f} '
          f'test_r2_all={m_test["r2_all_joints"]:.4f} '
          f'test_r2_active={m_test["r2_active_joints"]:.4f} '
          f'({train_time/60:.1f} min)', flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, choices=list(ARMS))
    ap.add_argument('--variant', required=True, choices=list(VARIANTS))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--frac', type=float, default=1.0)
    ap.add_argument('--out-dir', default='experimental_results/v2')
    ap.add_argument('--tag', default=None)
    ap.add_argument('--batch-size', type=int, default=None)
    ap.add_argument('--lr', type=float, default=None)
    ap.add_argument('--hidden-scale', type=float, default=1.0)
    ap.add_argument('--lambda-reg', type=float, default=None)
    ap.add_argument('--dual-delta-frac', type=float, default=None)
    ap.add_argument('--dual-lr', type=float, default=None)
    ap.add_argument('--dual-lambda-init', type=float, default=None)
    ap.add_argument('--skip-existing', action='store_true')
    a = ap.parse_args()

    name = a.tag or f'{a.variant}_{a.arm}_seed{a.seed}' + (f'_frac{a.frac:g}' if a.frac < 1.0 else '')
    if a.skip_existing and (ROOT / a.out_dir / f'{name}.json').exists():
        print(f'[skip] {name} 已存在')
        return
    run(a.arm, a.variant, a.seed, epochs=a.epochs, frac=a.frac, out_dir=a.out_dir,
        tag=a.tag, batch_size=a.batch_size, lr=a.lr, hidden_scale=a.hidden_scale,
        lambda_reg=a.lambda_reg, dual_delta_frac=a.dual_delta_frac,
        dual_lr=a.dual_lr, dual_lambda_init=a.dual_lambda_init)


if __name__ == '__main__':
    main()
