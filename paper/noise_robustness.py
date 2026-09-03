"""
噪声鲁棒性 (P1.1) —— 用真实训练好的权重，在测试输入上加高斯噪声后评估。

与旧版 run_noise_robustness.py 的区别：
  * 旧版最大只加到 σ=0.02（归一化尺度的 2%），R² 从 0.9428 掉到 0.9416 ——
    这个量级根本区分不出任何方法，等于没做实验。这里扫到 σ=0.3。
  * 噪声按**每个信号自身的标准差**成比例注入，并同时报告等效的物理量
    （rad / rad·s⁻¹ / rad·s⁻²），审稿人能对上真实传感器噪声水平。
  * 每个噪声档独立重复 n_rep 次取均值，误差棒同时包含种子间与噪声实现间的方差。

    python paper/noise_robustness.py --arm franka_ex
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paper import core
from paper.core import ARMS, save_json, provenance
from paper.verify_physics import load_ckpt

ROOT = core.ROOT
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NOISE_LEVELS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3]
DEFAULT_VARIANTS = ['full', 'enc_none', 'no_physics', 'no_uncertainty', 'mlp']


def r2_global(true, pred):
    return float(1 - ((true - pred) ** 2).sum() / ((true - true.mean(0)) ** 2).sum())


def evaluate(model, q, dq, ddq, tau, sigma, rng, chunk=2048):
    scale = [q.std(0), dq.std(0), ddq.std(0)]
    qn = q + rng.normal(0, 1, q.shape).astype(np.float32) * (sigma * scale[0])
    dqn = dq + rng.normal(0, 1, dq.shape).astype(np.float32) * (sigma * scale[1])
    ddqn = ddq + rng.normal(0, 1, ddq.shape).astype(np.float32) * (sigma * scale[2])
    outs = []
    with torch.enable_grad():
        for i in range(0, len(q), chunk):
            qt = torch.tensor(qn[i:i + chunk], device=DEVICE).requires_grad_(True)
            outs.append(model.predict_phys(
                qt, torch.tensor(dqn[i:i + chunk], device=DEVICE),
                torch.tensor(ddqn[i:i + chunk], device=DEVICE)).detach().cpu().numpy())
    return r2_global(tau, np.concatenate(outs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', default='franka_ex')
    ap.add_argument('--variants', nargs='*', default=DEFAULT_VARIANTS)
    ap.add_argument('--seeds', type=int, nargs='*', default=[0, 1, 2])
    ap.add_argument('--n-rep', type=int, default=3)
    ap.add_argument('--n-samples', type=int, default=8000)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    out = a.out or f'experimental_results/v2_noise/noise_{a.arm}.json'
    results = {}
    data_ref = None

    for v in a.variants:
        cks = [ROOT / 'results_v2' / f'{v}_{a.arm}_seed{s}.pth' for s in a.seeds]
        cks = [c for c in cks if c.exists()]
        if not cks:
            print(f'[skip] {v}: 没有 checkpoint')
            continue
        curves = []
        for ck in cks:
            model, cfg, arm, data, norm, meta = load_ckpt(ck)
            if data_ref is None:
                q, dq, ddq, tau = data['test']
                idx = np.linspace(0, len(q) - 1, min(a.n_samples, len(q))).astype(int)
                data_ref = (q[idx], dq[idx], ddq[idx], tau[idx])
            q, dq, ddq, tau = data_ref
            row = []
            for sig in NOISE_LEVELS:
                reps = [evaluate(model, q, dq, ddq, tau, sig,
                                 np.random.RandomState(1000 + r))
                        for r in range(a.n_rep if sig > 0 else 1)]
                row.append(float(np.mean(reps)))
            curves.append(row)
            print(f'  {v} {ck.stem}: ' + ' '.join(f'{x:.4f}' for x in row), flush=True)
        arr = np.array(curves)
        results[v] = {'noise_levels': NOISE_LEVELS,
                      'r2_mean': arr.mean(0).tolist(),
                      'r2_std': arr.std(0, ddof=1).tolist() if len(arr) > 1 else [0.0] * len(NOISE_LEVELS),
                      'per_ckpt': arr.tolist(),
                      'n_ckpt': len(cks),
                      'relative_r2': (arr / arr[:, :1]).mean(0).tolist()}

    q, dq, ddq, tau = data_ref
    payload = {
        'arm': a.arm, 'results': results, 'noise_levels': NOISE_LEVELS,
        'noise_definition': 'σ 为各信号自身标准差的倍数，独立加到 q / dq / ddq 上',
        'signal_std': {'q_rad': q.std(0).tolist(), 'dq_rad_s': dq.std(0).tolist(),
                       'ddq_rad_s2': ddq.std(0).tolist()},
        'physical_sigma_example': {
            f'sigma={s}': {'q_rad': float((s * q.std(0)).mean()),
                           'dq_rad_s': float((s * dq.std(0)).mean()),
                           'ddq_rad_s2': float((s * ddq.std(0)).mean())}
            for s in NOISE_LEVELS if s > 0},
        'n_samples': len(q), 'n_rep': a.n_rep,
        'provenance': provenance(),
    }
    save_json(ROOT / out, payload)

    print(f'\n=== {a.arm} 噪声鲁棒性（测试集 global R²）===')
    hdr = f"{'variant':<18s}" + ''.join(f'{s:>9.2f}' for s in NOISE_LEVELS)
    print(hdr); print('-' * len(hdr))
    for v, r in results.items():
        print(f'{v:<18s}' + ''.join(f'{x:>9.4f}' for x in r['r2_mean']))
    print(f'\n保存至 {out}')


if __name__ == '__main__':
    main()
