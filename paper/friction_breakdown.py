"""逐关节「摩擦 / 刚体力矩」比 —— §4.7 那张表的数据来源。

论文要解释的现象：加摩擦后纯结构分支的全局 R² 只从 1.0000 掉到 0.9871，
而逐关节 R² 在 J5 掉到 0.320、J7 掉到 −0.008。

解释是逐关节的信噪比：`Dataset_Franka_Fric` 里的 tau 是刚体力矩 + 解析摩擦，
而摩擦模型的系数 FC/FV 是已知的（`make_dataset_fric.py`），所以可以精确地把两部分
拆开，算出每个关节上「刚体模型表示不了的那部分」占多大。

J7 的刚体力矩 RMS 只有 7.5 mN·m，而它的摩擦 RMS 是 0.129 N·m —— 17 倍。刚体模型
在那个关节上根本没有可拟合的信号；同时 J7 只占全臂力矩能量的 0.003%，所以这个
彻底失败对全局 R² 几乎没有代价。这就是聚合指标掩盖失败的机制。

    python paper/friction_breakdown.py --dataset Dataset_Franka_Fric --n-files 12
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

# 与 make_dataset_fric.py 保持一致；**不要在这里调整**，这是数据生成时的真值
FC = np.array([1.00, 1.00, 0.70, 0.50, 0.30, 0.15, 0.08])   # 库仑, N·m
FV = np.array([0.60, 0.60, 0.40, 0.30, 0.20, 0.10, 0.05])   # 粘滞, N·m·s/rad
V_EPS = 0.01


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='Dataset_Franka_Fric')
    ap.add_argument('--split', default='Training')
    ap.add_argument('--n-files', type=int, default=12)
    ap.add_argument('--json', default=None)
    a = ap.parse_args()

    files = sorted(glob.glob(f'{a.dataset}/{a.split}/*.csv'))[:a.n_files]
    if not files:
        raise SystemExit(f'找不到数据：{a.dataset}/{a.split}/*.csv')
    A = np.concatenate([np.loadtxt(f, skiprows=1) for f in files])
    n = A.shape[1] // 4
    dq, tau = A[:, n:2 * n], A[:, 3 * n:4 * n]

    fric = FC * np.tanh(dq / V_EPS) + FV * dq
    rigid = tau - fric

    def rms(x):
        return np.sqrt((x ** 2).mean(0))

    r_r, r_f = rms(rigid), rms(fric)
    ratio = r_f / r_r
    energy = (tau ** 2).mean(0)
    share = energy / energy.sum()

    print(f'{a.dataset}/{a.split}：{len(files)} 条轨迹，{len(A)} 个样本\n')
    hdr = f'{"":26s}' + ''.join(f'{f"J{i+1}":>9s}' for i in range(n))
    print(hdr)
    for name, row, fmt in (('刚体力矩 RMS (N·m)', r_r, '9.4f'),
                           ('摩擦 RMS (N·m)', r_f, '9.4f'),
                           ('摩擦 / 刚体', ratio, '9.2f'),
                           ('占全臂力矩能量', share * 100, '8.3f')):
        suffix = '%' if '能量' in name else ''
        print(f'{name:24s}' + ''.join(f'{v:{fmt}}{suffix}' for v in row))

    overall = np.sqrt((fric ** 2).mean()) / np.sqrt((tau ** 2).mean())
    print(f'\n全局 摩擦/tau RMS = {overall:.4f}')
    worst = int(np.argmax(ratio))
    print(f'最差关节 J{worst+1}：摩擦是刚体力矩的 {ratio[worst]:.1f} 倍，'
          f'而它只占力矩能量的 {share[worst]*100:.3f}% —— '
          f'刚体模型在那里彻底失败，全局 R² 却几乎不受影响。')

    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(dict(
            dataset=a.dataset, split=a.split, n_files=len(files), n_samples=len(A),
            fc=FC.tolist(), fv=FV.tolist(), v_eps=V_EPS,
            rigid_rms=r_r.tolist(), fric_rms=r_f.tolist(),
            fric_over_rigid=ratio.tolist(),
            torque_energy_share=share.tolist(),
            overall_fric_over_tau_rms=float(overall)), indent=2), encoding='utf-8')
        print(f'\n写入 {a.json}')


if __name__ == '__main__':
    main()
