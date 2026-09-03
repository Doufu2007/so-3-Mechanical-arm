"""
生成激励充分的 Franka 数据集 (Dataset_Franka_Ex)。

为什么要重做数据：原 Dataset_Franka 的 15 条轨迹全部以 q=0 为中心、|q̇|≤0.85 rad/s，
在 7 维位形空间里只是 15 条一维曲线。实测：训练点彼此最近邻距离中位数 0.002，
而测试点到训练集的最近邻距离中位数 0.262（相差两个数量级）—— 任何全局动力学
模型在测试集上都是纯外推，物理结构再正确也救不回来（实测纯 DeLaN 训练 R²=0.996、
验证 R²=-77）。而且速度太低，H·q̈ 与科里奥利项被重力项完全淹没，"学动力学"
退化成"学重力"。

本脚本用机器人动力学参数辨识里的标准做法 —— 有限傅里叶级数激励轨迹
(Swevers et al., 1997) —— 重新生成：
  * 每条轨迹的中心位形 q0 在关节量程内随机，覆盖位形空间
  * 5 阶谐波叠加，速度提到 2~3 rad/s 量级，让惯性项和科里奥利项真正起作用
  * q̇ / q̈ 由级数解析求导得到（不是有限差分，无数值噪声）
  * 力矩标签用 mujoco.mj_inverse 精确计算

    python paper/make_franka_dataset.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paper import core

ROOT = core.ROOT
# MuJoCo 的 from_xml_path 走自有 VFS，绝对 Windows 路径（含中文）会 ValueError
# （ParseXML: Error opening file）。用相对路径，脚本须在项目根目录下运行
# （与 identifiability.py / base_reparam.py 同一约定）。
XML = 'tool/panda.xml'

N_JOINTS = 7
Q_LIMIT = 1.4          # rad，留出余量（xml 里是 ±2.9）
DQ_LIMIT = 3.0         # rad/s
N_HARM = 5


def fourier_traj(rng, duration, dt, f0):
    """q(t) = q0 + Σ_k [a_k/(ω k) sin(ωkt) − b_k/(ωk) cos(ωkt)]，解析给出 q̇、q̈。

    这种参数化天然周期、光滑，且首尾速度/加速度连续 —— 辨识激励轨迹的标准形式。
    """
    t = np.arange(0, duration, dt)
    w = 2 * np.pi * f0
    q = np.zeros((len(t), N_JOINTS))
    dq = np.zeros_like(q)
    ddq = np.zeros_like(q)
    for j in range(N_JOINTS):
        q0 = rng.uniform(-0.9 * Q_LIMIT, 0.9 * Q_LIMIT)
        a = rng.uniform(-1, 1, N_HARM)
        b = rng.uniform(-1, 1, N_HARM)
        qj = np.full(len(t), q0, dtype=float)
        dqj = np.zeros(len(t))
        ddqj = np.zeros(len(t))
        for k in range(1, N_HARM + 1):
            wk = w * k
            qj += a[k - 1] / wk * np.sin(wk * t) - b[k - 1] / wk * np.cos(wk * t)
            dqj += a[k - 1] * np.cos(wk * t) + b[k - 1] * np.sin(wk * t)
            ddqj += -a[k - 1] * wk * np.sin(wk * t) + b[k - 1] * wk * np.cos(wk * t)
        # 缩放到限幅内：位置贴着量程，速度不超过 DQ_LIMIT
        s_pos = Q_LIMIT / max(np.abs(qj).max(), 1e-9)
        s_vel = DQ_LIMIT / max(np.abs(dqj).max(), 1e-9)
        s = min(1.0, s_pos, s_vel)
        q[:, j], dq[:, j], ddq[:, j] = qj * s, dqj * s, ddqj * s
    return q, dq, ddq


def label_with_mujoco(q, dq, ddq, gravity=None):
    import mujoco
    m = mujoco.MjModel.from_xml_path(XML)
    if gravity is not None:
        m.opt.gravity[:] = gravity
    d = mujoco.MjData(m)
    tau = np.zeros_like(q)
    for i in range(len(q)):
        d.qpos[:] = q[i]
        d.qvel[:] = dq[i]
        d.qacc[:] = ddq[i]
        mujoco.mj_inverse(m, d)
        tau[i] = d.qfrc_inverse
    return tau


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='Dataset_Franka_Ex')
    ap.add_argument('--n-train', type=int, default=40)
    ap.add_argument('--n-val', type=int, default=8)
    ap.add_argument('--n-test', type=int, default=8)
    ap.add_argument('--duration', type=float, default=10.0)
    ap.add_argument('--dt', type=float, default=0.005)
    ap.add_argument('--seed', type=int, default=2024)
    ap.add_argument('--tilt-deg', type=float, default=0.0,
                    help='基座绕 tilt-axis 转多少度（0 = 水平基座，重力默认 (0,0,-9.81)）')
    ap.add_argument('--tilt-axis', default='x', choices=['x', 'y', 'z'])
    a = ap.parse_args()

    # 基座重定向：0 度用 XML 默认重力（水平）；否则旋转重力方向（等价于旋转基座）。
    gravity = core.rotated_gravity(a.tilt_deg, a.tilt_axis) if a.tilt_deg else None

    out = ROOT / a.out
    (out / 'Training').mkdir(parents=True, exist_ok=True)
    (out / 'Validation').mkdir(parents=True, exist_ok=True)
    (out / 'Test').mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(a.seed)
    header = (' '.join(f'q{i}' for i in range(7)) + ' ' +
              ' '.join(f'dq{i}' for i in range(7)) + ' ' +
              ' '.join(f'ddq{i}' for i in range(7)) + ' ' +
              ' '.join(f'tau{i}' for i in range(7)))

    # 文件名必须带 split 前缀：core.load_arm 用 basename 判定 train/test 是否重叠，
    # 三个目录都叫 ex_000.csv 会被误判成数据泄漏而直接抛错。前缀与产出论文数字的
    # 那份数据集一致（tr_/va_/te_），改名后归一化器与已有 checkpoint 逐位相同。
    PREFIX = {'Training': 'tr', 'Validation': 'va', 'Test': 'te'}

    stats = {}
    for split, n in [('Training', a.n_train), ('Validation', a.n_val), ('Test', a.n_test)]:
        allq, alldq, allddq, alltau = [], [], [], []
        for i in range(n):
            f0 = rng.uniform(0.15, 0.4)
            q, dq, ddq = fourier_traj(rng, a.duration, a.dt, f0)
            tau = label_with_mujoco(q, dq, ddq, gravity)
            arr = np.column_stack([q, dq, ddq, tau])
            np.savetxt(out / split / f'{PREFIX[split]}_ex_{i:03d}.csv', arr, delimiter=' ',
                       header=header, comments='', fmt='%.8f')
            allq.append(q); alldq.append(dq); allddq.append(ddq); alltau.append(tau)
        Q, DQ, DDQ, TAU = (np.concatenate(x) for x in [allq, alldq, allddq, alltau])
        stats[split] = dict(n=len(Q),
                            q_range=np.round(Q.max(0) - Q.min(0), 2).tolist(),
                            dq_p99=np.round(np.percentile(np.abs(DQ), 99, axis=0), 2).tolist(),
                            ddq_p99=np.round(np.percentile(np.abs(DDQ), 99, axis=0), 2).tolist(),
                            tau_std=np.round(TAU.std(0), 4).tolist())
        print(f'{split}: {n} 条轨迹, {len(Q)} 样本')
        for k, v in stats[split].items():
            if k != 'n':
                print(f'    {k:9s} {v}')

    # 惯性/科里奥利项相对重力的占比 —— 证明这套数据真的在考动力学而不是只考重力
    import mujoco
    m = mujoco.MjModel.from_xml_path(XML)
    if gravity is not None:
        m.opt.gravity[:] = gravity
    d = mujoco.MjData(m)
    files = sorted((out / 'Training').glob('*.csv'))
    arr = np.loadtxt(files[0], skiprows=1)
    q, dq, ddq, tau = arr[:, :7], arr[:, 7:14], arr[:, 14:21], arr[:, 21:28]
    grav = np.zeros_like(tau)
    for i in range(len(q)):
        d.qpos[:] = q[i]; d.qvel[:] = 0; d.qacc[:] = 0
        mujoco.mj_inverse(m, d); grav[i] = d.qfrc_inverse
    dyn = tau - grav
    print(f'\n动力学项 (τ − g) RMS = {np.sqrt((dyn**2).mean()):.4f} N·m，'
          f'重力项 RMS = {np.sqrt((grav**2).mean()):.4f} N·m，'
          f'占比 = {np.sqrt((dyn**2).mean())/np.sqrt((grav**2).mean()):.2f}')
    core.save_json(out / 'dataset_stats.json',
                   {'stats': stats, 'args': vars(a),
                    'gravity': (None if gravity is None else gravity.tolist()),
                    'dyn_over_grav_rms': float(np.sqrt((dyn**2).mean()) / np.sqrt((grav**2).mean())),
                    'provenance': core.provenance()})
    print(f'\n数据集写入 {out}')


if __name__ == '__main__':
    main()
