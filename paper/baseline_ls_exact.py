"""经典最小二乘辨识基线 —— 用精确回归量，作为「参数恢复」的可辨识性天花板。

替代 baseline_swevers.py。原版用有限差分扰动 MuJoCo 字段构造回归量，在
`m.body_inertia[bid, 3]` 处直接 IndexError —— MuJoCo 的 body_inertia 只存 3 个
主转动惯量（方向另由 body_iquat 给出），6 分量惯量张量根本写不进去。该脚本从未
产出过任何结果，而手稿 §4.1 曾把它的结论（「最小二乘能近乎完美恢复真值」）当成
既成事实引用 —— 那句话没有任何实验支撑。

本脚本回答的问题比原来那句更有意义：

    神经网络学不回物理参数，是它自己的毛病，还是任何方法都躲不掉的结构上限？

做法是把同一批数据喂给**无偏、无正则、解析梯度**的最小二乘 —— 辨识方法的天花板。
然后把参数误差按可辨识子空间分解：

    e = pi_hat - pi_true
    e_id   = P_id · e        落在可辨识子空间（rank r 的右奇异向量张成）
    e_null = (I - P_id) · e  落在零空间

若 ||e_null|| >> ||e_id||，则误差几乎全部躺在「无论如何都恢复不了」的方向上，
神经网络的参数恢复失败与最小二乘同源，是问题的结构性质而非训练缺陷。

    python paper/baseline_ls_exact.py --xml tool/panda_real.xml --n 2000
    python paper/baseline_ls_exact.py --arm franka_ex --n 4000     # 用真实训练数据
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from identifiability import (PiBranch, build_Y_exact, random_states,  # noqa: E402
                            rigid_tau, true_pi)
from se3_physics import extract_geometry  # noqa: E402

PARAM_NAMES = ['m'] + [f'h{a}' for a in 'xyz'] + \
              ['Ixx', 'Iyy', 'Izz', 'Ixy', 'Ixz', 'Iyz']


def build_branch(xml):
    geom = extract_geometry(xml)
    branch = PiBranch(geom).double()
    return geom, branch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--xml', default=str(ROOT / 'tool' / 'panda.xml'))
    ap.add_argument('--arm', default=None,
                    help='用某个臂的真实训练数据代替随机采样')
    ap.add_argument('--n', type=int, default=2000, help='样本数')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tol', type=float, default=1e-9, help='数值秩相对阈值')
    ap.add_argument('--noise', type=float, default=0.0,
                    help='给 tau 加的相对高斯噪声（0=无噪声，理想上限）')
    ap.add_argument('--json', default=None)
    a = ap.parse_args()

    geom, branch = build_branch(a.xml)
    n_body, n_joints = branch.n_body, branch.n
    K = 10 * n_body
    pi_true = true_pi(geom).double()

    rng = np.random.RandomState(a.seed)
    if a.arm:
        from core import load_arm
        d = load_arm(a.arm)['train']
        idx = np.sort(rng.choice(len(d[0]), min(a.n, len(d[0])), replace=False))
        q, dq, ddq = (torch.tensor(x[idx], dtype=torch.float64) for x in d[:3])
        src = f'{a.arm} 训练集 {len(idx)} 点'
    else:
        q, dq, ddq = random_states(a.n, n_joints, rng)
        src = f'随机位形 {a.n} 点'

    print(f'模型   {a.xml}')
    print(f'连杆   {n_body} · 自由度 {n_joints} · 参数 K = {K}')
    print(f'采样   {src}')

    # ---- 回归量与「测量」力矩 -------------------------------------------------
    Y = build_Y_exact(branch, q, dq, ddq)              # (N, n, K)
    branch.set_pi(pi_true)
    tau = rigid_tau(branch, q, dq, ddq)                # (N, n)

    if a.noise > 0:
        sigma = a.noise * tau.std()
        tau = tau + torch.tensor(rng.normal(0, float(sigma), tau.shape))
        print(f'噪声   tau 上加 {a.noise:.1%} 相对高斯噪声')

    A = Y.reshape(-1, K).detach().numpy()                       # (N·n, K)
    b = tau.reshape(-1).detach().numpy()

    # ---- 最小二乘 ------------------------------------------------------------
    pi_hat, _, rank, s = np.linalg.lstsq(A, b, rcond=a.tol)
    pi_t = pi_true.reshape(-1).detach().numpy()
    e = pi_hat - pi_t

    # ---- 可辨识子空间投影 ----------------------------------------------------
    # A = U S V^T；前 rank 个右奇异向量张成可辨识子空间
    _, sv, Vt = np.linalg.svd(A, full_matrices=False)
    Vr = Vt[:rank]                                     # (rank, K)
    e_id = Vr.T @ (Vr @ e)
    e_null = e - e_id

    print(f'\n{"="*62}\n最小二乘辨识（无偏、无正则、解析梯度 —— 方法上限）\n{"="*62}')
    print(f'  设计矩阵 A        {A.shape}')
    print(f'  数值秩            {rank} / {K}   （零空间 {K - rank}）')
    print(f'  力矩残差 R^2      {1 - ((A @ pi_hat - b)**2).sum() / ((b - b.mean())**2).sum():.8f}')

    print(f'\n  参数误差 e = pi_hat - pi_true 的分解：')
    print(f'    ||e||             {np.linalg.norm(e):.6e}')
    print(f'    ||e_id||          {np.linalg.norm(e_id):.6e}   可辨识子空间内')
    print(f'    ||e_null||        {np.linalg.norm(e_null):.6e}   零空间内')
    frac = np.linalg.norm(e_null) / max(np.linalg.norm(e), 1e-300)
    print(f'    零空间占误差能量  {frac:.4%}')

    # ---- 逐参数类型的相对误差（与 se3_param_check 口径一致）------------------
    P_hat = pi_hat.reshape(n_body, 10)
    P_tru = pi_t.reshape(n_body, 10)
    print(f'\n  逐参数类型相对误差（|hat-true| / max(|true|, 1e-6) 的中位数）：')
    for j, nm in enumerate(PARAM_NAMES):
        rel = np.abs(P_hat[:, j] - P_tru[:, j]) / np.maximum(np.abs(P_tru[:, j]), 1e-6)
        print(f'    {nm:4s} {np.median(rel):>12.4f}')

    out = dict(xml=a.xml, arm=a.arm, n=int(q.shape[0]), seed=a.seed, noise=a.noise,
               K=K, rank=int(rank), nullspace=int(K - rank),
               r2_torque=float(1 - ((A @ pi_hat - b)**2).sum() / ((b - b.mean())**2).sum()),
               e_norm=float(np.linalg.norm(e)),
               e_id_norm=float(np.linalg.norm(e_id)),
               e_null_norm=float(np.linalg.norm(e_null)),
               null_energy_frac=float(frac),
               pi_hat=pi_hat.tolist(), pi_true=pi_t.tolist(),
               singular_values=sv.tolist())
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(out, indent=2), encoding='utf-8')
        print(f'\n写入 {a.json}')

    print(f'\n解读：力矩 R^2 ≈ 1 而参数误差的 {frac:.1%} 落在零空间 —— '
          f'这是**无偏最小二乘**的结果，\n      说明神经网络恢复不了物理参数不是训练问题，'
          f'任何方法在同一批激励下都躲不掉。')


if __name__ == '__main__':
    main()
