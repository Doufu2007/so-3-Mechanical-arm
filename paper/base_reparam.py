"""计算惯性参数的可辨识子空间基 B，用于把 SE(3) 分支重参数化到 43 维 base 集。

动机：SE(3) 分支用 K=70（或 80）个重心参数 pi，但对力矩只有 43 个方向可辨识，
其余落在回归量零空间里——这是 audit 论文的负结果。本脚本算出可辨识子空间的
正交基 B ∈ R^{K×43}，下一步把 PiBranch 的参数从 K 维 pi 换成 43 维 θ（pi = B·θ），
从而让模型没有零空间、学回的就是真 base 参数。

用法：
    python paper/base_reparam.py --xml tool/panda.xml --n 1500 --out base_basis.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from se3_physics import extract_geometry          # noqa: E402
from identifiability import PiBranch, build_Y_exact, true_pi, random_states  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--xml', default='tool/panda.xml')
    ap.add_argument('--n', type=int, default=1500)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tol', type=float, default=1e-9)
    ap.add_argument('--out', default='base_basis.npz')
    a = ap.parse_args()

    geom = extract_geometry(a.xml)
    n_body, n_joints = len(geom['mass']), int(geom['n_joints'])
    K = 10 * n_body

    branch = PiBranch(geom).double()
    branch.eval()
    pi0 = true_pi(geom)

    rng = np.random.RandomState(a.seed)
    q, dq, ddq = random_states(a.n, n_joints, rng)

    print(f'模型 {a.xml} · 连杆 {n_body} · 参数 K={K} · 采样 {a.n} 点')
    Y = build_Y_exact(branch, q, dq, ddq)          # (N, n, K)
    A = Y.reshape(-1, K).numpy()
    scale = max(np.abs(A).max(), 1e-30)
    A = A / scale
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    cutoff = s[0] * a.tol
    rank = int((s > cutoff).sum())

    B = Vt[:rank].T          # (K, rank) 可辨识子空间正交基（列向量）
    Nnull = Vt[rank:]        # (K-rank, K) 零空间基（行向量）

    pi_true_np = pi0.detach().numpy().reshape(-1)
    pi_id = B @ (B.T @ pi_true_np)        # 可辨识分量
    pi_null = pi_true_np - pi_id          # 零空间分量（永远学不到）

    orth_err = np.abs(B.T @ B - np.eye(rank)).max()
    rec_err = np.abs(pi_null - Nnull.T @ (Nnull @ pi_null)).max()

    print(f'秩 rank = {rank} · 零空间维数 = {K - rank}')
    print(f'自检 B 正交            max|B^T B - I| = {orth_err:.3e}')
    print(f'自检 零空间重构         max|e - N^T N e| = {rec_err:.3e}')
    print(f'pi_true 可辨识分量范数   {np.linalg.norm(pi_id):.4f}')
    print(f'pi_true 零空间分量范数   {np.linalg.norm(pi_null):.4f}  <- 这部分永远学不到')

    np.savez(a.out, B=B, rank=rank, null_basis=Nnull, pi_true=pi_true_np,
             pi_identifiable=pi_id, pi_null=pi_null, K=K)
    print(f'\n写入 {a.out}  (B {B.shape}, rank {rank})')


if __name__ == '__main__':
    main()
