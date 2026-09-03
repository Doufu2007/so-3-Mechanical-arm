"""重力子空间是否含于动力学可辨识子空间？

论文要解释的一对表面矛盾：

  · 参数恢复失败 —— 70 个惯性参数里 27 个方向结构上不可辨识，学到的质量偏 72~92%、
    惯量偏 97~480%（se3_param_check）。
  · 闭环重力补偿却成功 —— 稳态误差 0.46 mrad，比最好基线好 18 倍（v2_control）。

若把「重力可辨识子空间 G」定义为静态回归量 Y_static（dq=ddq=0）的行空间，
「动力学可辨识子空间 D」定义为完整回归量 Y_full 的行空间，则只要

    G ⊆ D

两件事就不矛盾：模型从动态数据里能确定的信息，已经足以唯一确定 g(q)，
哪怕它确定不了单个物理参数。

用主角度检验包含关系：G 的每个基向量投影到 D 上，残差 ||(I-P_D) v|| 应为机器零。
反向（D ⊆ G）应当**不**成立 —— 否则说明静态数据就够了，动态激励毫无意义。

    python paper/gravity_subspace.py --xml tool/panda_real.xml --n 200
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

from identifiability import (PiBranch, build_Y_exact, random_states)  # noqa: E402
from se3_physics import extract_geometry  # noqa: E402


def rowspace(A, tol_rel=1e-9):
    """返回 (V_r, rank)：A 行空间的标准正交基（K 维空间里的 rank 个方向）。"""
    _, s, Vt = np.linalg.svd(A, full_matrices=False)
    rank = int((s > tol_rel * s[0]).sum())
    return Vt[:rank].T, rank, s          # V_r: (K, rank)


def containment(Vsub, Vsup):
    """Vsub 的列空间落在 Vsup 列空间外的最大残差（0 = 完全包含）。"""
    P = Vsup @ Vsup.T
    R = Vsub - P @ Vsub
    return float(np.linalg.norm(R, axis=0).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--xml', default=str(ROOT / 'tool' / 'panda_real.xml'))
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tol', type=float, default=1e-9)
    ap.add_argument('--json', default=None)
    a = ap.parse_args()

    geom = extract_geometry(a.xml)
    branch = PiBranch(geom).double()
    K = 10 * branch.n_body
    rng = np.random.RandomState(a.seed)

    q, dq, ddq = random_states(a.n, branch.n, rng)
    Y_full = build_Y_exact(branch, q, dq, ddq).reshape(-1, K).detach().numpy()

    qs, _, _ = random_states(a.n, branch.n, np.random.RandomState(a.seed + 1))
    z = torch.zeros_like(qs)
    Y_stat = build_Y_exact(branch, qs, z, z).reshape(-1, K).detach().numpy()

    Vd, rd, sd = rowspace(Y_full, a.tol)
    Vg, rg, sg = rowspace(Y_stat, a.tol)

    print(f'模型 {a.xml}')
    print(f'参数总数 K = {K}\n')
    print(f'  动力学可辨识子空间 D   dim = {rd}')
    print(f'  重力可辨识子空间   G   dim = {rg}')

    g_in_d = containment(Vg, Vd)
    d_in_g = containment(Vd, Vg)

    print(f'\n包含关系检验（残差，0 = 完全包含）：')
    print(f'  G ⊆ D ?   max||(I−P_D)·v|| = {g_in_d:.3e}   '
          f'{"是" if g_in_d < 1e-8 else "否"}')
    print(f'  D ⊆ G ?   max||(I−P_G)·v|| = {d_in_g:.3e}   '
          f'{"是" if d_in_g < 1e-8 else "否"}')

    # 主角度：更细的刻画
    C = Vg.T @ Vd
    sv = np.linalg.svd(C, compute_uv=False)
    ang = np.degrees(np.arccos(np.clip(sv, -1, 1)))
    print(f'\n  G 与 D 的主角度（度）：max = {ang.max():.3e}, '
          f'min = {ang.min():.3e}')

    ok = g_in_d < 1e-8 and d_in_g > 1e-8
    print(f'\n结论：{"G ⊆ D 且 D ⊄ G —— " if ok else "!! 未取得预期关系 —— "}', end='')
    if ok:
        print(f'动态数据能确定的信息严格多于静态数据，\n      '
              f'且已足以唯一确定 g(q)。这解释了为什么参数恢复失败（{K-rd}/{K} 个方向\n      '
              f'不可辨识）与闭环重力补偿成功（0.46 mrad）并不矛盾。')
    else:
        print('结论不成立，勿写入论文。')

    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(dict(
            xml=a.xml, n=a.n, seed=a.seed, K=K,
            dim_dynamic=rd, dim_gravity=rg,
            resid_G_in_D=g_in_d, resid_D_in_G=d_in_g,
            principal_angles_deg=ang.tolist(),
            sv_full=sd.tolist(), sv_static=sg.tolist()), indent=2), encoding='utf-8')
        print(f'\n写入 {a.json}')


if __name__ == '__main__':
    main()
