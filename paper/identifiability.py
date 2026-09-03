"""惯性参数可辨识性分析 —— 精确回归量、零空间、以及简并随激励的变化。

回答的问题：SE(3) 分支把力矩拟合到机器精度（R²=0.999999），可是学到的 80 个
物理参数偏离真值（惯量相对误差 ≈1.0）。这是训练没调好，还是结构上就不可能？

结论方向：后者。逆动力学对**重心参数**是严格线性的

    tau = Y(q, dq, ddq) · pi,
    pi_i = [m, h = m·c (3), I_o (6 个独立分量)]   —— 每连杆 10 个，关于连杆原点

因为空间惯量在这套参数下是线性的：

    G = [[ I_o , [h]x ], [ [h]x^T , m·I3 ]]

（对照 se3_physics.spatial_inertia_body 的 [[I_c + m[c]x[c]x^T, m[c]x],[.., m I3]]，
两者在 I_o = I_c + m[c]x[c]x^T、h = m·c 下完全一致。）

于是"多组参数给出同一个 tau"就等价于 **Y 有非平凡零空间**，而落在零空间里的
参数方向**永远不可能**从数据里恢复——与优化器、学习率、训练轮数全都无关。

**计算方式（本脚本，与 baseline_swevers 完全不同）**：不碰 SE3PhysicsBranch.forward()
（它内部 torch.func.jvp 与 torch.autograd.functional.jacobian 嵌套会断梯度），而是
直接实现纯函数式刚体动力学

    tau(pi) = M(q,pi)·ddq + C(q,dq,pi)·dq + g(q,pi)

其中 M、g、C 都对 pi 严格线性。利用线性性，回归量 Y 的每一列就是"把 pi 置为对应
基向量 e_k 时的 tau"：

    Y[:, k] = tau(pi=e_k)   （tau(pi=0)=0）

这比中心差分精确（线性函数差分即精确值）、比 autograd 雅可比可靠（不涉及二阶/q 嵌套）。
Coriolis 用 Christoffel 符号的等价形式，只需求 ∂M/∂q 和 ∂g/∂q 两个一阶 q 梯度，
与 selfcheck 里已验证过的 mass_matrix/gravity 同一条通路。

用法：

    python paper/identifiability.py --verify           # 先对 MuJoCo 验算 tau(pi_true)
    python paper/identifiability.py                    # 随机位形，结构性简并
    python paper/identifiability.py --arm franka       # 用真实数据集的位形
    python paper/identifiability.py --n 400 --json out.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from se3_physics import (SE3PhysicsBranch, extract_geometry, skew,  # noqa: E402
                         screw_exp, adjoint_inv)

# I_o 的 6 个独立分量按此顺序展开
SYM_IDX = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]


def sym_from_vec(v6):
    """(B,6) -> (B,3,3) 对称矩阵。"""
    B = v6.shape[0]
    I = v6.new_zeros(B, 3, 3)
    for k, (a, b) in enumerate(SYM_IDX):
        I[:, a, b] = v6[:, k]
        I[:, b, a] = v6[:, k]
    return I


class PiBranch(SE3PhysicsBranch):
    """把 SE3PhysicsBranch 的参数换成重心参数 pi，其余前向逻辑完全复用。

    注意：只复用 mass_matrix / potential / gravity（这三者对 pi 线性），
    **不复用 forward()**（它内部 jvp + autograd.grad(lagrangian,q) 在 pi 进图时断）。
    """

    def set_pi(self, pi):
        self._pi = pi                      # (n_body, 10)

    def pi(self):
        """重心参数 (n_body, 10)。子类（重参数化分支）覆写为 π = B·θ。"""
        return self._pi

    # 不能走 inertial_params() -> (m, c, I_c) 那条路：质心是 c = h/m，而精确回归量是
    # 逐列令 pi = e_k 构造的，大部分列 m = 0，于是 c = h/0 = inf、m·inf = nan，
    # 整个 Y 变成 nan（自检 ||Y·pi−tau||/||tau|| 会报 nan，SVD 随即不收敛）。
    #
    # 而这道中转本来就是多余的：空间惯量对 pi = (m, h, I_o) 本就是**线性**的
    #
    #     G = [[ I_o,     [h]x  ],
    #          [ [h]x^T,  m·I3  ]]
    #
    # 与 spatial_inertia_body 的 (I_c + m[c]x[c]x^T, m[c]x, m·I3) 逐项恒等
    # （I_o 即平行轴定理的结果，m[c]x = [m c]x = [h]x），只是不含除法。
    # 线性性正是 base parameter 分析的前提，绕开中转反而把它显式化了。
    def _spatial_inertia_pi(self):
        pi = self.pi()
        m = pi[:, 0]                                   # (n_body,)
        hx = skew(pi[:, 1:4])                          # (n_body,3,3)
        I_o = sym_from_vec(pi[:, 4:10])                # (n_body,3,3)
        I3 = torch.eye(3, device=pi.device, dtype=pi.dtype)
        top = torch.cat([I_o, hx], -1)                                  # (n_body,3,6)
        bot = torch.cat([hx.transpose(-1, -2), m.view(-1, 1, 1) * I3], -1)
        return torch.cat([top, bot], -2)               # (n_body,6,6)

    def mass_matrix(self, q_phys):
        B = q_phys.shape[0]
        A_R, A_p = self._fk(q_phys)
        J = self._jacobians(q_phys, A_R, A_p)
        G = self._spatial_inertia_pi()
        M = torch.zeros(B, self.n, self.n, device=q_phys.device, dtype=q_phys.dtype)
        for i in range(self.n_body):
            a = self.body_anc[i]
            R = torch.bmm(A_R[:, a], self.R0[i].expand(B, 3, 3))
            p = torch.bmm(A_R[:, a], self.p0[i].expand(B, 3).unsqueeze(-1)).squeeze(-1) + A_p[:, a]
            Adinv = adjoint_inv(R, p)
            Gs = torch.bmm(torch.bmm(Adinv.transpose(1, 2), G[i].expand(B, 6, 6)), Adinv)
            Ji = J[:, i]
            M = M + torch.bmm(torch.bmm(Ji.transpose(1, 2), Gs), Ji)
        return M

    def potential(self, q_phys, q_mean=None, q_std=None, g_dir=None):
        # V = −Σ mᵢ g·(R cᵢ + p) = −Σ g·(R hᵢ + mᵢ p)，同样只用 (m, h)，无需除法
        # g_dir 显式输入即「基座姿态等变」：V 对 g_dir 线性，换安装姿态 = 换 g_dir。
        B = q_phys.shape[0]
        gv = self.g_vec if g_dir is None else g_dir
        gv = gv.to(device=q_phys.device, dtype=q_phys.dtype)
        if gv.dim() == 1:
            gv = gv.unsqueeze(0).expand(B, 3)
        A_R, A_p = self._fk(q_phys)
        pi = self.pi()
        m, h = pi[:, 0], pi[:, 1:4]
        V = torch.zeros(B, device=q_phys.device, dtype=q_phys.dtype)
        for i in range(self.n_body):
            a = self.body_anc[i]
            R = torch.bmm(A_R[:, a], self.R0[i].expand(B, 3, 3))
            p = torch.bmm(A_R[:, a], self.p0[i].expand(B, 3).unsqueeze(-1)).squeeze(-1) + A_p[:, a]
            Rh = torch.bmm(R, h[i].expand(B, 3).unsqueeze(-1)).squeeze(-1)
            V = V - ((Rh + m[i] * p) * gv).sum(-1)
        return V


def true_pi(geom):
    """从 MuJoCo 几何算出真值重心参数。"""
    mass = torch.tensor(geom['mass'], dtype=torch.float64)
    com = torch.tensor(geom['com'], dtype=torch.float64)
    I_c = torch.tensor(geom['I'], dtype=torch.float64)      # (n,3,3) 已是对称
    if I_c.ndim == 2:                       # 保险：若是 (n,3) 对角则升维
        I_c = torch.diag_embed(I_c)
    cx = skew(com)
    I_o = I_c + mass.view(-1, 1, 1) * (cx @ cx.transpose(-1, -2))
    h = mass.unsqueeze(-1) * com
    v6 = torch.stack([I_o[:, a, b] for a, b in SYM_IDX], dim=-1)
    return torch.cat([mass.unsqueeze(-1), h, v6], dim=-1)     # (n_body,10)


def coriolis_dq(branch, q, dq):
    """C(q,dq)·dq，用 Christoffel 符号的等价形式，只需 ∂M/∂q（一阶 q 梯度）。

        (C dq)_i = Σ_{j,k} (∂M_ij/∂q_k) dq_j dq_k − ½ Σ_{a,b} (∂M_ab/∂q_i) dq_a dq_b

    与 SE3PhysicsBranch.forward() 里 Hdot_dq − ∂L/∂q 的科氏部分逐项等价。
    branch._pi 在调用期间当作常量（不要求 pi 进图）。
    """
    n = branch.n
    B = q.shape[0]
    qq = q.detach().requires_grad_(True)
    M = branch.mass_matrix(qq)                       # (B,n,n)
    dM = torch.zeros(B, n, n, n, dtype=M.dtype, device=M.device)
    for i in range(n):
        for j in range(n):
            dM[:, i, j, :] = torch.autograd.grad(
                M[:, i, j].sum(), qq, retain_graph=True)[0]
    C = torch.zeros(B, n, dtype=M.dtype, device=M.device)
    for i in range(n):
        t1 = torch.zeros(B, dtype=M.dtype, device=M.device)
        t2 = torch.zeros(B, dtype=M.dtype, device=M.device)
        for j in range(n):
            for k in range(n):
                t1 = t1 + dM[:, i, j, k] * dq[:, j] * dq[:, k]
        for a in range(n):
            for b in range(n):
                t2 = t2 + dM[:, a, b, i] * dq[:, a] * dq[:, b]
        C[:, i] = t1 - 0.5 * t2
    return C


def rigid_tau(branch, q, dq, ddq):
    """tau(pi) = M·ddq + C·dq + g，对 branch._pi 严格线性。返回 (B,n)。"""
    M = branch.mass_matrix(q)
    H_ddq = torch.bmm(M, ddq.unsqueeze(2)).squeeze(2)
    C = coriolis_dq(branch, q, dq)
    g = branch.gravity(q)
    return H_ddq + C + g


def build_Y_exact(branch, q, dq, ddq):
    """精确回归量 Y: (N, n, K)。利用 tau 对 pi 线性，每列 = tau(e_k)。"""
    n_body = branch.n_body
    n = branch.n
    K = 10 * n_body
    N = q.shape[0]
    Y = torch.zeros(N, n, K, dtype=q.dtype, device=q.device)
    for k in range(K):
        pi = torch.zeros(n_body, 10, dtype=q.dtype, device=q.device)
        pi[k // 10, k % 10] = 1.0
        branch.set_pi(pi)
        Y[:, :, k] = rigid_tau(branch, q, dq, ddq)
    return Y


def verify_against_mujoco(xml, branch, pi_true, n_test=16, seed=1):
    """把 pi 置为真值，验证 rigid_tau 与 mj_inverse 一致（含科氏项）。"""
    import mujoco
    m = mujoco.MjModel.from_xml_path(xml)
    d = mujoco.MjData(m)
    branch.set_pi(pi_true)
    rng = np.random.RandomState(seed)
    errs = []
    for _ in range(n_test):
        q = rng.uniform(-1.2, 1.2, m.nv)
        dq = rng.uniform(-2.0, 2.0, m.nv)
        ddq = rng.uniform(-8.0, 8.0, m.nv)
        d.qpos[:] = q; d.qvel[:] = dq; d.qacc[:] = ddq
        mujoco.mj_inverse(m, d)
        tau_true = np.asarray(d.qfrc_inverse).copy()
        qt = torch.tensor(q, dtype=torch.float64).unsqueeze(0)
        dqt = torch.tensor(dq, dtype=torch.float64).unsqueeze(0)
        ddqt = torch.tensor(ddq, dtype=torch.float64).unsqueeze(0)
        tau_pred = rigid_tau(branch, qt, dqt, ddqt)[0].detach().numpy()
        errs.append(float(np.abs(tau_pred - tau_true).max()))
    return float(np.max(errs))


def random_states(n, n_joints, rng, q_lim=1.4, dq_lim=3.0, ddq_lim=10.0):
    u = lambda lim, s: torch.tensor(rng.uniform(-lim, lim, s), dtype=torch.float64)
    return (u(q_lim, (n, n_joints)), u(dq_lim, (n, n_joints)), u(ddq_lim, (n, n_joints)))


def analyse(Y, K, tol_rel=1e-9):
    """SVD -> 数值秩、零空间、奇异值谱。"""
    A = Y.reshape(-1, K).numpy()
    scale = max(np.abs(A).max(), 1e-30)
    A = A / scale                            # 只做整体缩放，不改变零空间
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    cutoff = s[0] * tol_rel
    rank = int((s > cutoff).sum())
    null = Vt[rank:]                            # (K-rank, K) 零空间的正交基
    return s, rank, null


def per_body_report(null, n_body):
    """每个连杆有多少参数方向落在零空间里（按能量占比）。"""
    if null.shape[0] == 0:
        return [0.0] * n_body
    w = (null ** 2).sum(axis=0)                 # (K,) 每个参数在零空间里的能量
    return [float(w[10 * i:10 * (i + 1)].sum()) for i in range(n_body)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--xml', default=None, help='默认 tool/panda.xml')
    ap.add_argument('--arm', default=None,
                    help='用某个数据集的真实位形（franka / franka_ex）；默认随机采样')
    ap.add_argument('--n', type=int, default=300, help='采样点数')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tol', type=float, default=1e-9, help='数值秩的相对阈值')
    ap.add_argument('--static', action='store_true',
                    help='dq=ddq=0，只分析重力项 g(q) 的可辨识性')
    ap.add_argument('--json', default=None)
    ap.add_argument('--verify', action='store_true',
                    help='仅对 MuJoCo 验算 rigid_tau(pi_true)，然后退出')
    a = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    # MuJoCo 的 from_xml_path 用自有 VFS 解析 POSIX 路径，绝对 Windows 路径（尤其
    # 含中文）会失败。默认用相对路径，脚本须在项目根目录下运行。
    xml = a.xml or 'tool/panda.xml'
    geom = extract_geometry(xml)
    n_body, n_joints = len(geom['mass']), int(geom['n_joints'])
    K = 10 * n_body

    branch = PiBranch(geom).double()
    branch.eval()                        # 只用一阶 q 梯度，不需要 create_graph
    pi0 = true_pi(geom)

    if a.verify:
        err = verify_against_mujoco(xml, branch, pi0)
        print(f'[verify] rigid_tau(pi_true) vs mj_inverse 最大绝对误差 = {err:.3e}')
        print('[verify] ' + ('PASS  科氏/重力/惯性项一致' if err < 1e-5
                              else 'FAIL  公式有误，勿信后面的零空间结论'))
        sys.exit(0 if err < 1e-5 else 1)

    rng = np.random.RandomState(a.seed)
    if a.arm:
        from paper.core import load_arm
        d = load_arm(a.arm)['train']
        idx = np.sort(rng.choice(len(d[0]), min(a.n, len(d[0])), replace=False))
        q, dq, ddq = (torch.tensor(x[idx], dtype=torch.float64) for x in d[:3])
        src = f'{a.arm} 训练集 {len(idx)} 点'
    elif a.static:
        # 重力专项：dq=ddq=0，回归量只剩 g(q)。用来回答「闭环重力补偿为什么能 work，
        # 而参数恢复却失败」—— 若重力可辨识子空间显著大于完整 tau 的，说明控制用到的
        # 那部分参数组合恰好是可辨识的，两个结论并不矛盾。
        q, _, _ = random_states(a.n, n_joints, rng)
        dq = torch.zeros_like(q)
        ddq = torch.zeros_like(q)
        src = f'静态位形 {a.n} 点（|q|<=1.4, dq=ddq=0，仅重力项）'
    else:
        q, dq, ddq = random_states(a.n, n_joints, rng)
        src = f'随机位形 {a.n} 点（|q|<=1.4, |dq|<=3, |ddq|<=10）'

    print(f'模型   {xml}')
    print(f'连杆   {n_body} 个 · 自由度 {n_joints} · 参数 K = 10x{n_body} = {K}')
    print(f'采样   {src}')

    Y = build_Y_exact(branch, q, dq, ddq)

    # 线性性自检：Y·pi 必须精确等于 tau，否则参数化写错了
    branch.set_pi(pi0)
    tau_ref = rigid_tau(branch, q, dq, ddq)
    tau_lin = (Y.reshape(q.shape[0], -1, K) @ pi0.reshape(-1)).reshape_as(tau_ref)
    rel = (tau_lin - tau_ref).norm() / tau_ref.norm()
    print(f'\n线性性自检  ||Y·pi - tau|| / ||tau|| = {rel:.3e}', end='  ')
    print('OK' if rel < 1e-10 else '!! 参数化有误，后面的结论不可信')

    s, rank, null = analyse(Y, K, a.tol)
    print(f'\n{"="*58}\n可辨识性\n{"="*58}')
    print(f'  参数总数 K            {K}')
    print(f'  数值秩（可辨识）      {rank}')
    print(f'  零空间维数（不可辨识）{K - rank}   <-- 永远学不到的方向数')
    print(f'  不可辨识占比          {(K-rank)/K:.1%}')

    print(f'\n奇异值谱（相对 s[0]，看断崖在哪）：')
    for i in (list(range(min(3, len(s)))) +
              list(range(max(0, rank - 2), min(len(s), rank + 3)))):
        mark = '  <- 秩截断' if i == rank else ''
        print(f'    s[{i:3d}] = {s[i]/s[0]:.3e}{mark}')

    ebody = per_body_report(null, n_body)
    names = [f'body{int(b)}' for b in geom.get('body_ids', range(n_body))]
    print(f'\n各连杆落在零空间里的能量占比（越高=该连杆参数越不可辨识）：')
    for i in range(n_body):
        bar = '#' * int(round(ebody[i] / max(max(ebody), 1e-9) * 30))
        print(f'    {str(names[i]):8s} {ebody[i]:6.2f}  {bar}')

    if a.json:
        import json
        Path(a.json).write_text(json.dumps({
            'xml': xml, 'source': src, 'n_samples': int(q.shape[0]),
            'K': K, 'rank': int(rank), 'null_dim': int(K - rank),
            'tol_rel': a.tol,
            'singular_values_rel': (s / s[0]).tolist(),
            'null_energy_per_body': ebody,
            'body_names': [str(x) for x in names],
            'linearity_check': float(rel),
        }, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f'\n写入 {a.json}')

    print(f'\n解读：零空间维数 {K-rank} 就是"无论怎么训练都恢复不了"的参数方向数。')
    print(f'      这解释了 se3_param_check 里惯量相对误差 ~1.0 —— 不是没学好，是学不到。')


if __name__ == '__main__':
    main()
