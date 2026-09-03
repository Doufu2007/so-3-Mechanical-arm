"""
SE(3)/SO(3) 李群结构化物理分支 —— 把「SO(3) 李群」从死的输入编码，变成真的动力学结构。

背景与动机
----------
旧实现把 SO(3) 当**输入编码**（把关节角编成各连杆旋转矩阵 R_i(q) ∈ SO(3) 再喂网络），
但严谨复测证明该编码是负结果（旋转矩阵编码 0.9732 < sin/cos 0.9936 < 关节角 0.9936）。
这不是「SO(3) 没用」，而是「SO(3) 用错了地方」——SO(3)/SE(3) 李群真正的价值在**动力学结构**上：

    惯性矩阵的正确定律是  M(q) = Σ_i J_i(q)ᵀ G_i J_i(q)，

其中 J_i(q) ∈ R^{6×n} 是连杆 i 的 SE(3) 空间雅可比（由 product-of-exponentials 前向
运动学给出），G_i ∈ R^{6×6} 是连杆 i 的空间惯量（由质量 m_i、质心 c_i、质心系惯量张量
I_i 三个**学习参数**构造，严格对称正定）。这一结构内在地保证：
  1. M(q) 处处对称正定（SPD）——无需 DeLaN 式的 M·Mᵀ 后处理，SPD 是几何的、非数值的；
  2. 物理量纲正确、可直接用于控制（重力补偿用 ∂V/∂q，V = Σ m_i g·p_com,i）；
  3. 可学习参数只有 10·n_body 个（质量/质心/惯量），参数少、可解释、可辨识性可分析。

本文件实现这一结构，并带**强自检**：把学习参数设成 MuJoCo 真值后，M(q) 必须与
mj_fullM 一致到 ~1e-6，g(q) 必须与 mj_inverse(·,·,0) 一致。自检不通过 = 几何/数学有错，
任何静默出错都会被抓住。用法：

    python paper/se3_physics.py --xml tool/panda.xml          # 只跑自检

约定（Murray–Li–Sastry 空间螺旋轴约定）
--------------------------------------
- 关节 j 的螺旋轴 S_j = (ω_j, v_j) ∈ R⁶ 在**空间系（世界系）q=0 处**定义，v_j = −ω_j × r_j。
- 前向运动学  T_i(q) = exp([S_1]q_1) ⋯ exp([S_anc]q_anc) · M_i，M_i = T_i(0)。
- 空间雅可比 J_i^s(q) 的第 j 列 = Ad_{A_{j-1}(q)} S_j，A_{j-1} = Π_{k<j} exp([S_k]q_k)。
- 空间惯量 G_i^s(q) = Ad_{T_i(q)}^{-T} G_i^{body} Ad_{T_i(q)}^{-1}（Featherstone 约定）。
- M(q) = Σ_i (J_i^s)ᵀ G_i^s J_i^s，V(q) = Σ_i m_i g h_i = −Σ_i m_i (g_vec·p_com,i(q))，g(q) = ∂V/∂q。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ----------------------------------------------------------------------------
# SE(3) 数学工具
# ----------------------------------------------------------------------------

def skew(v):
    """向量 -> 反对称矩阵。v: (..., 3) -> (..., 3, 3)。"""
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    o = torch.zeros_like(x)
    return torch.stack([
        torch.stack([o, -z, y], -1),
        torch.stack([z, o, -x], -1),
        torch.stack([-y, x, o], -1),
    ], -2)


def screw_exp(omega, v, theta):
    """SE(3) 指数映射 exp([S]θ)，S=(ω,v) 单位螺旋轴。返回 (R, p) ∈ (B,3,3),(B,3)。

    约定 ‖ω‖=1（MuJoCo 关节轴是单位向量）。Rodrigues 公式 + 平移部分闭式解。
    """
    B = omega.shape[0]
    wh = skew(omega)               # (B,3,3)
    wh2 = torch.bmm(wh, wh)        # (B,3,3)
    s = torch.sin(theta).view(B, 1, 1)
    c = torch.cos(theta).view(B, 1, 1)
    th = theta.view(B, 1, 1)
    I = torch.eye(3, device=omega.device, dtype=omega.dtype).unsqueeze(0).expand(B, 3, 3)
    R = I + s * wh + (1.0 - c) * wh2
    G = th * I + (1.0 - c) * wh + (th - s) * wh2
    p = torch.bmm(G, v.unsqueeze(-1)).squeeze(-1)
    return R, p


def adjoint_inv(R, p):
    """Ad_{T^{-1}} ∈ R^{6×6}，用于把体框架惯量变换到空间系。T=(R,p)。"""
    B = R.shape[0]
    Rt = R.transpose(-1, -2)
    pt = -torch.bmm(Rt, p.unsqueeze(-1)).squeeze(-1)     # -Rᵀp
    zero = torch.zeros(B, 3, 3, device=R.device, dtype=R.dtype)
    top = torch.cat([Rt, zero], -1)                       # (B,3,6)
    bot = torch.cat([torch.bmm(skew(pt), Rt), Rt], -1)    # (B,3,6)
    return torch.cat([top, bot], -2)                      # (B,6,6)


def spatial_inertia_body(mass, com, I_com):
    """单连杆空间惯量（体框架、关于连杆原点），Featherstone 约定 [ω;v] 排序。

        G = [ I_c + m [c]×[c]×ᵀ ,  m [c]× ;  m [c]×ᵀ ,  m I₃ ]

    mass: 标量；com: (3,) 体框架质心；I_com: (3,3) 质心系惯量。返回 (6,6)。
    """
    cx = skew(com)                                  # (3,3)
    I11 = I_com + mass * (cx @ cx.transpose(-1, -2))
    I12 = mass * cx
    I22 = mass * torch.eye(3, device=com.device, dtype=com.dtype)
    return torch.cat([torch.cat([I11, I12], -1),
                      torch.cat([I12.transpose(-1, -2), I22], -1)], -2)


# ----------------------------------------------------------------------------
# 几何提取（从 MuJoCo 模型一次读出，之后纯 PyTorch 前向）
# ----------------------------------------------------------------------------

def _quat2mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _skew_np(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def _moving_bodies(m):
    """运动连杆（parentid != 0），与 baseline_swevers.py 一致。"""
    return [b for b in range(1, m.nbody) if m.body_parentid[b] != 0]


def _body_anc(m, bid):
    """连杆 bid 的祖先关节数（从世界走到该连杆所经过的铰链关节数，含自身关节）。

    注意：必须**累加 body_jntnum**（每个 body 上的关节数），而不是数经过的 body 个数——
    链条上存在焊死的 body（panda 的 base、hand），数 body 会多算。
    """
    c, b = 0, bid
    while b != 0:
        c += int(m.body_jntnum[b])
        b = m.body_parentid[b]
    return c


def extract_geometry(xml):
    """从 MuJoCo 模型提取 SE(3) 几何 + 真值惯量参数（自检用）。

    返回 dict:
      omega, v      : (n,3) 各关节螺旋轴（空间系 q=0）
      R0, p0        : (n_body,3,3),(n_body,3) 各运动连杆 q=0 位姿 M_i
      body_anc      : (n_body,) 各连杆祖先关节数
      body_ids      : (n_body,) MuJoCo body id
      gravity       : (3,) 重力向量
      mass, com, I  : (n_body,),(n_body,3),(n_body,3,3) 真值惯量（质心系、体框架）
      n_joints      : int
    """
    import mujoco
    m = mujoco.MjModel.from_xml_path(xml)
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)        # qpos <- qpos0
    mujoco.mj_forward(m, d)          # 零位前向，填 xmat/xpos/xanchor/xaxis

    n = m.nv
    if m.nq != n:
        raise ValueError('SE(3) 分支只支持全转动关节（nq==nv）；本模型可能含滑移关节')
    # PoE 以 q=0 为参考位形：M_i=T_i(0) 与螺旋轴都在此处读出，qpos0 必须为零位，
    # 否则 exp([S]q) 应以 (q-qpos0) 为幅角，与训练时喂入的物理关节角不一致。
    if not np.allclose(m.qpos0, 0.0, atol=1e-12):
        raise ValueError('SE(3) 分支要求 qpos0 == 0（模型参考位形非零位）')

    # MuJoCo 版本兼容：HINGE 枚举名在不同版本可能不同
    try:
        HINGE = int(mujoco.mjtJoint.mjJNT_HINGE)
    except AttributeError:
        HINGE = 3  # mjJNT_HINGE 在 MuJoCo 中的数值

    # 关节螺旋轴 S_j = (ω_j, v_j)：q=0 处 ω_j = 世界系关节轴 d.xaxis，
    # v_j = −ω_j × r_j，r_j = 世界系关节锚点 d.xanchor（Murray 空间螺旋轴约定）。
    # 注意 MjData 没有 jacp/jacr 属性（雅可比要 mj_jacBody 现算），xanchor/xaxis 是
    # mj_kinematics 直接填好的、对铰链关节精确的量。
    omega = np.zeros((n, 3)); v = np.zeros((n, 3))
    for j in range(n):
        jid = int(m.dof_jntid[j])            # dof j 所属关节（全铰链时 jid==j）
        if int(m.jnt_type[jid]) != HINGE:
            raise ValueError('SE(3) 分支只支持转动关节')
        w = np.asarray(d.xaxis[jid], dtype=np.float64)
        r = np.asarray(d.xanchor[jid], dtype=np.float64)
        nw = np.linalg.norm(w)
        assert abs(nw - 1.0) < 1e-6, f'关节 {jid} 轴非单位向量: {nw}'
        omega[j] = w / nw
        v[j] = -_skew_np(omega[j]) @ r       # −ω × r

    bodies = _moving_bodies(m)
    n_body = len(bodies)
    R0 = np.zeros((n_body, 3, 3)); p0 = np.zeros((n_body, 3))
    body_anc = np.zeros(n_body, dtype=np.int64)
    mass = np.zeros(n_body); com = np.zeros((n_body, 3)); I = np.zeros((n_body, 3, 3))
    for k, bid in enumerate(bodies):
        R0[k] = np.asarray(d.xmat[bid]).reshape(3, 3)
        p0[k] = np.asarray(d.xpos[bid])
        body_anc[k] = _body_anc(m, bid)
        mass[k] = m.body_mass[bid]
        com[k] = m.body_ipos[bid]
        # body_inertia 形状为 (nbody, 3)：ipos/iquat 框架下的**对角**惯量（非 6 元组）
        Ii = np.diag(np.asarray(m.body_inertia[bid], dtype=np.float64))
        Rq = _quat2mat(m.body_iquat[bid])
        I[k] = Rq @ Ii @ Rq.T      # 惯量框 -> 体框架（panda.xml 为 diaginertia，iquat=单位）

    return dict(omega=omega, v=v, R0=R0, p0=p0, body_anc=body_anc, body_ids=np.asarray(bodies),
                gravity=np.asarray(m.opt.gravity), mass=mass, com=com, I=I, n_joints=n)


# ----------------------------------------------------------------------------
# SE(3) 结构化物理分支
# ----------------------------------------------------------------------------

class SE3PhysicsBranch(nn.Module):
    """τ_rigid = H(q)q̈ + Ḣ(q,q̇)q̇ − ∂L/∂q，其中 H(q)=Σ JᵀGJ 是 SE(3) 结构（非任意神经网络）。

    学习参数（每根连杆 10 个）：质量 m（softplus(log_mass)）、质心 c∈R³、质心系惯量 I=L Lᵀ+εI
    （L 下三角，严格 SPD）。这些是**物理量**，量纲正确、可解释、可辨识性可分析。

    与 PhysicsBranch（DeLaN）保持同一 forward/gravity/construct_H 接口，便于在
    DynamicsModel 里无缝替换。q_mean/q_std 参数仅为接口兼容，SE(3) 结构不使用编码。
    """

    def __init__(self, geometry, epsilon=1e-6):
        super().__init__()
        g = geometry
        self.n = int(g['n_joints'])
        self.n_body = len(g['mass'])
        self.epsilon = epsilon
        self.body_anc = [int(a) for a in g['body_anc']]

        # 固定几何（非学习）：螺旋轴、q=0 位姿、重力
        self.register_buffer('omega', torch.tensor(g['omega'], dtype=torch.float32))
        self.register_buffer('v', torch.tensor(g['v'], dtype=torch.float32))
        self.register_buffer('R0', torch.tensor(g['R0'], dtype=torch.float32))
        self.register_buffer('p0', torch.tensor(g['p0'], dtype=torch.float32))
        # 缓冲区名不能叫 gravity：类里已有 gravity() 方法，register_buffer 会抛
        # KeyError("attribute 'gravity' already exists")。
        self.register_buffer('g_vec', torch.tensor(g['gravity'], dtype=torch.float32))

        # 学习惯量参数：中性初始化（质心在原点、质量≈0.69kg）。
        # L 不能用零初始化：I = L Lᵀ + εI 在 L=0 处对 L 的梯度恒为 0（LLᵀ 的鞍点），
        # 惯量会冻死在 εI 上、从不训练。用对角 0.05·I 给一个非零、量级合理的起点
        # （初值 I ≈ 0.0025·I，与 panda 各连杆真值惯量同量级）。
        self.log_mass = nn.Parameter(torch.zeros(self.n_body))
        self.com = nn.Parameter(torch.zeros(self.n_body, 3))
        self.L = nn.Parameter(0.05 * torch.eye(3).expand(self.n_body, 3, 3).clone())

    # ---- 学习参数 -> 物理量 ----
    def inertial_params(self):
        mass = F.softplus(self.log_mass) + 1e-3
        Ltril = torch.tril(self.L)
        I_com = Ltril @ Ltril.transpose(-1, -2) + self.epsilon * torch.eye(
            3, device=self.L.device, dtype=self.L.dtype)
        return mass, self.com, I_com

    def set_true_params(self, mass, com, I):
        """把学习参数设成 MuJoCo 真值（自检用）。"""
        with torch.no_grad():
            self.log_mass.copy_(torch.log(torch.exp(torch.tensor(mass, dtype=torch.float32) - 1e-3) - 1.0))
            self.com.copy_(torch.tensor(com, dtype=torch.float32))
            L = torch.linalg.cholesky(torch.tensor(I, dtype=torch.float32) - self.epsilon *
                                      torch.eye(3, dtype=torch.float32))
            self.L.copy_(L)

    # ---- 前向运动学 A_k(q) = Π_{l<k} exp([S_l] q_l) ----
    def _fk(self, q_phys):
        B = q_phys.shape[0]
        dev, dt = q_phys.device, q_phys.dtype
        R = torch.eye(3, device=dev, dtype=dt).expand(B, 1, 3, 3)
        p = torch.zeros(B, 1, 3, device=dev, dtype=dt)
        Rs, ps = [R], [p]
        for k in range(self.n):
            w = self.omega[k].expand(B, 3)
            vk = self.v[k].expand(B, 3)
            Rk, pk = screw_exp(w, vk, q_phys[:, k])
            R_prev = Rs[-1].squeeze(1)
            p_prev = ps[-1].squeeze(1)
            R_new = torch.bmm(R_prev, Rk)
            p_new = torch.bmm(R_prev, pk.unsqueeze(-1)).squeeze(-1) + p_prev
            Rs.append(R_new.unsqueeze(1)); ps.append(p_new.unsqueeze(1))
        return torch.cat(Rs, dim=1), torch.cat(ps, dim=1)   # (B,n+1,3,3),(B,n+1,3)

    # ---- 空间雅可比 J_i^s 与空间惯量 G_i^s ----
    def _jacobians(self, q_phys, A_R, A_p):
        B = q_phys.shape[0]
        dev, dt = q_phys.device, q_phys.dtype
        Sprime = torch.zeros(B, self.n, 6, device=dev, dtype=dt)
        for j in range(self.n):
            Rj, pj = A_R[:, j], A_p[:, j]
            wj = self.omega[j].expand(B, 3)
            vj = self.v[j].expand(B, 3)
            om = torch.bmm(Rj, wj.unsqueeze(-1)).squeeze(-1)                  # R ω
            vv = torch.bmm(skew(pj), om.unsqueeze(-1)).squeeze(-1) \
                + torch.bmm(Rj, vj.unsqueeze(-1)).squeeze(-1)                 # [p]×R ω + R v
            Sprime[:, j] = torch.cat([om, vv], -1)
        J = torch.zeros(B, self.n_body, 6, self.n, device=dev, dtype=dt)
        for i in range(self.n_body):
            a = self.body_anc[i]
            if a > 0:
                J[:, i, :, :a] = Sprime[:, :a].transpose(1, 2)               # (B,6,a)
        return J

    # ---- 惯性矩阵 M(q) ----
    def mass_matrix(self, q_phys):
        B = q_phys.shape[0]
        dev, dt = q_phys.device, q_phys.dtype
        A_R, A_p = self._fk(q_phys)
        J = self._jacobians(q_phys, A_R, A_p)
        mass, com, I_com = self.inertial_params()
        M = torch.zeros(B, self.n, self.n, device=dev, dtype=dt)
        for i in range(self.n_body):
            a = self.body_anc[i]
            R = torch.bmm(A_R[:, a], self.R0[i].expand(B, 3, 3))                       # (B,3,3)
            p = torch.bmm(A_R[:, a], self.p0[i].expand(B, 3).unsqueeze(-1)).squeeze(-1) + A_p[:, a]
            Adinv = adjoint_inv(R, p)                                                   # (B,6,6)
            Gb = spatial_inertia_body(mass[i], com[i], I_com[i]).unsqueeze(0).expand(B, 6, 6)
            Gs = torch.bmm(torch.bmm(Adinv.transpose(1, 2), Gb), Adinv)                 # (B,6,6)
            Ji = J[:, i]                                                                # (B,6,n)
            M = M + torch.bmm(torch.bmm(Ji.transpose(1, 2), Gs), Ji)                    # (B,n,n)
        return M

    # ---- 重力势能 V(q; g_dir) ----
    # g_dir 是**基座系里的重力方向**（3 维）。默认用 XML 里的 g_vec（水平基座）。
    # 传入 g_dir 即实现「基座姿态等变」：V 对 g_dir 线性，换安装姿态 = 换 g_dir，零重训。
    def potential(self, q_phys, q_mean=None, q_std=None, g_dir=None):
        B = q_phys.shape[0]
        gv = self.g_vec if g_dir is None else g_dir
        gv = gv.to(device=q_phys.device, dtype=q_phys.dtype)
        if gv.dim() == 1:
            gv = gv.unsqueeze(0).expand(B, 3)
        A_R, A_p = self._fk(q_phys)
        mass, com, _ = self.inertial_params()
        V = torch.zeros(B, device=q_phys.device, dtype=q_phys.dtype)
        for i in range(self.n_body):
            a = self.body_anc[i]
            R = torch.bmm(A_R[:, a], self.R0[i].expand(B, 3, 3))
            p = torch.bmm(A_R[:, a], self.p0[i].expand(B, 3).unsqueeze(-1)).squeeze(-1) + A_p[:, a]
            p_com = torch.bmm(R, com[i].expand(B, 3).unsqueeze(-1)).squeeze(-1) + p       # R c + p
            # 重力势能 U = m g h = −m (g_dir·p_com)；g(q)=∂U/∂q 与 mj_inverse 同号
            V = V - mass[i] * (p_com * gv).sum(-1)
        return V

    # ---- 与 PhysicsBranch 对齐的接口 ----
    def construct_H(self, q_phys, q_mean=None, q_std=None):
        return self.mass_matrix(q_phys)

    def gravity(self, q_phys, q_mean=None, q_std=None, g_dir=None):
        q = q_phys if q_phys.requires_grad else q_phys.detach().requires_grad_(True)
        V = self.potential(q, g_dir=g_dir)
        return torch.autograd.grad(V.sum(), q, create_graph=self.training)[0]

    def forward(self, q_phys, dq_phys, ddq_phys, q_mean=None, q_std=None, want_terms=False, g_dir=None):
        if not q_phys.requires_grad:
            q_phys = q_phys.requires_grad_(True)
        create_graph = self.training

        def h_fn(qq):
            Hq = self.construct_H(qq)
            return Hq, torch.bmm(Hq, dq_phys.unsqueeze(2)).squeeze(2)

        (H, _), (_, Hdot_dq) = torch.func.jvp(h_fn, (q_phys,), (dq_phys,))

        H_ddq = torch.bmm(H, ddq_phys.unsqueeze(2)).squeeze(2)
        kinetic = 0.5 * torch.bmm(torch.bmm(dq_phys.unsqueeze(1), H), dq_phys.unsqueeze(2)).squeeze(-1).squeeze(-1)
        V = self.potential(q_phys, g_dir=g_dir)
        lagrangian = kinetic - V
        dL_dq = torch.autograd.grad(lagrangian.sum(), q_phys, create_graph=create_graph, retain_graph=True)[0]

        tau = H_ddq + Hdot_dq - dL_dq
        comps = {'H': H}
        if want_terms:
            g = torch.autograd.grad(V.sum(), q_phys, create_graph=create_graph, retain_graph=True)[0]
            comps.update({'g': g, 'c': Hdot_dq - dL_dq - g, 'H_ddq': H_ddq, 'V': V})
        return tau, comps


# ----------------------------------------------------------------------------
# 强自检：M(q) vs mj_fullM，g(q) vs mj_inverse(·,·,0)
# ----------------------------------------------------------------------------

def _mj_fullM(m, d, dst):
    """MuJoCo 3.x 版本兼容的 mj_fullM。

    MuJoCo < 3.12：`mj_fullM(m, dst, d.qM)`（d.qM 存在，稀疏关节空间惯量）。
    MuJoCo >= 3.12：`mj_fullM(m, d, dst)`（d.qM 被移除，改为传整个 MjData）。
    用 `hasattr(d, 'qM')` 区分，两端都能跑，上传服务器也不会互相覆盖。
    """
    import mujoco
    if hasattr(d, 'qM'):
        mujoco.mj_fullM(m, dst, d.qM)
    else:
        mujoco.mj_fullM(m, d, dst)


def selfcheck(xml, n_random=64, seed=0, tol=1e-4):
    """把学习参数设为真值，验证 SE(3) 几何/数学正确。返回最大误差 dict。"""
    import mujoco
    m = mujoco.MjModel.from_xml_path(xml)
    d = mujoco.MjData(m)
    g = extract_geometry(xml)

    branch = SE3PhysicsBranch(g)
    branch.set_true_params(g['mass'], g['com'], g['I'])
    branch.eval()

    rng = np.random.RandomState(seed)
    qlo = np.maximum(-1.4, m.jnt_range[:, 0])
    qhi = np.minimum(1.4, m.jnt_range[:, 1])
    err_M, err_g = [], []
    for _ in range(n_random):
        q = rng.uniform(qlo, qhi, m.nv)
        d.qpos[:] = q; d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        M_true = np.zeros((m.nv, m.nv))
        _mj_fullM(m, d, M_true)
        # 重力：q̈=0, q̇=0 的逆动力学 = g(q)。
        # 必须在 mj_forward **之后**清零 qacc——mj_forward 会跑前向动力学写入自由落体
        # 加速度，不清零则 qfrc_inverse = M·q̈+bias ≈ 0，等于拿 0 当真值比。
        d.qacc[:] = 0.0
        mujoco.mj_inverse(m, d)
        g_true = np.asarray(d.qfrc_inverse).copy()

        qt = torch.tensor(q, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            M_pred = branch.mass_matrix(qt)[0].numpy()
        # gravity() 内部要 autograd.grad，不能放在 no_grad 上下文里
        g_pred = branch.gravity(qt)[0].detach().numpy()
        err_M.append(np.abs(M_pred - M_true).max())
        err_g.append(np.abs(g_pred - g_true).max())

    return {'M_max_abs_err': float(np.max(err_M)), 'g_max_abs_err': float(np.max(err_g)),
            'tol': tol, 'n_random': n_random}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--xml', default='tool/panda.xml')
    ap.add_argument('--n-random', type=int, default=64)
    a = ap.parse_args()
    err = selfcheck(a.xml, a.n_random)
    ok = err['M_max_abs_err'] < err['tol'] and err['g_max_abs_err'] < err['tol']
    print(f"[se3 selfcheck] M(q) 最大绝对误差 {err['M_max_abs_err']:.3e} / 阈值 {err['tol']:.0e}")
    print(f"[se3 selfcheck] g(q) 最大绝对误差 {err['g_max_abs_err']:.3e} / 阈值 {err['tol']:.0e}")
    print('[se3 selfcheck] ' + ('PASS ✓ 几何/数学正确' if ok else 'FAIL ✗ 存在错误，勿训练'))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
