"""
模型定义 — 单一真相来源，所有消融通过开关控制。

相对旧实现（pinn_delan_master.py / pinn_delan_master_franka.py）修正的三处实质问题：

  [BUG-1] 拉格朗日梯度用了 .mean() 而非 .sum()
      旧代码: dL_dq = autograd.grad(lagrangian.mean(), q)  ->  实际得到 (1/B)·∂L/∂q
      逐样本标量对自身输入求导时，正确写法是 .sum()。用 .mean() 会把重力项和
      科里奥利的第二项整体缩小 batch_size 倍（B=1024 时缩小 1000 倍），
      等于把 g(q) 从模型里抹掉了 —— 这正是「物理分支单独 R² 为负」的根因。

  [BUG-2] SO(3)/sin-cos 编码作用在**归一化后**的关节角上
      旧代码把 (q-μ)/σ 当作角度送进 sin/cos 和旋转矩阵，sin((q-μ)/σ) 与 sin(q)
      无关，所谓「真实连杆旋转矩阵」并不成立。此处一律先反归一化回物理弧度再编码。

  [BUG-3] 归一化破坏拉格朗日结构 / 正定性语义
      旧代码 dq、ddq 各自减均值除自身 std，导致网络看到的「速度」不是「位置」的
      时间导数；且逐关节缩放下网络内部的 H̃ 对称正定并不意味着物理 H 对称正定。
      此处物理分支**全程在物理单位下计算**（H、c、g 都有量纲意义，可直接用于控制），
      只在输出端做 (τ-μ_τ)/σ_τ 变换。
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------

@dataclass
class ModelCfg:
    kind: str = 'delan_ffnn'      # delan_ffnn | mlp | hnn | lnn | symoden
    gravity_input: bool = False   # mlp 基线是否把单位重力方向 ĝ 作为显式输入（生死门第三组对照）
    encoding: str = 'sincos'      # none | sincos | rotmat
    physics: bool = True          # 物理先验分支（DeLaN）
    physics_kind: str = 'delan'   # delan | se3（SE(3)/SO(3) 结构化惯量 M=ΣJᵀGJ）
    se3_xml: str = ''             # se3 分支的 MuJoCo 模型路径（空 = tool/panda.xml）
    se3_basis: str = ''           # se3_reparam 分支的可辨识子空间基 npz（空 = base_basis.npz）
    uncertainty: bool = True      # 不确定性补偿分支（FFNN）
    adaptive_lambda: bool = True  # 自适应 λ（只影响损失；physics+uncertainty 都在时才有意义）
    multiscale_loss: bool = True  # 多尺度关节加权损失
    phys_loss: bool = False       # 黑箱基线用物理单位损失+物理R²选模（归一化空间放大 J1/J7 退化关节噪声）
    enc_bottleneck: bool = True   # 编码特征是否线性压回 n_joints 维（复现旧实现的做法）
    lambda_reg: float = 0.05      # adaptive_lambda=False 时的固定 λ
    # ---- 对偶上升约束式 λ（把「自适应 λ 塌缩」的负结果升级成方法） ----
    # dual_ascent=True 时，λ 不再是可学习网络输出（那会塌缩到 0），也不是人工固定的
    # 超参，而是对偶变量：min_θ data_loss  s.t.  ‖ε‖² ≤ δ，λ 沿约束违反方向做投影梯度上升。
    #   残差预算 δ = dual_delta_frac² · mean(‖τ‖²)，即「残差分支 RMS 允许占扭矩 RMS 的比例」。
    dual_ascent: bool = False
    dual_delta_frac: float = 0.3   # 残差预算：残差 RMS 相对 τ RMS 的上限比例
    dual_lambda_init: float = 1.0
    dual_lr: float = 1e-3
    delan_hidden: tuple = (1024, 1024, 512, 256)
    unc_hidden: tuple = (2048, 1024, 512, 256)
    dropout: float = 0.1
    epsilon: float = 1e-6

    def to_dict(self):
        d = asdict(self)
        d['delan_hidden'] = list(self.delan_hidden)
        d['unc_hidden'] = list(self.unc_hidden)
        return d


class EnhancedMLP(nn.Module):
    """Linear -> LayerNorm -> SiLU (-> Dropout) 堆叠，Xavier 初始化。与旧实现一致。"""

    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.1):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.SiLU()]
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.network = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.network(x)


# ----------------------------------------------------------------------------
# 关节角编码
# ----------------------------------------------------------------------------

def _rot(axis, a):
    c, s = torch.cos(a), torch.sin(a)
    z, o = torch.zeros_like(a), torch.ones_like(a)
    if axis == 'z':
        rows = [[c, -s, z], [s, c, z], [z, z, o]]
    elif axis == 'y':
        rows = [[c, z, s], [z, o, z], [-s, z, c]]
    elif axis == 'x':
        rows = [[o, z, z], [z, c, -s], [z, s, c]]
    else:
        raise ValueError(axis)
    return torch.stack([torch.stack(r, dim=-1) for r in rows], dim=-2)


class JointEncoder(nn.Module):
    """把**物理**关节角 q (rad) 编码成 n_joints 维特征。

    none   : 直接用标准化后的 q（等价于欧拉角/关节角直接输入）
    sincos : [sin q, cos q] -> Linear(2n, n)，周期性编码，初始化为近似恒等
    rotmat : 名义关节轴的指数积得到各连杆姿态 R_i(q) ∈ SO(3)，
             展平 (n*9) -> Linear(n*9, n)。不需要连杆长度，只需要标称轴方向。
    """

    def __init__(self, n_joints, kind='sincos', axes='zyzyzyz', bottleneck=True):
        """bottleneck=True 复现旧实现：把编码特征线性压回 n_joints 维再进网络。
        bottleneck=False 则把编码特征原样送进后续网络。

        这个开关是必须的：sin/cos 是 2n 维、旋转矩阵是 9n 维，若一律压回 n 维，
        「编码方式」的比较就和「被压掉多少信息」混在一起 —— 压缩本身是有损的
        （q ↦ R 单射，但 63→7 的线性投影不可逆），测出来的差异未必来自编码方式。
        """
        super().__init__()
        self.kind = kind
        self.n = n_joints
        self.axes = axes[:n_joints]
        self.bottleneck = bottleneck
        raw_dim = {'none': n_joints, 'sincos': n_joints * 2, 'rotmat': n_joints * 9}
        if kind not in raw_dim:
            raise ValueError(kind)
        self.out_dim = n_joints if (bottleneck or kind == 'none') else raw_dim[kind]

        if kind == 'none' or not bottleneck:
            self.proj = None
        elif kind == 'sincos':
            self.proj = nn.Linear(n_joints * 2, n_joints)
            with torch.no_grad():
                self.proj.weight.zero_()
                self.proj.weight[:, :n_joints].copy_(torch.eye(n_joints))
                self.proj.bias.zero_()
        else:
            self.proj = nn.Linear(n_joints * 9, n_joints)
            with torch.no_grad():
                nn.init.xavier_uniform_(self.proj.weight)
                self.proj.bias.zero_()

    def link_rotations(self, q_phys):
        """R_i = Πⱼ≤i Rot(axis_j, q_j)，返回 (B, n, 3, 3)。"""
        Rs, R = [], None
        for i, ax in enumerate(self.axes):
            Ri = _rot(ax, q_phys[:, i])
            R = Ri if R is None else torch.bmm(R, Ri)
            Rs.append(R)
        return torch.stack(Rs, dim=1)

    def forward(self, q_phys, q_norm):
        if self.kind == 'none':
            return q_norm
        if self.kind == 'sincos':
            feat = torch.cat([torch.sin(q_phys), torch.cos(q_phys)], dim=-1)
        else:
            feat = self.link_rotations(q_phys).reshape(q_phys.shape[0], -1)
        return self.proj(feat) if self.proj is not None else feat


# ----------------------------------------------------------------------------
# 物理先验分支（DeLaN），全程物理单位
# ----------------------------------------------------------------------------

class PhysicsBranch(nn.Module):
    """τ_rigid = H(q)·q̈ + Ḣ(q,q̇)·q̇ − ∂L/∂q,  L = ½ q̇ᵀH(q)q̇ − V(q)

    H = M(q)·M(q)ᵀ + εI 严格对称正定（物理单位）。
    Ḣ·q̇ 用前向模式自动微分（JVP）一次算出，避免显式 Christoffel 符号的 O(n³) 存储。
    物理分支不使用 dropout：否则 construct_H 与 JVP 内重算的 H 会因随机 mask 不一致。
    """

    def __init__(self, n_joints, encoder: JointEncoder, hidden=(1024, 1024, 512, 256), epsilon=1e-6):
        super().__init__()
        self.n = n_joints
        self.epsilon = epsilon
        self.encoder = encoder
        d = encoder.out_dim
        self.H_net = EnhancedMLP(d, list(hidden), n_joints * n_joints, dropout_rate=0.0)
        self.V_net = EnhancedMLP(d, list(hidden), 1, dropout_rate=0.0)
        # 输出层缩小初始化：让 H≈εI、V≈0，初始 τ_rigid≈0。
        # 否则 Xavier 初始化下 H~O(1)，配上 Baxter 腕部 |q̈|~40 rad/s² 会让初始
        # 预测比目标大两个数量级，训练前期全在把 H 压回去。
        with torch.no_grad():
            for net in (self.H_net, self.V_net):
                net.network[-1].weight.mul_(0.1)
                net.network[-1].bias.zero_()

    def _enc(self, q_phys, q_mean, q_std):
        return self.encoder(q_phys, (q_phys - q_mean) / q_std)

    def construct_H(self, q_phys, q_mean, q_std):
        B, n = q_phys.shape[0], self.n
        M = self.H_net(self._enc(q_phys, q_mean, q_std)).reshape(B, n, n)
        H = torch.bmm(M, M.transpose(1, 2))
        I = torch.eye(n, device=q_phys.device, dtype=q_phys.dtype).expand(B, n, n)
        return H + self.epsilon * I

    def potential(self, q_phys, q_mean, q_std):
        return self.V_net(self._enc(q_phys, q_mean, q_std)).squeeze(-1)

    def gravity(self, q_phys, q_mean, q_std):
        """g(q) = ∂V/∂q，物理单位 (N·m)。控制器做重力补偿直接用这个。"""
        q = q_phys if q_phys.requires_grad else q_phys.detach().requires_grad_(True)
        V = self.potential(q, q_mean, q_std)
        return torch.autograd.grad(V.sum(), q, create_graph=self.training)[0]

    def forward(self, q_phys, dq_phys, ddq_phys, q_mean, q_std, want_terms=False):
        if not q_phys.requires_grad:
            q_phys = q_phys.requires_grad_(True)
        # 只有训练时才需要保留二阶图；评估时 create_graph=False 大幅省显存
        create_graph = self.training

        # H(q) 与 Ḣ(q,q̇)·q̇ 在同一次 JVP 里算出，保证两者用的是同一个 H
        def h_fn(qq):
            Hq = self.construct_H(qq, q_mean, q_std)
            return Hq, torch.bmm(Hq, dq_phys.unsqueeze(2)).squeeze(2)

        (H, _), (_, Hdot_dq) = torch.func.jvp(h_fn, (q_phys,), (dq_phys,))

        H_ddq = torch.bmm(H, ddq_phys.unsqueeze(2)).squeeze(2)
        kinetic = 0.5 * torch.bmm(torch.bmm(dq_phys.unsqueeze(1), H), dq_phys.unsqueeze(2)).squeeze(-1).squeeze(-1)
        V = self.potential(q_phys, q_mean, q_std)
        lagrangian = kinetic - V

        # [BUG-1 修正] 逐样本标量对自身输入求导 -> 必须用 sum()，用 mean() 会缩小 B 倍
        dL_dq = torch.autograd.grad(lagrangian.sum(), q_phys,
                                    create_graph=create_graph, retain_graph=True)[0]

        tau = H_ddq + Hdot_dq - dL_dq
        comps = {'H': H}
        if want_terms:
            g = torch.autograd.grad(V.sum(), q_phys, create_graph=create_graph, retain_graph=True)[0]
            comps.update({'g': g, 'c': Hdot_dq - dL_dq - g, 'H_ddq': H_ddq, 'V': V})
        return tau, comps


class AdaptiveLambda(nn.Module):
    """逐样本的物理约束权重 λ(q,q̇,q̈) ∈ (0, 0.1)。"""

    def __init__(self, n_joints, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_joints * 3, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1), nn.Sigmoid())
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, q, dq, ddq):
        return (self.net(torch.cat([q, dq, ddq], dim=-1)) * 0.1).squeeze(-1)


# ----------------------------------------------------------------------------
# 顶层模型
# ----------------------------------------------------------------------------

class NormalizedModel(nn.Module):
    """所有模型的公共外壳：**输入是物理单位** (rad, rad/s, rad/s²)，
    **输出是归一化扭矩** (τ−μ_τ)/s_τ。归一化常数作为 buffer 随 checkpoint 保存。
    """

    def __init__(self, norm):
        super().__init__()
        for k in ['q_mean', 'q_std', 'dq_mean', 'dq_std', 'ddq_mean', 'ddq_std',
                  'tau_mean', 'tau_scale']:
            self.register_buffer(k, torch.tensor(getattr(norm, k)))

    def zscore(self, q, dq, ddq):
        return torch.cat([(q - self.q_mean) / self.q_std,
                          (dq - self.dq_mean) / self.dq_std,
                          (ddq - self.ddq_mean) / self.ddq_std], dim=-1)

    def tau_to_norm(self, tau_phys):
        return (tau_phys - self.tau_mean) / self.tau_scale

    def tau_to_phys(self, tau_n):
        return tau_n * self.tau_scale + self.tau_mean

    def predict_phys(self, q, dq, ddq):
        """物理单位的 τ 预测，控制器直接用。"""
        return self.tau_to_phys(self(q, dq, ddq)[0])


class DynamicsModel(NormalizedModel):
    """双分支模型：物理先验分支（物理单位）+ 不确定性分支（z-score 输入）。"""

    def __init__(self, cfg: ModelCfg, n_joints, norm, axes='zyzyzyz'):
        super().__init__(norm)
        self.cfg = cfg
        self.n = n_joints
        if cfg.kind != 'delan_ffnn':
            raise ValueError('DynamicsModel 只处理 delan_ffnn；基线见 build_model')
        if not (cfg.physics or cfg.uncertainty):
            raise ValueError('physics 和 uncertainty 不能同时关闭')

        if cfg.physics:
            if cfg.physics_kind == 'se3':
                from se3_physics import extract_geometry, SE3PhysicsBranch
                from core import ROOT
                # MuJoCo 的 from_xml_path 走自有 VFS，绝对 Windows 路径（含中文）会
                # ValueError。用相对路径，脚本须在项目根目录下运行（与 identifiability.py 一致）。
                xml = cfg.se3_xml or 'tool/panda.xml'
                geom = extract_geometry(xml)
                if geom['n_joints'] != n_joints:
                    raise ValueError(f'SE(3) 几何关节数 {geom["n_joints"]} ≠ 模型关节数 {n_joints}')
                self.phys = SE3PhysicsBranch(geom, cfg.epsilon)
            elif cfg.physics_kind == 'se3_reparam':
                # 可辨识子空间重参数化：学 θ∈R^rank，π = B·θ（无零空间，学回真 base 参数）
                import numpy as np
                from se3_physics import extract_geometry
                from base_reparam_branch import ReparamBranch
                from core import ROOT
                # MuJoCo 的 from_xml_path 走自有 VFS，绝对 Windows 路径（含中文）会
                # ValueError。用相对路径，脚本须在项目根目录下运行（与 identifiability.py 一致）。
                xml = cfg.se3_xml or 'tool/panda.xml'
                geom = extract_geometry(xml)
                if geom['n_joints'] != n_joints:
                    raise ValueError(f'SE(3) 几何关节数 {geom["n_joints"]} ≠ 模型关节数 {n_joints}')
                basis_path = ROOT / (cfg.se3_basis or 'base_basis.npz')
                if not basis_path.exists():
                    raise FileNotFoundError(
                        f'缺少可辨识子空间基 {basis_path}。先跑 '
                        f'`python paper/base_reparam.py --xml {xml} --out {basis_path.name}`')
                self.phys = ReparamBranch(geom, np.load(basis_path)['B'], cfg.epsilon)
            else:
                enc = JointEncoder(n_joints, cfg.encoding, axes, cfg.enc_bottleneck)
                self.phys = PhysicsBranch(n_joints, enc, cfg.delan_hidden, cfg.epsilon)
        if cfg.uncertainty:
            self.unc = EnhancedMLP(3 * n_joints, list(cfg.unc_hidden), n_joints, cfg.dropout)
        if cfg.adaptive_lambda:
            self.lam_net = AdaptiveLambda(n_joints)
        # 只有物理分支时需要一个常数偏置来承担 μ_τ（重力网络也能学，但给它更好的起点）
        self.tau_bias = nn.Parameter(torch.zeros(n_joints))

    def forward(self, q, dq, ddq, want_terms=False, g_dir=None):
        comps = {}
        tau_n = self.tau_bias.expand(q.shape[0], -1)

        if self.cfg.physics:
            if self.cfg.physics_kind in ('se3', 'se3_reparam'):
                tau_phys, pc = self.phys(q, dq, ddq, self.q_mean, self.q_std, want_terms, g_dir=g_dir)
            else:
                tau_phys, pc = self.phys(q, dq, ddq, self.q_mean, self.q_std, want_terms)
            comps.update(pc)
            comps['tau_rigid_phys'] = tau_phys
            comps['tau_rigid_n'] = self.tau_to_norm(tau_phys)
            tau_n = tau_n + comps['tau_rigid_n']

        if self.cfg.uncertainty:
            eps = self.unc(self.zscore(q, dq, ddq))
            comps['epsilon'] = eps
            tau_n = tau_n + eps

        if self.cfg.adaptive_lambda:
            zq, zdq, zddq = self.zscore(q, dq, ddq).chunk(3, dim=-1)
            comps['lambda'] = self.lam_net(zq, zdq, zddq)

        return tau_n, comps

    def gravity_phys(self, q, g_dir=None):
        """g(q) = ∂V/∂q，物理单位 N·m（重力补偿控制用）。

        g_dir 仅在 SE(3) 分支有意义（基座姿态等变）：换安装姿态 = 换 g_dir。
        """
        if not self.cfg.physics:
            raise RuntimeError('该变体没有物理分支，无法输出 g(q)')
        if self.cfg.physics_kind in ('se3', 'se3_reparam'):
            return self.phys.gravity(q, self.q_mean, self.q_std, g_dir=g_dir)
        return self.phys.gravity(q, self.q_mean, self.q_std)

    def inertia_phys(self, q):
        if not self.cfg.physics:
            raise RuntimeError('该变体没有物理分支，无法输出 H(q)')
        return self.phys.construct_H(q, self.q_mean, self.q_std)


class PlainMLP(NormalizedModel):
    """纯数据驱动基线：ReLU MLP，输入 z-score 后的 [q, dq, ddq]（可选 ⊕ 单位重力方向 ĝ）。

    gravity_input=True（生死门第三组对照）：把单位重力方向 ĝ 作为第 22~24 维显式输入，
    和结构模型拿到的是**同样的信息**（q,q̇,q̈ + 重力方向），差别只在结构——
    结构模型把 g 约束成 g_dir 的线性函数，这里没有。架构/隐层/超参与裸黑箱逐位相同。
    """

    def __init__(self, norm, n_joints=7, hidden=(1024, 1024, 512, 256), gravity_input=False):
        super().__init__(norm)
        self.gravity_input = gravity_input
        in_dim = 3 * n_joints + (3 if gravity_input else 0)
        layers, prev = [], in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, n_joints))
        self.net = nn.Sequential(*layers)

    def forward(self, q, dq, ddq, want_terms=False, g_dir=None):
        z = self.zscore(q, dq, ddq)
        if self.gravity_input:
            if g_dir is None:
                raise ValueError('gravity_input=True 的 MLP 必须显式喂 g_dir（重力方向）')
            g = g_dir.to(device=z.device, dtype=z.dtype)
            if g.dim() == 1:
                g = g.unsqueeze(0).expand(z.shape[0], 3)
            # 单位化：所有安装姿态的重力模长都等于 9.81，模长不携带"换姿态"信息，
            # 只有方向携带；喂单位向量既信息无损，又避开 9.81 与 z-score 的尺度失配。
            g = g / (g.norm(dim=-1, keepdim=True) + 1e-8)
            z = torch.cat([z, g], dim=-1)
        return self.net(z), {}


class _WrapBaseline(NormalizedModel):
    """HNN / LNN / SymODEN：沿用原实现，喂 z-score 输入。"""

    def __init__(self, m, norm):
        super().__init__(norm)
        self.m = m

    def forward(self, q, dq, ddq, want_terms=False):
        z = self.zscore(q, dq, ddq)
        zq, zdq, zddq = z.chunk(3, dim=-1)
        out = self.m(zq.clone().requires_grad_(True), zdq, zddq)
        pred = out[0] if isinstance(out, (tuple, list)) else out
        return pred, {}


def build_model(cfg: ModelCfg, n_joints, norm, axes='zyzyzyz'):
    if cfg.kind == 'mlp':
        return PlainMLP(norm, n_joints, gravity_input=cfg.gravity_input)
    if cfg.kind in ('hnn', 'lnn', 'symoden'):
        from baseline_hnn_lnn import HNNWrapper, LNNWrapper, SymODENWrapper
        cls = {'hnn': HNNWrapper, 'lnn': LNNWrapper, 'symoden': SymODENWrapper}[cfg.kind]
        return _WrapBaseline(cls(n_joints=n_joints), norm)
    return DynamicsModel(cfg, n_joints, norm, axes)


# ----------------------------------------------------------------------------
# 损失
# ----------------------------------------------------------------------------

def joint_weights_multiscale(tau_true):
    """小扭矩关节（腕部）加权，避免均方误差被大扭矩关节主导。"""
    n = tau_true.shape[1]
    std = tau_true.std(0)
    std_n = std / (std.mean() + 1e-8)
    w = torch.ones(n, device=tau_true.device, dtype=tau_true.dtype)
    n_low = max(1, n * 3 // 7)
    order = torch.argsort(std)
    low, high = order[:n_low], order[n_low:]
    w[high] = 1.2 * (1.0 + 0.5 * (1.0 - std_n[high]))
    w[low] = 20.0 * (1.0 + 0.5 * (1.0 - std_n[low]))
    return w


def compute_loss(tau_pred, tau_true, comps, cfg: ModelCfg, lam_dual=None):
    if cfg.multiscale_loss:
        w = joint_weights_multiscale(tau_true)
        data_loss = (w.unsqueeze(0) * F.huber_loss(tau_pred, tau_true, delta=1.0, reduction='none')).mean()
    else:
        data_loss = F.huber_loss(tau_pred, tau_true, delta=1.0)

    reg = tau_pred.new_zeros(())
    lam_mean = 0.0
    residual_sq_mean = 0.0     # 对偶上升用：本 batch 的 ‖ε‖²（归一化扭矩空间）
    dual_delta = 0.0           # 对偶上升用：残差预算 δ
    if cfg.uncertainty and 'epsilon' in comps:
        eps = comps['epsilon']
        if cfg.dual_ascent:
            # 约束式：min_θ data_loss  s.t. ‖ε‖² ≤ δ。θ 只对 λ·‖ε‖² 求导（−λδ 是常数，
            # 不进梯度）；λ 作为对偶变量在 train.py 里做投影梯度上升，故这里 lam_dual 是常数。
            eps_sq = eps.pow(2).mean()
            reg = lam_dual * eps_sq
            lam_mean = float(lam_dual)
            residual_sq_mean = float(eps_sq.detach())
            # 残差预算：允许残差分支 RMS 占扭矩 RMS 的比例（两者同处归一化扭矩空间）
            dual_delta = float(cfg.dual_delta_frac ** 2 * tau_true.pow(2).mean().detach())
        elif cfg.adaptive_lambda and 'lambda' in comps:
            lam = comps['lambda']
            reg = (lam * eps.pow(2).mean(dim=1)).mean()
            lam_mean = float(lam.mean().detach())
        else:
            reg = cfg.lambda_reg * eps.pow(2).mean()
            lam_mean = cfg.lambda_reg
    total = data_loss + reg
    return total, {'data_loss': float(data_loss), 'reg_loss': float(reg),
                   'lambda_mean': lam_mean, 'residual_sq_mean': residual_sq_mean,
                   'dual_delta': dual_delta}
