"""可辨识子空间重参数化分支 —— 学 θ∈R^{rank}，π = B·θ。

把 SE(3) 分支的参数从 K 维重心参数 pi 换成 rank 维可辨识参数 θ
（Gautier/Khalil base parameter 思想）。π = B·θ 严格限制在可辨识子空间内，
模型**没有零空间**，学到的 θ 就是真 base 参数 θ_true = Bᵀ·π_true。

这是把 audit 的负结果（K 参数里 K-rank 个方向永远学不到）翻成工程正结果的
核心改动：与其让模型在零空间里瞎追，不如直接把参数空间压到可辨识子空间，
于是「学回真参数」从不可能变成可能。

与 PiBranch 共享 mass_matrix / potential（对 pi 线性），只把 pi 的来源从
`set_pi` 外部注入换成 θ 经线性层 `π = B·θ` 算出。forward() 沿用 SE3PhysicsBranch
的 jvp 实现（θ 是 nn.Parameter，梯度经 π=B·θ 正常回传，与 DeLaN 捕获网络权重的
方式一致）。

用法（先生成 B，再训练）：
    python paper/base_reparam.py --xml tool/panda.xml --out base_basis.npz
    python paper/train.py --arm franka_ex --variant se3_reparam_no_uncertainty --seed 0
    python paper/reparam_param_check.py --ckpt results_v2/se3_reparam_no_uncertainty_franka_ex_seed0.pth
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from se3_physics import skew
from identifiability import PiBranch, sym_from_vec


class ReparamBranch(PiBranch):
    """SE(3) 分支的可辨识子空间重参数化：唯一学习参数 θ∈R^{rank}，π = B·θ。"""

    def __init__(self, geometry, B, epsilon=1e-6):
        super().__init__(geometry, epsilon)
        # 移除父类创建的重心参数 log_mass/com/L——重参数化分支只用 θ。
        # 不注册即不进 state_dict / 不进 optimizer，避免死参数虚报 n_params。
        for name in ('log_mass', 'com', 'L'):
            self._parameters.pop(name, None)

        B = torch.as_tensor(np.asarray(B), dtype=torch.float32)
        if B.ndim != 2:
            raise ValueError(f'B 必须是 (K, rank) 矩阵，得到 {B.shape}')
        self.rank = int(B.shape[1])
        self.register_buffer('B', B)                      # (K, rank) 可辨识子空间正交基
        # 中性初始化 θ=0 → π=0 → 初始 τ=0（与 SE3PhysicsBranch 中性初始化一致）。
        # θ 由数据唯一确定（Y·B 满列秩，无零空间），收敛后即 θ_true = Bᵀ·π_true。
        self.theta = nn.Parameter(torch.zeros(self.rank))

    def pi(self):
        """π = B·θ ∈ R^K → reshape (n_body, 10)。"""
        return (self.B @ self.theta).reshape(self.n_body, 10)

    def set_pi(self, pi):
        raise RuntimeError('ReparamBranch 的 π 由 θ 决定，不支持 set_pi')

    def inertial_params(self):
        """把重心参数 π=(m, h, I_o) 转回物理量 (m, c, I_c)，供参数核验脚本对比真值。

        c = h/m、I_c = I_o − m[c]×[c]×ᵀ（true_pi 的逆映射）。训练后 m>0，无除零；
        初始 θ=0 时 m=0，c 未定义——本方法仅在核验脚本（训练后）调用，安全。
        """
        pi = self.pi()
        m = pi[:, 0]
        h = pi[:, 1:4]
        I_o = sym_from_vec(pi[:, 4:10])
        # clamp_min 只护住 m≈0 的退化情形（质量≈0 的连杆质心无物理意义，取 0）
        c = h / m.clamp_min(1e-9).unsqueeze(-1)
        cx = skew(c)
        I_c = I_o - m.view(-1, 1, 1) * (cx @ cx.transpose(-1, -2))
        return m, c, I_c
