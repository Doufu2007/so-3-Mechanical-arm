"""
统一实验框架 — 数据、划分、归一化、指标、溯源。

设计原则（针对审稿人会查的点）：
  1. 轨迹级 train / val / test 三划分，**验证集选模，测试集只用一次**
     （旧代码用 test 集上的 best R² 做选模，属于测试集泄漏）
  2. 归一化统计量只从 train 划分估计
  3. 指标同时报告 全关节均值 / 活跃关节均值 / 逐关节
     （旧代码两个版本口径不一致：run_unified 全关节、franka 版剔除静态关节）
  4. 每个结果 json 都带完整溯源：git commit、seed、硬件、耗时、复现命令
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ----------------------------------------------------------------------------
# 机械臂配置
# ----------------------------------------------------------------------------

@dataclass
class ArmSpec:
    name: str
    train_dir: str
    test_dir: str
    has_ddq: bool          # 数据里是否直接给了加速度；否则中心差分
    dt: float
    joint_names: list
    axes: str              # 名义关节轴序列，用于 rotmat 编码（product of exponentials）
    n_val_traj: int        # 从 Training 目录里划出多少条轨迹做验证集（val_dir 为空时生效）
    val_dir: str = ''      # 已经单独存好验证集时直接指定
    note: str = ''

ARMS = {
    'baxter': ArmSpec(
        name='baxter',
        train_dir='Dataset/Training', test_dir='Dataset/Test',
        has_ddq=False, dt=0.005,
        joint_names=['S0', 'S1', 'E0', 'E1', 'W0', 'W1', 'W2'],
        # Baxter 左臂标称轴序：肩偏航(z)-肩俯仰(y)-上臂滚转(x)-肘俯仰(y)-前臂滚转(x)-腕俯仰(y)-腕滚转(x)
        axes='zyxyxyx',
        n_val_traj=3,
    ),
    'franka': ArmSpec(
        name='franka',
        train_dir='Dataset_Franka/Training', test_dir='Dataset_Franka/Test',
        has_ddq=True, dt=0.005,
        joint_names=['J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7'],
        # 与 tool/panda.xml 一致（简化 7 轴串联链）
        axes='zyzyzyz',
        n_val_traj=3,
        note='原始数据：15 条以 q=0 为中心的低速轨迹，位形空间覆盖极差，'
             '测试集相对训练集是外推。保留它作为「窄覆盖/外推」工况。',
    ),
    'franka_ex': ArmSpec(
        name='franka_ex',
        train_dir='Dataset_Franka_Ex/Training', test_dir='Dataset_Franka_Ex/Test',
        val_dir='Dataset_Franka_Ex/Validation',
        has_ddq=True, dt=0.005,
        joint_names=['J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7'],
        axes='zyzyzyz',
        n_val_traj=0,
        note='有限傅里叶级数激励轨迹，中心位形随机覆盖关节量程，|q̇|≤3 rad/s，'
             '动力学项 RMS 已超过重力项（比值 1.26）。见 paper/make_franka_dataset.py。',
    ),
}


# ----------------------------------------------------------------------------
# 基座姿态等变：重力方向工具
# ----------------------------------------------------------------------------

GRAVITY_WORLD = np.array([0.0, 0.0, -9.81])

def rotated_gravity(tilt_deg, axis='x'):
    """基座绕 axis 转 tilt_deg 后，重力方向在**基座系**里的表示 (3,)。

    世界系重力 g_world 固定，基座绕 axis 转 θ ⇒ 基座系里看到的重力 = R_axis(−θ)·g_world。
    这是「基座重定向」实验的核心：同一套模型，换 g_dir 就换安装姿态，零重训。
    """
    th = np.radians(tilt_deg)
    c, s = np.cos(th), np.sin(th)
    R = {'x': np.array([[1, 0, 0], [0, c, s], [0, -s, c]]),
         'y': np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]]),
         'z': np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])}.get(axis)
    if R is None:
        raise ValueError(f'未知 tilt-axis: {axis}（x/y/z）')
    return R @ GRAVITY_WORLD


# ----------------------------------------------------------------------------
# 数据加载
# ----------------------------------------------------------------------------

def _read_traj(path: Path, spec: ArmSpec):
    import pandas as pd
    arr = pd.read_csv(path, sep=r'\s+', header=0).values.astype(np.float32)
    if spec.has_ddq:
        q, dq, ddq, tau = arr[:, 0:7], arr[:, 7:14], arr[:, 14:21], arr[:, 21:28]
    else:
        q, dq, tau = arr[:, 0:7], arr[:, 7:14], arr[:, 14:21]
        ddq = np.zeros_like(dq)
        dt = spec.dt
        ddq[1:-1] = (dq[2:] - dq[:-2]) / (2 * dt)
        ddq[0] = (dq[1] - dq[0]) / dt
        ddq[-1] = (dq[-1] - dq[-2]) / dt
    return dict(q=q, dq=dq, ddq=ddq, tau=tau, name=path.name, n=len(q))


def load_arm(arm: str, val_seed: int = 12345):
    """返回 {'train':..., 'val':..., 'test':...}，每个是 (q, dq, ddq, tau) 元组。

    验证集从 Training 目录里按固定随机种子抽出整条轨迹（轨迹级划分，避免
    同一条轨迹的相邻帧同时出现在 train 和 val 里造成乐观偏差）。
    """
    spec = ARMS[arm]
    train_files = sorted((ROOT / spec.train_dir).glob('*.csv'))
    test_files = sorted((ROOT / spec.test_dir).glob('*.csv'))
    if not train_files or not test_files:
        raise FileNotFoundError(f'{arm}: 找不到数据 {spec.train_dir} / {spec.test_dir}')

    names_tr = {f.name for f in train_files}
    names_te = {f.name for f in test_files}
    overlap = names_tr & names_te
    if overlap:
        raise RuntimeError(f'{arm}: train/test 文件重叠 {overlap}')

    parts = {'train': [], 'val': [], 'test': []}
    manifest = {'train': [], 'val': [], 'test': []}

    if spec.val_dir:
        val_files = sorted((ROOT / spec.val_dir).glob('*.csv'))
        if not val_files:
            raise FileNotFoundError(f'{arm}: 找不到验证集 {spec.val_dir}')
        for f in val_files:
            parts['val'].append(_read_traj(f, spec))
            manifest['val'].append(f.name)
        for f in train_files:
            parts['train'].append(_read_traj(f, spec))
            manifest['train'].append(f.name)
    else:
        rng = np.random.RandomState(val_seed)
        idx = rng.permutation(len(train_files))
        val_idx = set(idx[:spec.n_val_traj].tolist())
        for i, f in enumerate(train_files):
            key = 'val' if i in val_idx else 'train'
            parts[key].append(_read_traj(f, spec))
            manifest[key].append(f.name)

    for f in test_files:
        parts['test'].append(_read_traj(f, spec))
        manifest['test'].append(f.name)

    out = {}
    for key, trajs in parts.items():
        out[key] = tuple(np.concatenate([t[k] for t in trajs]) for k in ['q', 'dq', 'ddq', 'tau'])
    out['manifest'] = manifest
    out['trajs'] = parts
    out['spec'] = spec
    return out


# ----------------------------------------------------------------------------
# 归一化
# ----------------------------------------------------------------------------

TAU_SCALE_FLOOR_RATIO = 0.02   # 扭矩尺度下限 = 该臂最大关节扭矩 std 的 2%


class Normalizer:
    """统计量只从 train 划分估计。模型内部持有这些常数。

    设计要点：
      * **物理分支吃物理单位**（rad, rad/s, rad/s², N·m），H、c、g 因此有真实量纲，
        可直接用于控制；对称正定性也是对物理 H 成立，而不是对某个缩放后的矩阵。
      * **黑箱分支吃 z-score 输入**，保证条件数正常（Baxter 腕关节 σ_q 很小，
        若沿用物理单位或按 σ_q 缩放，ddq 的 p99 会到 10²~10³ 量级，MLP 根本没法学）。
      * **扭矩尺度设下限**：Franka 的 J1/J7 扭矩 std 只有 3.5e-2 / 1e-3 N·m，
        直接除以它会让损失被这两个数值退化的关节完全主导。下限取该臂最大关节
        std 的 2%。注意逐关节 R² 与尺度无关，该处理只影响训练目标，不粉饰指标。
    """

    def __init__(self, q, dq, ddq, tau):
        f = lambda x: x.astype(np.float32)
        self.q_mean, self.q_std = f(q.mean(0)), f(q.std(0) + 1e-8)
        self.dq_mean, self.dq_std = f(dq.mean(0)), f(dq.std(0) + 1e-8)
        self.ddq_mean, self.ddq_std = f(ddq.mean(0)), f(ddq.std(0) + 1e-8)
        self.tau_mean = f(tau.mean(0))
        raw = tau.std(0)
        self.tau_scale = f(np.maximum(raw, TAU_SCALE_FLOOR_RATIO * raw.max()) + 1e-8)
        self.tau_std_raw = f(raw)

    def norm_tau(self, tau):
        return (tau - self.tau_mean) / self.tau_scale

    def denorm_tau(self, tau_n):
        return tau_n * self.tau_scale + self.tau_mean

    def state_dict(self):
        return {k: getattr(self, k).tolist() for k in
                ['q_mean', 'q_std', 'dq_mean', 'dq_std', 'ddq_mean', 'ddq_std',
                 'tau_mean', 'tau_scale', 'tau_std_raw']}


# ----------------------------------------------------------------------------
# 指标
# ----------------------------------------------------------------------------

# 「活跃关节」判据：扭矩标准差不低于该臂最大关节扭矩标准差的 5%。
# 用相对判据而不是绝对阈值：Franka 的 J7 扭矩 std 只有 9e-4 N·m（末端只挂一个
# 小球，绕自身轴既无重力矩也几乎无惯量），它的 R² 是纯数值噪声，能到 -10⁴，
# 放进任何平均值里都会把整张表变成噪声表。
ACTIVE_STD_RATIO = 0.05

def metrics(tau_true: np.ndarray, tau_pred: np.ndarray, joint_names=None):
    """真实物理单位下的指标。返回逐关节 + 两种平均口径。"""
    n = tau_true.shape[1]
    joint_names = joint_names or [f'J{i+1}' for i in range(n)]
    res = tau_true - tau_pred
    ss_res = (res ** 2).sum(0)
    ss_tot = ((tau_true - tau_true.mean(0)) ** 2).sum(0)
    r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)
    rmse = np.sqrt((res ** 2).mean(0))
    rng = tau_true.max(0) - tau_true.min(0)
    nrmse = rmse / np.maximum(rng, 1e-12)
    std = tau_true.std(0)
    active = std >= ACTIVE_STD_RATIO * std.max()
    # 全局 R²：把所有关节的残差和方差合并统计，等价于按扭矩量级加权，
    # 不会被数值退化关节（Franka J1/J7）主导，是三种口径里最保守也最不易被质疑的
    r2_global = float(1.0 - ss_res.sum() / max(ss_tot.sum(), 1e-12))
    return {
        'r2_global': r2_global,
        'r2_per_joint': {joint_names[i]: float(r2[i]) for i in range(n)},
        'rmse_per_joint': {joint_names[i]: float(rmse[i]) for i in range(n)},
        'nrmse_per_joint': {joint_names[i]: float(nrmse[i]) for i in range(n)},
        'tau_std_per_joint': {joint_names[i]: float(std[i]) for i in range(n)},
        'r2_all_joints': float(r2.mean()),
        'r2_active_joints': float(r2[active].mean()) if active.any() else float('nan'),
        'active_joints': [joint_names[i] for i in range(n) if active[i]],
        'rmse_overall': float(np.sqrt((res ** 2).mean())),
        'nrmse_mean': float(nrmse.mean()),
    }


# ----------------------------------------------------------------------------
# 溯源
# ----------------------------------------------------------------------------

def git_commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return 'unknown'


def provenance(extra=None):
    import torch
    p = {
        'git_commit': git_commit(),
        'python': sys.version.split()[0],
        'torch': torch.__version__,
        'numpy': np.__version__,
        'platform': platform.platform(),
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
        'cuda': torch.version.cuda,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'command': ' '.join([sys.executable] + sys.argv),
    }
    if extra:
        p.update(extra)
    return p


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)
    return path


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f'not serializable: {type(o)}')
