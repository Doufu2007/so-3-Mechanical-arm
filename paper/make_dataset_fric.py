"""生成"带摩擦 / 带摩擦+测量噪声"的 Franka 数据集。

目的：破掉 R²=1.0 的自证循环。

现状问题：Dataset_Franka_Ex 的标签是 mj_inverse 的精确输出（零噪声），
而 tool/panda.xml 没有 damping / frictionloss / armature（零摩擦）。
SE(3) 分支的模型类精确包含这个生成过程，所以 R²=1.0000 是数值必然，
不是"我的方法好"。

本脚本在同一套激励轨迹上追加两种真实世界效应：

  Dataset_Franka_Fric   刚体 + 关节摩擦（库仑 + 粘滞）
  Dataset_Franka_FN     刚体 + 关节摩擦 + 测量噪声（q / dq / ddq / tau）

摩擦模型用辨识文献的标准形式（Swevers et al. 1997）：

    tau_fric[j] = fc[j] * tanh(dq[j] / v_eps) + fv[j] * dq[j]

用 tanh 而不是 sign，是为了避免零速穿越处的不连续（真实关节也有 Stribeck
过渡区，tanh 是常见的光滑近似）。v_eps 取 0.01 rad/s。

**为什么摩擦加在 Python 里而不是写进 XML**：models.py 的 SE(3) 分支通过
se3_physics.extract_geometry(tool/panda.xml) 抽螺旋轴与零位姿。改 XML 会连带
改动几何提取路径，引入无关变量。在 Python 里加摩擦，几何完全不变，
唯一被改变的就是"标签里多了一项 SE(3) 分支无法表示的力矩"——这正是要测的东西。

关键预期（这就是跑这一轮的目的）：
  * se3_no_uncertainty（纯结构、80 参数、无残差）R² **应该明显掉下来**——
    它没有摩擦项，物理上不可能拟合。如果它还是 1.0000，说明摩擦没真加进去。
  * se3（结构 + 残差分支）应该**保持高位**——残差分支正是为这种未建模效应准备的。
  * 这一升一降就是论文"残差分支为什么必要"的第一份真实证据。
    现在的零摩擦数据里，残差分支根本没活干（franka_ex 上 -0.0152，不显著）。

真实摩擦系数会写进 dataset_stats.json。后续可以把残差分支学到的东西
和真值 fc / fv 对照——那是一张很有说服力的图。

    python paper/make_dataset_fric.py                # 两个数据集都生成
    python paper/make_dataset_fric.py --only fric    # 只生成不带噪声那个
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paper import core
from paper.make_franka_dataset import N_JOINTS, fourier_traj

ROOT = core.ROOT
XML = str(ROOT / 'tool' / 'panda.xml')

# 关节摩擦真值（近端大、远端小，与连杆惯量量级一致）
# 该臂 tau RMS ≈ 9 N·m，故 J1 的库仑摩擦约占 11%——显著但不至于淹没动力学
FC = np.array([1.00, 1.00, 0.70, 0.50, 0.30, 0.15, 0.08])   # 库仑,   N·m
FV = np.array([0.60, 0.60, 0.40, 0.30, 0.20, 0.10, 0.05])   # 粘滞,   N·m·s/rad
V_EPS = 0.01                                                 # tanh 过渡宽度, rad/s

# 测量噪声，按各信号自身标准差的比例给
NOISE_FRAC = {'q': 0.001, 'dq': 0.005, 'ddq': 0.02, 'tau': 0.01}


def friction_torque(dq):
    """库仑 + 粘滞关节摩擦。"""
    return FC * np.tanh(dq / V_EPS) + FV * dq


def label_rigid(q, dq, ddq):
    """mj_inverse 的刚体力矩（与原脚本一致）。"""
    import mujoco
    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    tau = np.zeros_like(q)
    for i in range(len(q)):
        d.qpos[:] = q[i]
        d.qvel[:] = dq[i]
        d.qacc[:] = ddq[i]
        mujoco.mj_inverse(m, d)
        tau[i] = d.qfrc_inverse
    return tau


def add_noise(rng, arr, frac):
    """按列的标准差比例加高斯噪声。"""
    sigma = frac * arr.std(axis=0, keepdims=True)
    return arr + rng.normal(0.0, 1.0, arr.shape) * sigma


# 见 make_franka_dataset.py 同处注释：core.load_arm 用 basename 判 train/test 重叠，
# 三个 split 重名会被误判成数据泄漏。前缀与产出论文数字的那份数据集一致。
SPLIT_PREFIX = {'Training': 'tr', 'Validation': 'va', 'Test': 'te'}


def write_split(out, split, n, rng, duration, dt, with_noise, header):
    allq, alldq, allddq, alltau, allfric = [], [], [], [], []
    for i in range(n):
        f0 = rng.uniform(0.15, 0.4)
        q, dq, ddq = fourier_traj(rng, duration, dt, f0)
        tau_rigid = label_rigid(q, dq, ddq)
        tau_fric = friction_torque(dq)
        tau = tau_rigid + tau_fric

        if with_noise:
            # 噪声加在摩擦之后：传感器测到的是含摩擦的真实力矩
            q = add_noise(rng, q, NOISE_FRAC['q'])
            dq = add_noise(rng, dq, NOISE_FRAC['dq'])
            ddq = add_noise(rng, ddq, NOISE_FRAC['ddq'])
            tau = add_noise(rng, tau, NOISE_FRAC['tau'])

        arr = np.column_stack([q, dq, ddq, tau])
        np.savetxt(out / split / f'{SPLIT_PREFIX[split]}_ex_{i:03d}.csv', arr, delimiter=' ',
                   header=header, comments='', fmt='%.8f')
        allq.append(q); alldq.append(dq); allddq.append(ddq)
        alltau.append(tau); allfric.append(tau_fric)

    Q, DQ, TAU, FRIC = (np.concatenate(x) for x in [allq, alldq, alltau, allfric])
    frac_fric = float(np.sqrt((FRIC ** 2).mean()) / np.sqrt((TAU ** 2).mean()))
    print(f'  {split:11s} {n:2d} 条 / {len(Q):6d} 样本 · '
          f'摩擦占 tau RMS {frac_fric:.1%}')
    return dict(n=int(len(Q)), fric_over_tau_rms=frac_fric,
                tau_std=np.round(TAU.std(0), 4).tolist())


def build(tag, with_noise, a):
    out = ROOT / tag
    for s in ('Training', 'Validation', 'Test'):
        (out / s).mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(a.seed)
    header = (' '.join(f'q{i}' for i in range(7)) + ' ' +
              ' '.join(f'dq{i}' for i in range(7)) + ' ' +
              ' '.join(f'ddq{i}' for i in range(7)) + ' ' +
              ' '.join(f'tau{i}' for i in range(7)))

    print(f'\n>>> {tag}  (摩擦={"是"} 噪声={"是" if with_noise else "否"})')
    stats = {}
    for split, n in [('Training', a.n_train), ('Validation', a.n_val), ('Test', a.n_test)]:
        stats[split] = write_split(out, split, n, rng, a.duration, a.dt,
                                   with_noise, header)

    core.save_json(out / 'dataset_stats.json', {
        'stats': stats,
        'args': vars(a),
        'friction_model': 'tau_fric = fc*tanh(dq/v_eps) + fv*dq  (Swevers 1997 形式)',
        'fc_true': FC.tolist(),
        'fv_true': FV.tolist(),
        'v_eps': V_EPS,
        'measurement_noise_frac': NOISE_FRAC if with_noise else None,
        'base_xml': 'tool/panda.xml (未改动)',
        'provenance': core.provenance(),
    })
    print(f'  写入 {out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', choices=['fric', 'fn'], default=None,
                    help='只生成其中一个数据集（默认两个都生成）')
    ap.add_argument('--n-train', type=int, default=40)
    ap.add_argument('--n-val', type=int, default=8)
    ap.add_argument('--n-test', type=int, default=8)
    ap.add_argument('--duration', type=float, default=10.0)
    ap.add_argument('--dt', type=float, default=0.005)
    ap.add_argument('--seed', type=int, default=2024)
    a = ap.parse_args()

    print('关节摩擦真值：')
    print(f'  fc (库仑)  = {FC.tolist()}  N·m')
    print(f'  fv (粘滞)  = {FV.tolist()}  N·m·s/rad')

    if a.only in (None, 'fric'):
        build('Dataset_Franka_Fric', with_noise=False, a=a)
    if a.only in (None, 'fn'):
        build('Dataset_Franka_FN', with_noise=True, a=a)

    print('\n完成。下一步：python paper/add_arms.py')


if __name__ == '__main__':
    main()
