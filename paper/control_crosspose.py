"""
闭环跨姿态验证 —— 把「零重训」从 R² 升级成「换安装姿态后控制器还能不能用」。

任务：重力补偿定点调节（沿用 control_sim.py 的设定，但跨多个安装姿态）。
在 0° / 30° / 60° / 90° 各跑一组目标姿态，比较四种控制器：
    pd             纯 PD，无补偿（下界：无补偿时重力拉下垂）
    pd+struct_g    结构模型 ĝ(q; g_dir) 补偿（等变，1 姿态训练 → 各姿态应保持）
    pd+mlp_g       黑箱 τ(q,0,0; g_dir) 补偿（1 姿态训练 → 倾斜姿态应下垂）
    pd+true_g      MuJoCo 真值 g(q)（上界）

卖点：结构模型「单姿态训练 → 全姿态零重训」从 R² 的软指标，变成「控制器在
    换安装姿态后稳态误差不涨」的硬指标；黑箱即使拿到 g_dir 输入，换姿态补偿失效
    （因为它把重力学进权重，没有约束 g 对 g_dir 线性）。

用法（先确保 Tilt30/60/90 等数据集已生成）：
    python paper/control_crosspose.py --axis x --tilts 30,60,90 --epochs 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))          # paper/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # 项目根
from paper import core
from paper.core import Normalizer, load_arm, rotated_gravity, save_json, provenance, ROOT
from paper.train import set_seed, to_t, DEFAULTS
from paper.equivar_experiment import (train_model, cfg_structured, cfg_mlp_grav,
                                      read_gravity, tilted_dir)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
XML = 'tool/panda.xml'   # MuJoCo VFS 用相对路径（含中文的绝对路径会 ValueError）

# 关节 7 扭矩 std≈1e-3 N·m（末端只挂小球、绕自身轴无重力矩），
# 对它做重力补偿没有意义且数值不稳，全程锁住、不计入指标。
FREE_JOINTS = np.arange(6)


# ----------------------------------------------------------------------------
# 补偿函数（物理单位 N·m，返回 (N,7)）
# ----------------------------------------------------------------------------

def struct_gravity(model, q, g_dir):
    """结构模型 ĝ(q; g_dir) —— 等变：换 g_dir 就是换安装姿态，零重训。"""
    qt = torch.tensor(q, dtype=torch.float32, device=DEVICE).requires_grad_(True)
    with torch.enable_grad():
        g = model.gravity_phys(qt, g_dir=to_t(np.asarray(g_dir, dtype=np.float64))).detach().cpu().numpy()
    g[:, 6] = 0.0
    return g


def mlp_static(model, q, g_dir):
    """黑箱给不出 g(q)；用它在 q̇=q̈=0 时的输出当补偿，代表黑箱能做的极限。"""
    n = len(q)
    z = np.zeros((n, 7), dtype=np.float32)
    qt = torch.tensor(q, dtype=torch.float32, device=DEVICE)
    zt = torch.tensor(z, device=DEVICE)
    with torch.enable_grad():
        t = model(qt.clone().requires_grad_(True), zt, zt,
                  g_dir=to_t(np.asarray(g_dir, dtype=np.float64)))[0]
        t = model.tau_to_phys(t).detach().cpu().numpy()
    t[:, 6] = 0.0
    return t


def true_gravity(q, gravity):
    import mujoco
    m = mujoco.MjModel.from_xml_path(XML)
    m.opt.gravity[:] = gravity
    d = mujoco.MjData(m)
    out = np.zeros_like(q)
    for i in range(len(q)):
        d.qpos[:] = q[i]; d.qvel[:] = 0; d.qacc[:] = 0
        mujoco.mj_inverse(m, d)
        out[i] = d.qfrc_inverse
    out[:, 6] = 0.0
    return out


def setpoint_sim(target, comp_fn, kp, kd, gravity, steps=3000):
    """从零位形用 PD 拉到 target 并在重力下保持，返回稳态误差与力矩。

    简化 panda 模型原 XML 没有关节阻尼，纯 PD 在无阻尼+大初始误差下数值发散。
    这里给 dof_damping 加一个小的黏性阻尼（真实关节传动都有），并把积分步长降到
    2 ms —— 这是对「被控对象」的合理物理修正，不是对控制器的调参；无补偿基线
    仍按同一对象跑，对比公平。
    """
    import mujoco
    m = mujoco.MjModel.from_xml_path(XML)
    m.opt.gravity[:] = gravity
    m.dof_damping[:] = 1.0
    m.dof_armature[:] = 0.05    # 转子/传动惯量：真实关节都有，缺失会导致无阻尼数值发散
    m.opt.timestep = 0.002
    d = mujoco.MjData(m)
    d.qpos[:] = 0.0; d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    comp = comp_fn(target[None, :])[0] if comp_fn is not None else np.zeros(7)
    err_hist = []
    for i in range(steps):
        e = target - d.qpos
        tau = comp + kp * e + kd * (-d.qvel)
        tau[6] = 0.0
        d.qfrc_applied[:] = tau
        mujoco.mj_step(m, d)
        err_hist.append(e[FREE_JOINTS].copy())
        if not np.isfinite(d.qpos).all():
            return float('nan'), float('nan'), False
    err = np.array(err_hist[-1000:])
    rmse = float(np.sqrt((err ** 2).mean()))
    maxe = float(np.abs(err).max())
    return rmse, maxe, True


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------

def run(axis, tilts, seed, hp, kp=100.0, n_postures=8, target_range=0.6,
        out_dir='experimental_results/equivar'):
    # 训练数据 = 水平 0°（单姿态训练，这是零重训的前提）
    data = load_arm('franka_ex')
    train, val = data['train'], data['val']
    g_h = read_gravity('Dataset_Franka_Ex', 0, axis)
    norm = Normalizer(*train)

    print('训练结构模型（仅水平）...')
    model_struct, vs, _ = train_model(cfg_structured(), train, val, norm, g_h, g_h,
                                      seed, hp, 'structured')
    print(f'  结构模型 val_r2={vs:.4f}')
    print('训练黑箱 MLP（g_dir 输入，仅水平）...')
    model_mlp, vm, _ = train_model(cfg_mlp_grav(), train, val, norm, g_h, g_h,
                                   seed, hp, 'mlp_grav')
    print(f'  黑箱 val_r2={vm:.4f}')

    # 姿态列表：0° 用水平数据集的 g；其余用 tilted 数据集的 g（与训练数据同一约定）
    poses = [(0.0, g_h, '0deg')]
    for deg in tilts:
        g = read_gravity(tilted_dir(int(deg)), int(deg), axis)
        poses.append((float(deg), g, f'{int(deg)}deg'))

    kd = 2.0 * np.sqrt(kp) * 0.7
    rng = np.random.RandomState(1234)
    targets = rng.uniform(-target_range, target_range, (n_postures, 7)).astype(np.float32)
    targets[:, 6] = 0.0

    records, summary = [], {}
    for deg, g, name in poses:
        g_arr = np.asarray(g, dtype=np.float64)
        controllers = {
            'pd':          lambda q: np.zeros((len(q), 7)),
            'pd+mlp_g':    lambda q: mlp_static(model_mlp, q, g_arr),
            'pd+struct_g': lambda q: struct_gravity(model_struct, q, g_arr),
            'pd+true_g':   lambda q: true_gravity(q, g_arr),
        }
        print(f'\n===== 姿态 {name}（g_dir = {np.round(g_arr, 2)}）=====')
        print(f'{"controller":<18s}{"稳态RMSE(mrad)":>16s}{"最大误差(mrad)":>16s}')
        for c, comp_fn in controllers.items():
            rmses, maxes = [], []
            for i, tgt in enumerate(targets):
                rmse, maxe, ok = setpoint_sim(tgt, comp_fn, kp, kd, g_arr)
                rmses.append(rmse); maxes.append(maxe)
                records.append({'pose': name, 'g_dir': g_arr.tolist(), 'controller': c,
                                'posture': i, 'stable': ok,
                                'rmse_rad': rmse, 'max_abs_err_rad': maxe})
            m = np.mean(rmses)
            summary[f'{name}:{c}'] = m
            print(f'{c:<18s}{m*1000:>14.2f}±{np.std(rmses, ddof=1)*1000:<10.2f}'
                  f'{np.mean(maxes)*1000:>14.2f}')

    # 关键判据：结构补偿跨姿态误差应≈不涨，黑箱补偿应明显涨
    print('\n' + '=' * 70)
    print('跨姿态漂移（相对 0deg 的稳态 RMSE 变化，单位 mrad）')
    print('=' * 70)
    print(f'{"controller":<18s}' + ''.join(f'{p[2]:>12s}' for p in poses))
    for c in ['pd', 'pd+mlp_g', 'pd+struct_g', 'pd+true_g']:
        base = summary[f'0deg:{c}']
        cells = ''.join(f'{(summary[f"{p[2]}:{c}"]-base)*1000:>12.2f}' for p in poses)
        print(f'{c:<18s}{cells}')

    res = {
        'task': 'gravity-compensated setpoint regulation, cross base-pose',
        'axis': axis, 'tilts': tilts, 'seed': seed, 'kp': kp, 'kd': kd,
        'n_postures': n_postures, 'target_range': target_range,
        'structured_val_r2': vs, 'mlp_grav_val_r2': vm,
        'poses': [{'deg': p[0], 'g_dir': np.asarray(p[1]).tolist(), 'name': p[2]} for p in poses],
        'summary': summary, 'records': records,
        'plant': 'MuJoCo tool/panda.xml（简化 7 轴串联臂，无关节阻尼；J7 锁住不计入）',
        'provenance': provenance(),
    }
    save_json(ROOT / out_dir / f'control_crosspose_axis{axis}_tilts{"-".join(map(str, tilts))}_seed{seed}.json', res)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--axis', default='x', choices=['x', 'y', 'z'])
    ap.add_argument('--tilts', default='30,60,90', help='跨姿态测试角（逗号分隔）')
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--batch-size', type=int, default=None)
    ap.add_argument('--lr', type=float, default=None)
    ap.add_argument('--kp', type=float, default=100.0)
    ap.add_argument('--n-postures', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    hp = dict(DEFAULTS)
    if a.epochs:
        hp['epochs'] = a.epochs
    if a.batch_size:
        hp['batch_size'] = a.batch_size
    if a.lr:
        hp['lr'] = a.lr

    tilts = [int(x) for x in a.tilts.split(',') if x.strip()]
    run(a.axis, tilts, a.seed, hp, kp=a.kp, n_postures=a.n_postures)


if __name__ == '__main__':
    main()
