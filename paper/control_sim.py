"""
闭环控制验证 (P4) —— 学到的物理分量能不能用来控。

任务：**重力补偿定点调节**。给一组目标姿态，用 PD 把机械臂拉到该姿态并在重力下
保持，比较：
  pd             τ = Kp·e + Kd·ė
  pd_grav_learn  + 学到的 ĝ(q)（物理分支输出，只有物理信息模型给得出）
  pd_grav_true   + MuJoCo 真值 g(q)
  pd_grav_mlp    + 黑箱 MLP 输出的「τ(q,0,0)」当补偿（MLP 给不出 g，用静平衡 τ 近似）

为什么用定点调节而不是整段轨迹跟踪：简化 panda 模型没有关节阻尼，轨迹跟踪的
收敛对增益和参考加速度高度敏感，且会与「学到的模型」本身的误差纠缠在一起，
分不清是控制器没调好还是模型不行。定点调节是**对 g(q) 的直接检验**：无补偿时
重力会拉着机械臂下垂，补偿越准，稳态误差越小。

    python paper/control_sim.py --ckpt-phys results_v2/enc_none_franka_ex_seed0.pth \
                                --ckpt-mlp  results_v2/mlp_franka_ex_seed0.pth
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paper import core
from paper.core import ARMS, provenance, save_json
from paper.verify_physics import load_ckpt

ROOT = core.ROOT
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
XML = str(ROOT / 'tool' / 'panda.xml')

# 关节 7 的扭矩 std≈1e-3 N·m（末端只挂小球、绕自身轴无重力矩），
# 对它做重力补偿没有意义且数值不稳，全程锁住、不计入指标。
FREE_JOINTS = np.arange(6)


def learned_gravity(model, q):
    qt = torch.tensor(q, dtype=torch.float32, device=DEVICE).requires_grad_(True)
    with torch.enable_grad():
        g = model.gravity_phys(qt).detach().cpu().numpy()
    g[:, 6] = 0.0
    return g


def mlp_static_torque(model, q):
    """黑箱 MLP 给不出 g(q)；用它在 q̇=q̈=0 时的输出当作补偿，代表黑箱能做的极限。"""
    n = len(q)
    z = np.zeros((n, 7), dtype=np.float32)
    qt = torch.tensor(q, dtype=torch.float32, device=DEVICE)
    zt = torch.tensor(z, device=DEVICE)
    with torch.enable_grad():
        t = model.predict_phys(qt.clone().requires_grad_(True), zt, zt).detach().cpu().numpy()
    t[:, 6] = 0.0
    return t


def true_gravity(q):
    import mujoco
    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    out = np.zeros_like(q)
    for i in range(len(q)):
        d.qpos[:] = q[i]; d.qvel[:] = 0; d.qacc[:] = 0
        mujoco.mj_inverse(m, d)
        out[i] = d.qfrc_inverse
    out[:, 6] = 0.0
    return out


def setpoint_sim(target, comp_fn, kp, kd, steps=3000):
    """从零位形用 PD 拉到 target 并在重力下保持，返回稳态误差与力矩。

    简化 panda 模型原 XML 没有关节阻尼，纯 PD 在无阻尼+大初始误差下数值发散。
    这里给 dof_damping 加一个小的黏性阻尼（真实关节传动都有），并把积分步长降到
    2 ms —— 这是对「被控对象」的合理物理修正，不是对控制器的调参；无补偿基线
    仍按同一对象跑，对比公平。
    """
    import mujoco
    m = mujoco.MjModel.from_xml_path(XML)
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


def run(ckpt_phys, ckpt_mlp, kp=20.0, n_postures=8, target_range=0.6,
        out='experimental_results/v2_control/control.json'):
    model_phys, cfg, arm, _, _, _ = load_ckpt(ckpt_phys)
    model_mlp, _, _, _, _, _ = load_ckpt(ckpt_mlp)
    assert arm.startswith('franka'), '只在有解析真值的仿真臂上做'

    rng = np.random.RandomState(1234)
    targets = rng.uniform(-target_range, target_range, (n_postures, 7)).astype(np.float32)
    targets[:, 6] = 0.0
    kd = 2.0 * np.sqrt(kp) * 0.7

    controllers = {
        'pd': lambda q: np.zeros((len(q), 7)),
        'pd_grav_mlp': lambda q: mlp_static_torque(model_mlp, q),
        'pd_grav_learn': lambda q: learned_gravity(model_phys, q),
        'pd_grav_true': lambda q: true_gravity(q),
    }

    records = []
    print(f'kp={kp:g}, kd={kd:.2f}, {n_postures} 个目标姿态\n'
          f'{"controller":<18s}{"稳态RMSE(mrad)":>16s}{"最大误差(mrad)":>16s}')
    for c, comp_fn in controllers.items():
        rmses, maxes = [], []
        for i, tgt in enumerate(targets):
            rmse, maxe, ok = setpoint_sim(tgt, comp_fn, kp, kd)
            rmses.append(rmse); maxes.append(maxe)
            records.append({'controller': c, 'posture': i, 'kp': kp, 'stable': ok,
                            'rmse_rad': rmse, 'max_abs_err_rad': maxe})
        m = np.mean(rmses); s = np.std(rmses, ddof=1) if len(rmses) > 1 else 0.0
        print(f'{c:<18s}{m*1000:>14.2f}±{s*1000:<10.2f}{np.mean(maxes)*1000:>14.2f}')

    res = {
        'task': 'gravity-compensated setpoint regulation',
        'controllers': list(controllers), 'kp': kp, 'kd': kd,
        'n_postures': n_postures, 'records': records,
        'summary': {c: {'rmse_mean': float(np.mean([r['rmse_rad'] for r in records
                                                   if r['controller'] == c])),
                        'rmse_std': float(np.std([r['rmse_rad'] for r in records
                                                 if r['controller'] == c], ddof=1))}
                    for c in controllers},
        'ckpt_phys': str(ckpt_phys), 'ckpt_mlp': str(ckpt_mlp),
        'plant': 'MuJoCo tool/panda.xml（简化 7 轴串联臂，无关节阻尼；J7 锁住不计入）',
        'provenance': provenance(),
    }
    save_json(ROOT / out, res)
    print(f'\n结果写入 {ROOT / out}')
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-phys', required=True)
    ap.add_argument('--ckpt-mlp', required=True)
    ap.add_argument('--kp', type=float, default=100.0)
    ap.add_argument('--n-postures', type=int, default=8)
    ap.add_argument('--target-range', type=float, default=0.6)
    ap.add_argument('--out', default='experimental_results/v2_control/control.json')
    a = ap.parse_args()
    run(ROOT / a.ckpt_phys, ROOT / a.ckpt_mlp, kp=a.kp, n_postures=a.n_postures,
        target_range=a.target_range, out=a.out)


if __name__ == '__main__':
    main()
