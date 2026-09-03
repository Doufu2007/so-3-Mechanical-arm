"""
!! 已废弃，勿用 —— 请改用 `paper/baseline_ls_exact.py`。!!

本脚本从未成功产出过任何结果：它用有限差分扰动 MuJoCo 的 `m.body_inertia` 字段来
构造回归量，而 MuJoCo 的 `body_inertia` 只有 3 列（3 个主转动惯量，方向另存于
`body_iquat`），所以写到第 4 个分量时必然 `IndexError: index 3 is out of bounds`。
手稿曾把它的「结论」（最小二乘能把真值恢复到浮点精度）当既成事实引用 ——
那句话没有任何实验支撑，且事后证明是**错的**：用精确解析回归量做无偏最小二乘，
力矩 R²=1.00000000，而 100% 的参数误差落在零空间（`baseline_ls_exact.py`）。

保留此文件仅为记录该错误的来源。

---

Swevers 经典刚体辨识基线 —— 线性最小二乘逆动力学参数辨识。

    python paper/baseline_swevers.py --arm franka_ex
    python paper/baseline_swevers.py --arm franka

背景：审稿人会问「为什么不和经典刚体辨识（Swevers et al., 1997）比？」。
本脚本实现这条基线，并只对**有解析刚体模型（tool/panda.xml）的仿真臂**可用
（franka / franka_ex）。Baxter 是实机数据、无解析模型，不适用——这也正是
PINN 残差分支的价值所在（实机上可吸收摩擦/柔性等刚体模型外的东西）。

方法：τ = Y(q,dq,ddq)·θ，θ 是每根运动连杆的 10 个惯性参数
（质量 m、质心 c∈R³、质心系惯性张量 I∈R⁶）。Y 用 MuJoCo 逆动力学对这些参数的
中心差分数值构造——中心差分对 τ 里的 m·c²（Steiner 平行轴）二次项也精确，
故等价于解析回归量的数值实现。θ 在训练集上最小二乘拟合（伪逆最小范数解，
自动处理不可辨识方向），拟合参数写回 MuJoCo 模型，在验证/测试集上做**精确**
前向预测（跑真模型，不是线性化外推，无数值线性化误差）。

自检（防静默出错）：拟合模型在训练集上的 R² 应 ≈ 1，证明回归量构造正确；
脚本会把「拟合后 R²」和「oracle（真值参数）R²」一起打印，两者应几乎一致。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paper import core
from paper.core import ARMS, load_arm, metrics, provenance, save_json

ROOT = core.ROOT
XML = str(ROOT / 'tool' / 'panda.xml')

# 每个连杆的 10 个惯性参数 → (字段, 下标)
FIELD_DEFS = [('mass', -1)] + \
             [('ipos', j) for j in range(3)] + \
             [('inertia', j) for j in range(6)]

# 中心差分步长（按参数量级缩放，避免有限差分噪声）
EPS = {'mass': 1e-5, 'ipos': 1e-6, 'inertia': 1e-8}


def _moving_bodies(m):
    """返回所有「运动」连杆：有父体（parentid != 0，世界为 0）的 body。
    base 是焊死在 world 上的固定 body（parentid==0），其质量不影响 7 自由度动力学，
    自动排除。link0..link6 + hand（焊在 link6 上的末端负载）都含在内。"""
    return [b for b in range(1, m.nbody) if m.body_parentid[b] != 0]


def _read_field(m, bid, field, idx):
    if field == 'mass':
        return m.body_mass[bid]
    if field == 'ipos':
        return m.body_ipos[bid, idx]
    return m.body_inertia[bid, idx]


def _write_field(m, bid, field, idx, val):
    if field == 'mass':
        m.body_mass[bid] = val
    elif field == 'ipos':
        m.body_ipos[bid, idx] = val
    else:
        m.body_inertia[bid, idx] = val


def tau_at(q, dq, ddq, m, d):
    """对 (q,dq,ddq) 跑 MuJoCo 逆动力学，返回 (N, nv) 力矩。"""
    import mujoco
    out = np.zeros((len(q), m.nv))
    for i in range(len(q)):
        d.qpos[:] = q[i]
        d.qvel[:] = dq[i]
        d.qacc[:] = ddq[i]
        mujoco.mj_inverse(m, d)
        out[i] = d.qfrc_inverse
    return out


def build_regressor(m, d, bodies, q, dq, ddq, eps=EPS):
    """中心差分构造回归量 Y：(N_fit, nv, K)，K = 10·len(bodies)。"""
    import mujoco
    N, nv = len(q), m.nv
    K = 10 * len(bodies)
    Y = np.zeros((N, nv, K))
    cols = []
    for bid in bodies:
        for field, idx in FIELD_DEFS:
            cols.append((bid, field, idx))
    for k, (bid, field, idx) in enumerate(cols):
        base = _read_field(m, bid, field, idx)
        e = eps[field]
        _write_field(m, bid, field, idx, base + e)
        tp = tau_at(q, dq, ddq, m, d)
        _write_field(m, bid, field, idx, base - e)
        tm = tau_at(q, dq, ddq, m, d)
        _write_field(m, bid, field, idx, base)   # 恢复
        Y[:, :, k] = (tp - tm) / (2 * e)
    return Y, cols


def set_params(m, bodies, theta):
    """把扁平参数向量 theta（按 FIELD_DEFS 顺序）写回 MuJoCo 模型。"""
    k = 0
    for bid in bodies:
        for field, idx in FIELD_DEFS:
            _write_field(m, bid, field, idx, theta[k])
            k += 1


def theta_from_model(m, bodies):
    """读取当前模型参数，拼成扁平向量（oracle 用）。"""
    return np.concatenate([[_read_field(m, b, f, i)] for b in bodies for f, i in FIELD_DEFS])


def run(arm, seed=0, n_fit=2000, out_dir='experimental_results/v2'):
    import mujoco
    if arm not in ('franka', 'franka_ex'):
        raise ValueError(f'Swevers 基线只支持仿真臂 franka / franka_ex，got {arm}')

    spec = ARMS[arm]
    data = load_arm(arm)
    q_tr, dq_tr, ddq_tr, tau_tr = data['train']

    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    bodies = _moving_bodies(m)
    K = 10 * len(bodies)
    print(f'[swevers/{arm}] 运动连杆 {len(bodies)} 个，惯性参数 K={K}')

    # 从训练集均匀抽 n_fit 个点构造回归量（覆盖轨迹多样性，控制成本）
    rng = np.random.RandomState(seed)
    idx = np.sort(rng.choice(len(q_tr), min(n_fit, len(q_tr)), replace=False))
    qf, dqf, ddqf, tauf = q_tr[idx], dq_tr[idx], ddq_tr[idx], tau_tr[idx]

    t0 = time.time()
    Y, cols = build_regressor(m, d, bodies, qf, dqf, ddqf)
    print(f'  回归量 Y ({Y.shape[0]}×{Y.shape[1]}×{Y.shape[2]}) 构造 {(time.time()-t0):.1f}s')

    # 最小二乘拟合（最小范数解，自动处理不可辨识方向）
    A = Y.reshape(-1, K)
    b = tauf.reshape(-1)
    theta_fit, _, rank, _ = np.linalg.lstsq(A, b, rcond=1e-9)
    print(f'  拟合完成，rank={rank}/{K}（<{K} 说明存在不可辨识方向）')

    # 自检 1：oracle（真值参数）与拟合参数在训练集上的前向 R²，两者应都 ≈1
    theta_nom = theta_from_model(m, bodies)
    set_params(m, bodies, theta_nom)
    tau_oracle = tau_at(qf, dqf, ddqf, m, d)
    set_params(m, bodies, theta_fit)
    tau_fit_train = tau_at(qf, dqf, ddqf, m, d)
    r2_oracle = metrics(tauf, tau_oracle, spec.joint_names)['r2_global']
    r2_fit_train = metrics(tauf, tau_fit_train, spec.joint_names)['r2_global']
    print(f'  自检：oracle train R²={r2_oracle:.6f}，拟合 train R²={r2_fit_train:.6f}')
    if r2_fit_train < 0.999:
        print('  ⚠ 拟合训练集 R² < 0.999：回归量可能构造有误，请检查！')

    # 验证/测试集：用拟合参数跑真模型，精确预测
    def _eval(split):
        q, dq, ddq, tau = split
        pred = tau_at(q, dq, ddq, m, d)   # 模型当前参数 = theta_fit
        return metrics(tau, pred, spec.joint_names)

    m_val = _eval(data['val'])
    m_test = _eval(data['test'])
    train_time = time.time() - t0

    # 参数相对真值的偏差（诊断不可辨识方向，写进 json）
    param_err = float(np.abs(theta_fit - theta_nom).max())
    param_rel = float(np.abs(theta_fit - theta_nom).mean() /
                      (np.abs(theta_nom).mean() + 1e-9))

    result = {
        'variant': 'swevers', 'arm': arm, 'seed': seed, 'frac': 1.0,
        'cfg': {'method': 'linear least-squares rigid-body identification',
                'regressor': 'central finite-difference of MuJoCo inverse dynamics',
                'n_bodies': len(bodies), 'n_params': K,
                'xml': str(Path(XML).relative_to(ROOT)) if Path(XML).exists() else XML},
        'hp': {'n_fit': int(len(qf)), 'eps': EPS},
        'n_params': K,
        'n_train': int(len(q_tr)), 'n_val': int(len(data['val'][0])),
        'n_test': int(len(data['test'][0])),
        'split_manifest': data['manifest'],
        'best_epoch': 0, 'val_r2_norm': m_val['r2_global'],
        'val': m_val, 'test': m_test,
        'train_time_s': train_time,
        'curve': {'regressor_rank': int(rank),
                  'r2_oracle_train': r2_oracle,
                  'r2_fit_train': r2_fit_train,
                  'param_err_max': param_err, 'param_rel_err': param_rel},
        'normalizer': {},
        'provenance': provenance({'extra': 'Swevers rigid-body identification baseline'}),
    }
    name = f'swevers_{arm}_seed{seed}'
    save_json(ROOT / out_dir / f'{name}.json', result)
    print(f'[done] {name}: test R²_global={m_test["r2_global"]:.6f} '
          f'R²_all={m_test["r2_all_joints"]:.6f} '
          f'R²_active={m_test["r2_active_joints"]:.6f} '
          f'RMSE={m_test["rmse_overall"]:.4f} N·m ({train_time:.1f}s)')
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, choices=['franka', 'franka_ex'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n-fit', type=int, default=2000)
    ap.add_argument('--out-dir', default='experimental_results/v2')
    a = ap.parse_args()
    try:
        run(a.arm, a.seed, a.n_fit, a.out_dir)
    except ImportError as e:
        print(f'缺少依赖（MuJoCo 只在仿真机上可用）：{e}')
        sys.exit(1)


if __name__ == '__main__':
    main()
