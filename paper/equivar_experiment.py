"""
基座姿态等变逆动力学 —— 生死门实验（go/no-go 门）。

四组对照（见 选项B_立项任务清单.md §1）：
  ① 结构模型 se3_no_uncertainty（g 对 g_dir 构造性线性），仅水平训练
  ② 裸黑箱 MLP（不喂重力方向），仅水平训练                         [sanity check]
  ③ 带 g_dir 输入的黑箱 MLP（gravity_input=True），仅水平训练      ← 真正的对手
  ④ 带 g_dir 输入的黑箱 MLP，多姿态训练，留出姿态测试              ← 审稿人必问

判据：
  ✅ ① 换姿态误差不涨（零重训成功）；
  ✅ ③ 换姿态误差明显涨（证明线性结构约束值钱，而非"喂个输入谁都会"）。
  ② 裸黑箱失效只是 sanity check；④ 决定论文定位档位。

用法（先造水平 + 倾斜数据集；倾斜目录命名约定 Dataset_Franka_Ex_Tilt{deg}）：
    python paper/make_franka_dataset.py --out Dataset_Franka_Ex
    python paper/make_franka_dataset.py --out Dataset_Franka_Ex_Tilt30 --tilt-deg 30 --tilt-axis x
    python paper/equivar_experiment.py --tilts 30 --axis x

第四组（多姿态黑箱，需额外生成 15/30/45/60 四个倾斜集）：
    python paper/equivar_experiment.py --train-tilts 15,30,45 --test-tilts 60 --axis x
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))          # paper/（models.py 内部裸导入 se3_physics 需要）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # 项目根（from paper import ...）
from paper import core
from paper.core import Normalizer, metrics, save_json, provenance, rotated_gravity, ROOT
from paper.models import ModelCfg, build_model, compute_loss
from paper.train import set_seed, to_t, r2_norm, DEFAULTS

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
JOINTS = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7']

GRAV_LEVEL = np.array([0.0, 0.0, -9.81])   # 水平基座的重力方向（基座系）


# ----------------------------------------------------------------------------
# 数据
# ----------------------------------------------------------------------------

def tilted_dir(deg):
    return f'Dataset_Franka_Ex_Tilt{deg}'


def load_split(dir_name, split):
    """直接读某数据集目录的 split，返回 (q, dq, ddq, tau) 物理单位数组。"""
    import pandas as pd
    d = ROOT / dir_name
    files = sorted((d / split).glob('*.csv'))
    if not files:
        raise FileNotFoundError(f'{dir_name}/{split} 无数据。先跑 make_franka_dataset.py 生成。')
    arrs = [pd.read_csv(f, sep=r'\s+', header=0).values.astype(np.float32) for f in files]
    A = np.concatenate(arrs, axis=0)
    return A[:, :7], A[:, 7:14], A[:, 14:21], A[:, 21:28]


def read_gravity(dir_name, deg, axis):
    """从 dataset_stats.json 读重力方向；缺失则按 tilt 角度解析计算（deg=0 → 水平）。"""
    p = ROOT / dir_name / 'dataset_stats.json'
    if p.exists():
        with open(p, encoding='utf-8') as f:
            g = json.load(f).get('gravity')
        if g:
            return np.array(g, dtype=np.float64)
    return rotated_gravity(deg, axis)


# ----------------------------------------------------------------------------
# 训练 / 评估
# ----------------------------------------------------------------------------

def predict_gdir(model, q, dq, ddq, g_dir=None, chunk=2048):
    """分块前向（SE(3) 分支需要 autograd），g_dir 为 (3,) 物理重力方向（numpy 或 tensor 均可）。"""
    model.eval()
    if g_dir is None:
        gd = None
    elif isinstance(g_dir, torch.Tensor):
        gd = g_dir.to(DEVICE)
    else:
        gd = to_t(g_dir)                                  # (3,)，模型内部自行 expand 到 (B,3)
    outs = []
    for i in range(0, len(q), chunk):
        p, _ = model(q[i:i + chunk].clone().requires_grad_(True),
                     dq[i:i + chunk], ddq[i:i + chunk], g_dir=gd)
        outs.append(p.detach())
    return torch.cat(outs)


def train_model(cfg, train, val, norm, g_dir_train, g_dir_val, seed, hp, label):
    """训练单个模型。选模只看 val。

    g_dir_train / g_dir_val 三选一：
      * None         —— 模型不喂重力方向（裸黑箱）
      * (3,)         —— 全程同一个重力方向（水平训练的结构模型 / 带 g_dir 黑箱）
      * (N,3)        —— 逐样本重力方向（多姿态黑箱，组④）
    """
    q_tr, dq_tr, ddq_tr, tau_tr = train
    Xq, Xdq, Xddq, Xtau = (to_t(q_tr), to_t(dq_tr), to_t(ddq_tr), to_t(norm.norm_tau(tau_tr)))

    per_sample = g_dir_train is not None and np.asarray(g_dir_train).ndim == 2
    if per_sample:
        gd_tr = to_t(g_dir_train)
        ds = torch.utils.data.TensorDataset(Xq, Xdq, Xddq, Xtau, gd_tr)
    else:
        gd_const = None if g_dir_train is None else to_t(g_dir_train)
        ds = torch.utils.data.TensorDataset(Xq, Xdq, Xddq, Xtau)
    loader = torch.utils.data.DataLoader(ds, batch_size=hp['batch_size'], shuffle=True,
                                         drop_last=len(q_tr) > hp['batch_size'])
    Xva = (to_t(val[0]), to_t(val[1]), to_t(val[2]), to_t(norm.norm_tau(val[3])))
    Xva_phys = to_t(val[3])                     # 物理单位验证目标
    scale = to_t(norm.tau_scale)                # (7,) 每关节扭矩尺度，物理损失/物理R²用
    tau_mean_t = to_t(norm.tau_mean)

    set_seed(seed)
    model = build_model(cfg, 7, norm, axes='zyzyzyz').to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())

    opt = torch.optim.AdamW(model.parameters(), lr=hp['lr'], weight_decay=hp['weight_decay'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=hp['epochs'])

    best = {'val_r2': -1e9, 'epoch': -1, 'state': None}
    t0 = time.time()
    for ep in range(1, hp['epochs'] + 1):
        model.train()
        total = 0.0
        for batch in loader:
            if per_sample:
                qb, dqb, ddqb, taub, gdb = batch
            else:
                qb, dqb, ddqb, taub = batch
                gdb = gd_const
            opt.zero_grad(set_to_none=True)
            pred, comps = model(qb.clone().requires_grad_(True), dqb, ddqb, g_dir=gdb)
            if cfg.phys_loss:
                # 物理单位损失：归一化残差 × tau_scale = 物理残差（offset 相消），
                # 避免归一化空间放大 J1/J7 退化关节、饿死 J2/J3 大扭矩关节。
                loss = F.huber_loss(pred * scale, taub * scale, delta=1.0)
            else:
                loss, _ = compute_loss(pred, taub, comps, cfg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), hp['grad_clip'])
            opt.step()
            total += float(loss)
        sched.step()
        if ep % hp['eval_every'] == 0 or ep == hp['epochs'] or ep == 1:
            vp = predict_gdir(model, Xva[0], Xva[1], Xva[2], g_dir_val)
            if cfg.phys_loss:
                vp_phys = vp * scale + tau_mean_t
                ss_res = ((vp_phys - Xva_phys) ** 2).sum()
                ss_tot = ((Xva_phys - Xva_phys.mean(0)) ** 2).sum()
                vr2 = float(1.0 - ss_res / ss_tot.clamp_min(1e-12))
            else:
                vr2 = r2_norm(vp, Xva[3])
            if vr2 > best['val_r2']:
                best.update(val_r2=vr2, epoch=ep,
                            state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            print(f'  {label} ep{ep:4d} loss={total/max(len(loader),1):.5f} '
                  f'val_r2={vr2:.4f} best={best["val_r2"]:.4f} {time.time()-t0:.0f}s', flush=True)

    model.load_state_dict(best['state'])
    return model, best['val_r2'], n_params


def eval_orientation(model, te, g_te, norm):
    """物理单位下的指标。te=(q,dq,ddq,tau) numpy，g_te=(3,) 或 None。返回 core.metrics dict。"""
    pred_n = predict_gdir(model, to_t(te[0]), to_t(te[1]), to_t(te[2]), g_te)
    pred = norm.denorm_tau(pred_n.cpu().numpy())
    return metrics(te[3], pred, JOINTS)


# ----------------------------------------------------------------------------
# 组定义
# ----------------------------------------------------------------------------

def cfg_structured():
    return ModelCfg(encoding='none', physics_kind='se3', uncertainty=False, adaptive_lambda=False)


def cfg_mlp():
    return ModelCfg(kind='mlp', multiscale_loss=False, phys_loss=True)


def cfg_mlp_grav():
    return ModelCfg(kind='mlp', gravity_input=True, multiscale_loss=False, phys_loss=True)


# ----------------------------------------------------------------------------
# 组 ①②③：核心 go/no-go
# ----------------------------------------------------------------------------

def run_groups_123(axis, tilts, seed, hp):
    data = core.load_arm('franka_ex')
    train, val, test_h = data['train'], data['val'], data['test']
    g_h = read_gravity('Dataset_Franka_Ex', 0, axis)

    norm = Normalizer(*train)

    test_orientations = {'0deg': (test_h, g_h)}
    for deg in tilts:
        te = load_split(tilted_dir(deg), 'Test')
        test_orientations[f'{deg}deg'] = (te, read_gravity(tilted_dir(deg), deg, axis))

    groups = [
        ('structured', cfg_structured(), g_h, g_h),   # 训练/验证都喂水平 g_dir
        ('mlp_bare',   cfg_mlp(),       None, None),  # 不喂重力方向
        ('mlp_grav',   cfg_mlp_grav(),  g_h, g_h),    # 训练/验证喂水平 g_dir（模型内部单位化）
    ]

    results = {}
    for name, cfg, gd_tr, gd_val in groups:
        print(f'\n===== 训练 {name} =====')
        model, best_val, n_params = train_model(cfg, train, val, norm, gd_tr, gd_val,
                                                seed, hp, name)
        row = {'best_val_r2': best_val, 'n_params': n_params, 'test_r2_global': {}}
        for oname, (te, g_te) in test_orientations.items():
            m = eval_orientation(model, te, g_te, norm)
            row['test_r2_global'][oname] = m['r2_global']
            print(f'  {name} @ {oname}: r2_global={m["r2_global"]:.4f}')
        results[name] = row

    return results, test_orientations


# ----------------------------------------------------------------------------
# 组 ④：多姿态黑箱（审稿人必问）
# ----------------------------------------------------------------------------

def run_group_4(axis, train_tilts, test_tilts, seed, hp):
    g_h = read_gravity('Dataset_Franka_Ex', 0, axis)

    # 训练数据 = 水平 Training + 各 train_tilt 的 Training，g_dir 逐样本变化
    splits = [load_split('Dataset_Franka_Ex', 'Training')]
    g_list = [np.broadcast_to(g_h, (len(splits[0][0]), 3))]
    for deg in train_tilts:
        s = load_split(tilted_dir(deg), 'Training')
        splits.append(s)
        g_list.append(np.broadcast_to(read_gravity(tilted_dir(deg), deg, axis), (len(s[0]), 3)))
    q_tr = np.concatenate([s[0] for s in splits])
    dq_tr = np.concatenate([s[1] for s in splits])
    ddq_tr = np.concatenate([s[2] for s in splits])
    tau_tr = np.concatenate([s[3] for s in splits])
    g_tr = np.concatenate(g_list)

    norm = Normalizer(q_tr, dq_tr, ddq_tr, tau_tr)
    val = load_split('Dataset_Franka_Ex', 'Validation')   # 水平验证，仅选模

    print(f'\n===== 训练 mlp_grav_multiorient（训练姿态 {[0] + train_tilts}°） =====')
    model, best_val, n_params = train_model(cfg_mlp_grav(), (q_tr, dq_tr, ddq_tr, tau_tr),
                                            val, norm, g_tr, g_h, seed, hp, 'mlp_grav_multiorient')

    row = {'best_val_r2': best_val, 'n_params': n_params, 'train_tilts': train_tilts,
           'test_r2_global': {}}
    # 留出姿态（不含在训练里）+ 水平，各测一次
    for deg in test_tilts:
        te = load_split(tilted_dir(deg), 'Test')
        g_te = read_gravity(tilted_dir(deg), deg, axis)
        m = eval_orientation(model, te, g_te, norm)
        row['test_r2_global'][f'{deg}deg'] = m['r2_global']
        print(f'  mlp_grav_multiorient @ {deg}deg: r2_global={m["r2_global"]:.4f}')
    return row


# ----------------------------------------------------------------------------
# 判定
# ----------------------------------------------------------------------------

def verdict(results):
    """打印生死门判定表。r2 单位：全局 R²（越高越好）。"""
    print('\n' + '=' * 70)
    print('生死门判定（全局 R²，越高越好）')
    print('=' * 70)
    orient = list(next(iter(results.values()))['test_r2_global'].keys())
    hdr = '模型'.ljust(24) + ''.join(o.rjust(12) for o in orient)
    print(hdr)
    print('-' * len(hdr))
    for name, row in results.items():
        cells = ''.join(f'{row["test_r2_global"][o]:12.4f}' for o in orient)
        print(f'{name:<24}{cells}')
    print('=' * 70)

    # 自动判定（仅对 ①②③）
    if 'structured' in results and 'mlp_grav' in results:
        r = results
        h = orient[0]                       # 水平
        t = orient[-1]                      # 最后一个倾斜姿态
        d_structured = r['structured']['test_r2_global'][t] - r['structured']['test_r2_global'][h]
        d_mlp_grav = r['mlp_grav']['test_r2_global'][t] - r['mlp_grav']['test_r2_global'][h]
        print(f'\n① 结构模型 换姿态 ΔR² = {d_structured:+.4f}（应 ≈0，零重训成立）')
        print(f'③ 黑箱+g_dir 换姿态 ΔR² = {d_mlp_grav:+.4f}（应明显 <0，结构约束值钱）')
        ok_struct = d_structured > -0.02            # 几乎不涨误差
        ok_grav = d_mlp_grav < -0.05                # 明显变差
        if ok_struct and ok_grav:
            print('✅ 生死门通过：结构等变零重训成立，且黑箱喂 g_dir 也学不会 ⇒ 结构约束有独立价值。')
        elif ok_struct and not ok_grav:
            print('❌ 生死门失败：③ 黑箱喂 g_dir 也零重训成功 ⇒ "喂个输入谁都会"，结构约束卖点塌。')
        elif not ok_struct:
            print('❌ 生死门失败：结构模型自己换姿态就涨误差 ⇒ 等变实现有 bug 或假设不成立。')
        return {'delta_structured': d_structured, 'delta_mlp_grav': d_mlp_grav,
                'passed': ok_struct and ok_grav}
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--axis', default='x', choices=['x', 'y', 'z'])
    ap.add_argument('--tilts', default='30', help='组①②③的测试倾斜角（逗号分隔）')
    ap.add_argument('--train-tilts', default='', help='组④的多姿态训练角（逗号分隔，留空跳过组④）')
    ap.add_argument('--test-tilts', default='', help='组④的留出测试角（逗号分隔）')
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--batch-size', type=int, default=None)
    ap.add_argument('--lr', type=float, default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out-dir', default='experimental_results/equivar')
    a = ap.parse_args()

    hp = dict(DEFAULTS)
    if a.epochs:
        hp['epochs'] = a.epochs
    if a.batch_size:
        hp['batch_size'] = a.batch_size
    if a.lr:
        hp['lr'] = a.lr

    tilts = [int(x) for x in a.tilts.split(',') if x.strip()]
    results123, orientations = run_groups_123(a.axis, tilts, a.seed, hp)

    out = {'axis': a.axis, 'tilts': tilts, 'seed': a.seed, 'hp': hp,
           'groups_123': results123, 'verdict': verdict(results123)}

    train_tilts = [int(x) for x in a.train_tilts.split(',') if x.strip()] if a.train_tilts else []
    test_tilts = [int(x) for x in a.test_tilts.split(',') if x.strip()] if a.test_tilts else []
    if train_tilts and test_tilts:
        overlap = set(train_tilts) & set(test_tilts)
        if overlap:
            raise ValueError(f'留出姿态 {overlap} 与训练姿态重叠，组④无效')
        out['group_4'] = run_group_4(a.axis, train_tilts, test_tilts, a.seed, hp)

    out['provenance'] = provenance()
    g4 = f'_g4train{"-".join(map(str, train_tilts))}_test{"-".join(map(str, test_tilts))}' if train_tilts else ''
    name = f'equivar_axis{a.axis}_tilts{"-".join(map(str, tilts))}_seed{a.seed}{g4}'
    save_json(ROOT / a.out_dir / f'{name}.json', out)
    print(f'\n结果写入 {a.out_dir}/{name}.json')


if __name__ == '__main__':
    main()
