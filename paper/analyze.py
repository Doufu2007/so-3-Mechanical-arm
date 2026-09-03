"""
结果汇总 + 显著性检验 (P3.3 / P5) —— 从 experimental_results/v2/*.json 生成论文表格。

    python paper/analyze.py

产出：
  experimental_results/v2_tables/table1_main.{json,md}      主对比表（含逐关节）
  experimental_results/v2_tables/table2_ablation.{json,md}  单因素消融 + Welch t 检验
  experimental_results/v2_tables/table3_per_joint.md        逐关节 R²
  experimental_results/v2_tables/stats.json                 全部成对检验明细
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paper import core
from paper.core import ARMS, save_json

ROOT = core.ROOT
IN_DIR = ROOT / 'experimental_results/v2'
OUT_DIR = ROOT / 'experimental_results/v2_tables'

# 主表里出现的方法及展示名
DISPLAY = {
    'full': 'Ours (rotmat, 7-dim bottleneck)',
    'enc_rotmat_wide': 'Ours (rotmat, no bottleneck)',
    'enc_sincos_wide': 'Ours (sin/cos, no bottleneck)',
    'enc_sincos': 'Ours (sin/cos enc.)',
    'enc_none': 'Ours (raw joint angles)',
    'no_physics': 'Black-box FFNN (no physics)',
    'no_uncertainty': 'Pure DeLaN (no residual)',
    'no_adaptive_lam': 'Ours, fixed λ',
    'no_multiscale': 'Ours, plain loss',
    'dual_lam': 'Ours, dual-ascent λ',
    'dual_lam_wide': 'Ours, dual-ascent λ (sin/cos, no bottleneck)',
    'mlp': 'MLP',
    'hnn': 'HNN',
    'lnn': 'LNN',
    'symoden': 'SymODEN',
    'swevers': 'Swevers rigid-body ID (classical LS)',
    'se3': 'Ours, SE(3) structure (M=ΣJᵀGJ)',
    'se3_no_uncertainty': 'SE(3) structure only (no residual)',
}
METRIC = 'r2_global'          # 主指标：全局 R²（最保守，不被退化关节支配）


def load_all(in_dir=IN_DIR):
    runs = []
    for f in sorted(Path(in_dir).glob('*.json')):
        with open(f, encoding='utf-8') as fh:
            runs.append(json.load(fh))
    return runs


def group(runs, key=lambda r: (r['arm'], r['variant'])):
    g = defaultdict(list)
    for r in runs:
        g[key(r)].append(r)
    for k in g:
        g[k].sort(key=lambda r: r['seed'])
    return g


def welch(a, b):
    """Welch t 检验（不假设等方差），返回 t, p, df, Cohen's d。"""
    from scipy import stats
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return dict(t=float('nan'), p=float('nan'), df=float('nan'), cohen_d=float('nan'),
                    note='样本量不足')
    t, p = stats.ttest_ind(a, b, equal_var=False)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    df = (va / len(a) + vb / len(b)) ** 2 / (
        (va / len(a)) ** 2 / (len(a) - 1) + (vb / len(b)) ** 2 / (len(b) - 1))
    pooled = np.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2))
    return dict(t=float(t), p=float(p), df=float(df),
                cohen_d=float((a.mean() - b.mean()) / pooled) if pooled > 0 else float('nan'))


def holm(pvals):
    """Holm–Bonferroni 校正，返回校正后 p 值（同顺序）。"""
    idx = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    prev = 0.0
    for rank, i in enumerate(idx):
        val = (m - rank) * pvals[i]
        prev = max(prev, min(val, 1.0))
        adj[i] = prev
    return adj.tolist()


def fmt(mean, std, n, scale=1.0, dec=4):
    return f'{mean*scale:.{dec}f} ± {std*scale:.{dec}f} ({n})'


def main():
    runs = load_all()
    if not runs:
        print('没有找到 v2 结果，先跑 paper/run_grid.py'); return
    g = group(runs)
    arms = sorted({r['arm'] for r in runs})
    variants = [v for v in DISPLAY if any((a, v) in g for a in arms)]

    # ---------------- Table 1 主对比 ----------------
    t1 = {}
    for a in arms:
        for v in variants:
            rs = g.get((a, v), [])
            if not rs:
                continue
            vals = {k: np.array([r['test'][k] for r in rs])
                    for k in ['r2_global', 'r2_all_joints', 'r2_active_joints',
                              'rmse_overall', 'nrmse_mean']}
            t1[f'{a}/{v}'] = {
                'arm': a, 'variant': v, 'display': DISPLAY[v], 'n_seeds': len(rs),
                'seeds': [r['seed'] for r in rs],
                'n_params': rs[0]['n_params'],
                'train_time_min': float(np.mean([r['train_time_s'] for r in rs]) / 60),
                'best_epoch': [r['best_epoch'] for r in rs],
                **{f'{k}_mean': float(x.mean()) for k, x in vals.items()},
                **{f'{k}_std': float(x.std(ddof=1)) if len(rs) > 1 else 0.0
                   for k, x in vals.items()},
                **{f'{k}_values': x.tolist() for k, x in vals.items()},
                'val_r2_global_mean': float(np.mean([r['val']['r2_global'] for r in rs])),
            }

    # ---------------- 显著性检验 ----------------
    COMPARISONS = [
        ('full', 'no_physics',     '物理结构的价值：完整模型 vs 去掉物理分支'),
        ('full', 'mlp',            '相对纯数据驱动 MLP'),
        ('full', 'enc_sincos',     '编码：旋转矩阵 vs sin/cos'),
        ('full', 'enc_none',       '编码：旋转矩阵 vs 关节角直接输入'),
        ('enc_sincos', 'enc_none', '编码：sin/cos vs 关节角直接输入'),
        ('full', 'enc_rotmat_wide', '编码瓶颈：rotmat 压到 7 维 vs 无瓶颈'),
        ('enc_sincos', 'enc_sincos_wide', '编码瓶颈：sin/cos 压到 7 维 vs 无瓶颈'),
        ('enc_rotmat_wide', 'enc_none', '无瓶颈 rotmat vs 关节角直接输入'),
        ('full', 'no_uncertainty', '残差分支的价值'),
        ('full', 'no_adaptive_lam', '自适应 λ 的价值'),
        ('full', 'no_multiscale',  '多尺度损失的价值'),
        ('no_physics', 'mlp',      '同为黑箱：大 FFNN vs 标准 MLP'),
        ('full', 'dual_lam',       'λ 机制：自适应(塌缩) vs 对偶上升'),
        ('dual_lam', 'no_adaptive_lam', 'λ 机制：对偶上升 vs 固定 λ'),
        ('dual_lam_wide', 'enc_sincos_wide', 'λ 机制：对偶上升 vs 固定 λ（最优编码）'),
        ('full', 'swevers',        '经典刚体辨识 vs Ours（干净仿真数据的辨识天花板）'),
        ('swevers', 'mlp',         '经典刚体辨识 vs 纯数据驱动 MLP'),
        ('se3', 'dual_lam_wide',   'SE(3) 结构 vs 任意神经网络 H（单因素，核心）'),
        ('se3', 'full',            'SE(3) 结构 vs 原完整模型'),
        ('se3_no_uncertainty', 'no_uncertainty', 'SE(3) 单独 vs DeLaN 单独（归纳偏置）'),
        ('se3', 'swevers',         'SE(3) 学习惯量 vs 经典刚体辨识天花板'),
    ]
    stats_out, praw, keys = {}, [], []
    for a in arms:
        for x, y, desc in COMPARISONS:
            if (a, x) not in g or (a, y) not in g:
                continue
            va = [r['test'][METRIC] for r in g[(a, x)]]
            vb = [r['test'][METRIC] for r in g[(a, y)]]
            w = welch(va, vb)
            k = f'{a}/{x}_vs_{y}'
            stats_out[k] = {'arm': a, 'a': x, 'b': y, 'desc': desc,
                            'metric': METRIC,
                            'mean_a': float(np.mean(va)), 'mean_b': float(np.mean(vb)),
                            'delta': float(np.mean(va) - np.mean(vb)),
                            'n_a': len(va), 'n_b': len(vb), **w}
            if np.isfinite(w['p']):
                praw.append(w['p']); keys.append(k)
    for k, p in zip(keys, holm(praw)):
        stats_out[k]['p_holm'] = float(p)

    # ---------------- 逐关节 ----------------
    per_joint = {}
    for a in arms:
        names = ARMS[a].joint_names
        for v in variants:
            rs = g.get((a, v), [])
            if not rs:
                continue
            arr = np.array([[r['test']['r2_per_joint'][n] for n in names] for r in rs])
            per_joint[f'{a}/{v}'] = {
                'joints': names,
                'mean': arr.mean(0).tolist(),
                'std': (arr.std(0, ddof=1) if len(rs) > 1 else np.zeros(len(names))).tolist(),
                'tau_std': [rs[0]['test']['tau_std_per_joint'][n] for n in names],
                'rmse_mean': np.array([[r['test']['rmse_per_joint'][n] for n in names]
                                       for r in rs]).mean(0).tolist(),
            }

    save_json(OUT_DIR / 'table1_main.json', t1)
    save_json(OUT_DIR / 'stats.json', stats_out)
    save_json(OUT_DIR / 'table3_per_joint.json', per_joint)

    # ---------------- Markdown ----------------
    lines = ['# 主结果（测试集，验证集选模，均值 ± 标准差(seed 数)）', '']
    for a in arms:
        lines += [f'## {a.capitalize()}', '',
                  '| 方法 | R² (global) | R² (全关节均值) | R² (活跃关节) | RMSE (N·m) | 参数量 | 训练(min) |',
                  '|---|---|---|---|---|---|---|']
        for v in variants:
            k = f'{a}/{v}'
            if k not in t1:
                continue
            e = t1[k]
            lines.append(
                f"| {e['display']} | {fmt(e['r2_global_mean'], e['r2_global_std'], e['n_seeds'])} "
                f"| {fmt(e['r2_all_joints_mean'], e['r2_all_joints_std'], e['n_seeds'])} "
                f"| {fmt(e['r2_active_joints_mean'], e['r2_active_joints_std'], e['n_seeds'])} "
                f"| {e['rmse_overall_mean']:.4f} | {e['n_params']/1e6:.2f}M "
                f"| {e['train_time_min']:.1f} |")
        lines.append('')
    (OUT_DIR / 'table1_main.md').write_text('\n'.join(lines), encoding='utf-8')

    lines = ['# 消融与显著性检验（指标：测试集 global R²，Welch t 检验，Holm 校正）', '',
             '| 臂 | 对比 | ΔR² | t | p (raw) | p (Holm) | Cohen d | 结论 |', '|---|---|---|---|---|---|---|---|']
    for k, s in stats_out.items():
        sig = '显著' if s.get('p_holm', 1) < 0.05 else '不显著'
        lines.append(f"| {s['arm']} | {s['desc']} | {s['delta']:+.4f} | {s['t']:.2f} "
                     f"| {s['p']:.2e} | {s.get('p_holm', float('nan')):.2e} "
                     f"| {s['cohen_d']:+.2f} | {sig} |")
    (OUT_DIR / 'table2_ablation.md').write_text('\n'.join(lines), encoding='utf-8')

    lines = ['# 逐关节 R²（测试集）', '']
    for a in arms:
        names = ARMS[a].joint_names
        lines += [f'## {a.capitalize()}', '',
                  '| 方法 | ' + ' | '.join(names) + ' |',
                  '|---' * (len(names) + 1) + '|']
        for v in variants:
            k = f'{a}/{v}'
            if k not in per_joint:
                continue
            pj = per_joint[k]
            lines.append(f'| {DISPLAY[v]} | ' +
                         ' | '.join(f'{m:.3f}' for m in pj['mean']) + ' |')
        k0 = next((f'{a}/{v}' for v in variants if f'{a}/{v}' in per_joint), None)
        if k0:
            lines.append('| *τ std (N·m)* | ' +
                         ' | '.join(f'{s:.3f}' for s in per_joint[k0]['tau_std']) + ' |')
        lines.append('')
    (OUT_DIR / 'table3_per_joint.md').write_text('\n'.join(lines), encoding='utf-8')

    # ---------------- 控制台输出 ----------------
    for a in arms:
        print(f'\n===== {a.upper()} (测试集, {METRIC}) =====')
        print(f"{'方法':<32s}{'R2_global':>18s}{'R2_all':>18s}{'R2_active':>18s}")
        for v in variants:
            k = f'{a}/{v}'
            if k not in t1:
                continue
            e = t1[k]
            print(f"{e['display']:<32s}"
                  f"{e['r2_global_mean']:>10.4f}±{e['r2_global_std']:<7.4f}"
                  f"{e['r2_all_joints_mean']:>10.4f}±{e['r2_all_joints_std']:<7.4f}"
                  f"{e['r2_active_joints_mean']:>10.4f}±{e['r2_active_joints_std']:<7.4f}")
    print('\n===== 显著性检验 =====')
    for k, s in stats_out.items():
        print(f"{k:<42s} Δ={s['delta']:+.4f}  p={s['p']:.2e}  p_holm={s.get('p_holm', float('nan')):.2e}")
    print(f'\n表格已写入 {OUT_DIR}')


if __name__ == '__main__':
    main()
