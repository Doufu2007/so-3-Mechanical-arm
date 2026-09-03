"""
λ 权衡分析 —— 精度(R²) ↔ 物理保真度(‖Δg‖/‖g‖) ↔ 可用性(闭环RMSE) 随 λ 的变化。

核心论点（P0 的「负结果转卖点」）：
  λ→0 时残差分支自由补偿、R² 最高，但 g(q) 学错、闭环重力补偿失效；
  加大 λ 逼物理分支学准 g(q)，代价是 R² 基本不变甚至微降。
这张图/表就是这条论点的定量证据。

数据来源（全部由流水线产出，脚本只读不写数字）：
  R²          experimental_results/v2_hparam/lam{λ}_franka_ex_seed{s}.json → test.r2_global
  ‖Δg‖/‖g‖    experimental_results/v2_physics/lam{λ}_franka_ex_seed0_physics.json → D_vs_mujoco.g_rel_err
  闭环 RMSE    experimental_results/v2_control/control_lam{λ}.json → summary.pd_grav_learn.rmse_mean
  参考基线      v2_control/control.json（自适应λ→0）与 mlp 静平衡 / 真值 g(q)

    python paper/lam_tradeoff.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paper import core
from paper.core import save_json

ROOT = core.ROOT
LAM = [0.0, 0.1, 0.5, 1.0, 5.0]
SEEDS = [0, 1, 2]
HPDIR = ROOT / 'experimental_results/v2_hparam'
PHDIR = ROOT / 'experimental_results/v2_physics'
CTLDIR = ROOT / 'experimental_results/v2_control'
TBL = ROOT / 'experimental_results/v2_tables/lam_tradeoff.json'
FIG_DIR = ROOT / 'experimental_results/v2_figures'


def _j(p):
    p = Path(p)
    if not p.exists():
        return None
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def lam_name(lam):
    # 与 run_grid.py 的 tag 生成一致：普通 float 格式（0.0→"lam0.0", 1.0→"lam1.0"）
    return f'lam{lam}'


def collect():
    rows = []
    for lam in LAM:
        tag = lam_name(lam)
        # R²（多个 seed 聚合）
        r2s = []
        for s in SEEDS:
            j = _j(HPDIR / f'{tag}_franka_ex_seed{s}.json')
            if j and 'test' in j and 'r2_global' in j['test']:
                r2s.append(j['test']['r2_global'])
        # g(q) 误差（seed0）
        g = _j(PHDIR / f'{tag}_franka_ex_seed0_physics.json')
        g_rel = (g['D_vs_mujoco']['g_rel_err']
                 if g and g.get('D_vs_mujoco', {}).get('available') else None)
        # 闭环（seed0）
        c = _j(CTLDIR / f'control_{tag}.json')
        rmse = (c['summary']['pd_grav_learn']['rmse_mean'] * 1000
                if c and 'pd_grav_learn' in c.get('summary', {}) else None)
        rows.append({
            'lam': lam,
            'r2_global_mean': float(np.mean(r2s)) if r2s else None,
            'r2_global_std': float(np.std(r2s, ddof=1)) if len(r2s) > 1 else None,
            'n_seeds': len(r2s),
            'g_rel_err': g_rel,
            'ctrl_rmse_mrad': rmse,
        })
    # 参考基线
    c0 = _j(CTLDIR / 'control.json')
    refs = {}
    if c0:
        refs['ctrl_rmse_mrad_adaptive_lam0'] = (
            c0['summary']['pd_grav_learn']['rmse_mean'] * 1000)
        refs['ctrl_rmse_mrad_mlp'] = c0['summary']['pd_grav_mlp']['rmse_mean'] * 1000
        refs['ctrl_rmse_mrad_true'] = c0['summary']['pd_grav_true']['rmse_mean'] * 1000
        refs['ctrl_rmse_mrad_none'] = c0['summary']['pd']['rmse_mean'] * 1000
    out = {'arm': 'franka_ex', 'lam_values': LAM, 'rows': rows, 'references': refs,
           'note': 'R² 为 3 seeds 测试集 r2_global 聚合；g(q) 误差与闭环 RMSE 用 seed0 '
                   '（物理核验与闭环只跑了 seed0）。闭环任务为重力补偿定点调节。',
           'provenance': core.provenance()}
    save_json(TBL, out)
    return out


def figure(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    SURFACE, INK, INK_2, GRID, MUTED = '#fcfcfb', '#0b0b0b', '#52514e', '#e6e5e1', '#8a8984'
    BLUE, ORANGE, GREEN = '#2a78d6', '#eb6834', '#1baf7a'
    plt.rcParams.update({'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
                         'savefig.facecolor': SURFACE, 'font.size': 10,
                         'axes.edgecolor': GRID, 'axes.labelcolor': INK, 'text.color': INK,
                         'xtick.color': INK_2, 'ytick.color': INK_2, 'axes.grid': True,
                         'grid.color': GRID, 'grid.linewidth': 0.8, 'axes.axisbelow': True,
                         'lines.linewidth': 2.0, 'lines.markersize': 6, 'figure.dpi': 160})

    rows = [r for r in out['rows'] if r['r2_global_mean'] is not None]
    xs = [r['lam'] for r in rows]
    r2 = [r['r2_global_mean'] for r in rows]
    r2e = [r['r2_global_std'] or 0.0 for r in rows]
    rmse = [r['ctrl_rmse_mrad'] for r in rows if r['ctrl_rmse_mrad'] is not None]
    rmse_x = [r['lam'] for r in rows if r['ctrl_rmse_mrad'] is not None]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))

    ax = axes[0]
    ax.errorbar(xs, r2, yerr=r2e, color=BLUE, marker='o', capsize=3, lw=2.0)
    ax.set_xscale('log')
    ax.set_xticks(xs); ax.get_xaxis().set_major_formatter(
        matplotlib.ticker.ScalarFormatter())
    for x, y, s in zip(xs, r2, r2e):
        ax.text(x, y + s + 0.004, f'{y:.3f}', ha='center', va='bottom',
                fontsize=8, color=INK_2)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    ax.grid(axis='x', visible=False)
    ax.set_ylim(min(r2) - 0.03, max(r2) + 0.02)
    ax.set_xlabel('physics weight λ (log)'); ax.set_ylabel('Test $R^2$ (global)')
    ax.set_title('Fitting accuracy is flat as λ grows', loc='left',
                 color=INK, fontweight='bold')

    ax = axes[1]
    ax.plot(rmse_x, rmse, color=ORANGE, marker='o', lw=2.0, label='learned g(q)')
    refs = out.get('references', {})
    if refs.get('ctrl_rmse_mrad_mlp') is not None:
        ax.axhline(refs['ctrl_rmse_mrad_mlp'], color=MUTED, lw=1.4, ls='--',
                   label='MLP static τ')
    if refs.get('ctrl_rmse_mrad_true') is not None:
        ax.axhline(refs['ctrl_rmse_mrad_true'], color=GREEN, lw=1.4, ls='--',
                   label='true g(q)')
    if refs.get('ctrl_rmse_mrad_none') is not None:
        ax.axhline(refs['ctrl_rmse_mrad_none'], color='#e34948', lw=1.4, ls=':',
                   label='PD only (no comp.)')
    ax.set_xscale('log')
    ax.set_xticks(rmse_x); ax.get_xaxis().set_major_formatter(
        matplotlib.ticker.ScalarFormatter())
    for x, y in zip(rmse_x, rmse):
        ax.text(x, y + 1.5, f'{y:.1f}', ha='center', va='bottom',
                fontsize=8, color=INK_2)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    ax.grid(axis='x', visible=False)
    ax.set_xlabel('physics weight λ (log)')
    ax.set_ylabel('steady-state error RMSE (mrad)')
    ax.set_title('…but gravity compensation becomes usable', loc='left',
                 color=INK, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')

    fig.suptitle('The accuracy–usability trade-off of the physics weight λ '
                 '(7-DoF arm, rich excitation)', x=0.02, ha='left', fontsize=11,
                 color=INK_2)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ['png', 'pdf']:
        fig.savefig(FIG_DIR / f'fig9_lambda_tradeoff.{ext}', bbox_inches='tight')
    plt.close(fig)
    print(f'  [fig] {FIG_DIR / "fig9_lambda_tradeoff.png"}')


def markdown(out):
    rows = out['rows']
    refs = out.get('references', {})
    L = [f"| λ | R²(global) | ‖Δg‖/‖g‖ | 闭环稳态 RMSE (mrad) |",
         '|---|---|---|---|']
    for r in rows:
        r2 = f"{r['r2_global_mean']:.4f} ± {r['r2_global_std']:.4f}" \
            if r['r2_global_mean'] is not None else '—'
        g = f"{r['g_rel_err']:.3f}" if r['g_rel_err'] is not None else '—'
        c = f"{r['ctrl_rmse_mrad']:.2f}" if r['ctrl_rmse_mrad'] is not None else '—'
        L.append(f"| {r['lam']:g} | {r2} | {g} | {c} |")
    L.append('')
    if refs:
        L.append('参考：自适应 λ→0 的闭环 RMSE = '
                 f"{refs.get('ctrl_rmse_mrad_adaptive_lam0', float('nan')):.2f} mrad，"
                 f"MLP 静平衡 = {refs.get('ctrl_rmse_mrad_mlp', float('nan')):.2f} mrad，"
                 f"真值 g(q) = {refs.get('ctrl_rmse_mrad_true', float('nan')):.2e} mrad，"
                 f"无补偿 = {refs.get('ctrl_rmse_mrad_none', float('nan')):.2f} mrad。")
    return '\n'.join(L)


def main():
    out = collect()
    if not any(r['r2_global_mean'] is not None for r in out['rows']):
        print('[skip] λ 权衡数据还没生成（v2_hparam 为空）')
        return
    md = markdown(out)
    (ROOT / 'experimental_results/v2_tables/lam_tradeoff.md').write_text(md, encoding='utf-8')
    print(md)
    try:
        figure(out)
    except Exception as ex:
        import traceback
        print(f'[fail] 图生成失败: {ex}')
        traceback.print_exc()


if __name__ == '__main__':
    main()
