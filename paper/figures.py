"""
论文图件生成 —— 全部从 experimental_results/ 的 json 读数，不写死任何数字。

    python paper/figures.py            # 生成能生成的全部图
    python paper/figures.py --only main per_joint

配色遵循 dataviz 规范：固定顺序的分类色（色盲安全，已过 validate_palette 六项检查）、
条形图不用颜色重复 x 轴信息（单色 + 高亮本文方法）、顺序色阶单色相、
每图 ≥2 序列必带图例、数值直接标注、无双 y 轴。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paper import core
from paper.core import ARMS

ROOT = core.ROOT
FIG_DIR = ROOT / 'experimental_results/v2_figures'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ---- 设计参数（dataviz 参考实例，已验证）----
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_2 = '#52514e'
MUTED = '#8a8984'
GRID = '#e6e5e1'
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300',
          '#4a3aa7', '#e34948']
BLUE_RAMP = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7',
             '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b']
HIGHLIGHT = '#2a78d6'
NEUTRAL = '#b9b8b2'

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
    'savefig.facecolor': SURFACE,
    'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
    'axes.edgecolor': GRID, 'axes.labelcolor': INK, 'text.color': INK,
    'xtick.color': INK_2, 'ytick.color': INK_2,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,
    'axes.axisbelow': True, 'legend.frameon': False,
    'lines.linewidth': 2.0, 'lines.markersize': 6,
    'figure.dpi': 160,
})

DISPLAY = {
    'full': 'Ours (rotmat)', 'enc_sincos': 'Ours (sin/cos)', 'enc_none': 'Ours (raw q)',
    'no_physics': 'No physics (FFNN)', 'no_uncertainty': 'Pure DeLaN',
    'no_adaptive_lam': 'Fixed λ', 'no_multiscale': 'Plain loss',
    'mlp': 'MLP', 'hnn': 'HNN', 'lnn': 'LNN', 'symoden': 'SymODEN',
}
COLOR = {k: SERIES[i % len(SERIES)] for i, k in enumerate(
    ['full', 'enc_sincos', 'enc_none', 'no_uncertainty', 'no_physics', 'mlp', 'hnn', 'lnn'])}
COLOR['no_adaptive_lam'] = SERIES[3]
COLOR['no_multiscale'] = SERIES[4]
COLOR['symoden'] = SERIES[7]

ARM_LABEL = {'baxter': 'Baxter (real robot)',
             'franka_ex': '7-DoF sim. arm (rich excitation)',
             'franka': '7-DoF sim. arm (narrow coverage)'}


def _clean(ax, ylabel=None, xlabel=None, title=None):
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    ax.grid(axis='x', visible=False)
    if ylabel: ax.set_ylabel(ylabel)
    if xlabel: ax.set_xlabel(xlabel)
    if title: ax.set_title(title, loc='left', color=INK, fontweight='bold')
    return ax


def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ['png', 'pdf']:
        fig.savefig(FIG_DIR / f'{name}.{ext}', bbox_inches='tight')
    plt.close(fig)
    print(f'  [fig] {FIG_DIR / (name + ".png")}')


def load_json(p):
    p = ROOT / p
    if not p.exists():
        return None
    with open(p, encoding='utf-8') as f:
        return json.load(f)


# ------------------------------------------------------------------ 图 1 主对比
def fig_main(t1):
    arms = sorted({v['arm'] for v in t1.values()},
                  key=lambda a: ['baxter', 'franka_ex', 'franka'].index(a))
    order = ['full', 'enc_sincos', 'enc_none', 'no_uncertainty', 'no_physics',
             'mlp', 'hnn', 'lnn', 'symoden']
    fig, axes = plt.subplots(1, len(arms), figsize=(5.2 * len(arms), 4.2))
    axes = np.atleast_1d(axes)
    for ax, arm in zip(axes, arms):
        ks = [v for v in order if f'{arm}/{v}' in t1]
        m = [t1[f'{arm}/{v}']['r2_global_mean'] for v in ks]
        s = [t1[f'{arm}/{v}']['r2_global_std'] for v in ks]
        # 条形图不用颜色重复 x 轴信息：单色 + 高亮本文完整模型
        colors = [HIGHLIGHT if v == 'full' else NEUTRAL for v in ks]
        x = np.arange(len(ks))
        ax.bar(x, m, yerr=s, color=colors, width=0.68, capsize=3,
               error_kw=dict(ecolor=INK_2, lw=1.2))
        for xi, mi, si in zip(x, m, s):
            ax.text(xi, max(mi, 0) + si + 0.02, f'{mi:.3f}', ha='center',
                    va='bottom', fontsize=8, color=INK_2)
        ax.set_xticks(x)
        ax.set_xticklabels([DISPLAY[v] for v in ks], rotation=38, ha='right', fontsize=8)
        ax.set_ylim(min(0, min(m) - 0.1), 1.0)
        ax.axhline(0, color=MUTED, lw=1)
        _clean(ax, ylabel='Test $R^2$ (global)' if ax is axes[0] else None,
               title=ARM_LABEL.get(arm, arm))
    fig.suptitle('Inverse-dynamics accuracy (mean ± s.d. over seeds, model selected on validation set)',
                 x=0.02, ha='left', fontsize=11, color=INK_2)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, 'fig1_main_comparison')


# ------------------------------------------------------------------ 图 2 逐关节
def fig_per_joint(pj, t1):
    arms = sorted({k.split('/')[0] for k in pj},
                  key=lambda a: ['baxter', 'franka_ex', 'franka'].index(a))
    show = ['full', 'no_physics', 'mlp']
    fig, axes = plt.subplots(2, len(arms), figsize=(5.4 * len(arms), 6.0),
                             gridspec_kw={'height_ratios': [2.2, 1]}, sharex='col')
    axes = axes.reshape(2, -1)
    for j, arm in enumerate(arms):
        names = ARMS[arm].joint_names
        ax = axes[0, j]
        ks = [v for v in show if f'{arm}/{v}' in pj]
        w = 0.8 / max(len(ks), 1)
        for i, v in enumerate(ks):
            d = pj[f'{arm}/{v}']
            x = np.arange(len(names)) + (i - (len(ks) - 1) / 2) * w
            ax.bar(x, d['mean'], yerr=d['std'], width=w * 0.9, label=DISPLAY[v],
                   color=COLOR[v], capsize=2, error_kw=dict(ecolor=INK_2, lw=0.9))
        ax.set_ylim(-0.6, 1.0)
        ax.axhline(0, color=MUTED, lw=1)
        ax.legend(fontsize=8, ncol=len(ks))
        _clean(ax, ylabel='Test $R^2$' if j == 0 else None, title=ARM_LABEL.get(arm, arm))

        ax2 = axes[1, j]
        std = pj[f'{arm}/{ks[0]}']['tau_std']
        ax2.bar(np.arange(len(names)), std, color=NEUTRAL, width=0.62)
        ax2.set_yscale('log')
        ax2.set_xticks(np.arange(len(names)))
        ax2.set_xticklabels(names)
        thr = 0.05 * max(std)
        ax2.axhline(thr, color=SERIES[1], lw=1.4, ls='--')
        ax2.text(len(names) - 0.4, thr, ' 5% of max', color=SERIES[1], fontsize=8,
                 va='bottom', ha='right')
        _clean(ax2, ylabel='torque s.d. (N·m)' if j == 0 else None, xlabel='joint')
    fig.suptitle('Per-joint accuracy, and why some joints are unpredictable '
                 '(their torque barely varies)', x=0.02, ha='left', fontsize=11, color=INK_2)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, 'fig2_per_joint')


# ------------------------------------------------------------------ 图 3 收敛曲线
def fig_curves(runs):
    show = ['full', 'no_physics', 'mlp']
    arms = sorted({r['arm'] for r in runs},
                  key=lambda a: ['baxter', 'franka_ex', 'franka'].index(a))
    fig, axes = plt.subplots(1, len(arms), figsize=(5.2 * len(arms), 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, arm in zip(axes, arms):
        for v in show:
            r = next((x for x in runs if x['arm'] == arm and x['variant'] == v
                      and x['seed'] == 0 and x.get('frac', 1.0) == 1.0), None)
            if not r:
                continue
            c = r['curve']
            ax.plot(c['val_epoch'], c['train_r2'], color=COLOR[v], lw=1.4, ls=':', alpha=0.8)
            ax.plot(c['val_epoch'], c['val_r2'], color=COLOR[v], lw=2.0,
                    label=DISPLAY[v], marker='o', markevery=max(1, len(c['val_epoch']) // 8),
                    markersize=4)
        ax.set_ylim(0, 1.02)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.legend(fontsize=8, loc='lower right')
        _clean(ax, ylabel='$R^2$' if ax is axes[0] else None, xlabel='epoch',
               title=ARM_LABEL.get(arm, arm))
    axes[0].text(0.03, 0.06, 'dotted = train,  solid = validation',
                 transform=axes[0].transAxes, fontsize=8, color=INK_2)
    fig.suptitle('Convergence and the train–validation gap (seed 0)',
                 x=0.02, ha='left', fontsize=11, color=INK_2)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, 'fig3_convergence')


# ------------------------------------------------------------------ 图 4 数据效率
def fig_data_eff(runs_de, t1):
    show = ['full', 'no_physics', 'mlp']
    arms = sorted({r['arm'] for r in runs_de},
                  key=lambda a: ['baxter', 'franka_ex', 'franka'].index(a))
    fig, axes = plt.subplots(1, len(arms), figsize=(5.2 * len(arms), 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, arm in zip(axes, arms):
        for v in show:
            xs, ys, es = [], [], []
            for fr in [0.05, 0.1, 0.25, 0.5, 1.0]:
                if fr == 1.0:
                    k = f'{arm}/{v}'
                    if k not in t1:
                        continue
                    xs.append(100.0); ys.append(t1[k]['r2_global_mean'])
                    es.append(t1[k]['r2_global_std'])
                else:
                    rs = [r for r in runs_de if r['arm'] == arm and r['variant'] == v
                          and abs(r['frac'] - fr) < 1e-9]
                    if not rs:
                        continue
                    vals = [r['test']['r2_global'] for r in rs]
                    xs.append(fr * 100); ys.append(np.mean(vals))
                    es.append(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
            if xs:
                ax.errorbar(xs, ys, yerr=es, color=COLOR[v], label=DISPLAY[v],
                            marker='o', markersize=6, capsize=3, lw=2.0)
        ax.set_xscale('log')
        ax.set_xticks([5, 10, 25, 50, 100])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.legend(fontsize=8, loc='lower right')
        _clean(ax, ylabel='Test $R^2$ (global)' if ax is axes[0] else None,
               xlabel='training data used (%)', title=ARM_LABEL.get(arm, arm))
    fig.suptitle('Data efficiency', x=0.02, ha='left', fontsize=11, color=INK_2)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, 'fig4_data_efficiency')


# ------------------------------------------------------------------ 图 5 噪声
def fig_noise(noise_by_arm):
    arms = list(noise_by_arm)
    fig, axes = plt.subplots(1, len(arms), figsize=(5.2 * len(arms), 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, arm in zip(axes, arms):
        d = noise_by_arm[arm]
        for v, r in d['results'].items():
            ax.errorbar([s * 100 for s in r['noise_levels']], r['r2_mean'],
                        yerr=r['r2_std'], color=COLOR.get(v, MUTED),
                        label=DISPLAY.get(v, v), marker='o', markersize=6,
                        capsize=3, lw=2.0)
        ax.legend(fontsize=8, loc='lower left')
        _clean(ax, ylabel='Test $R^2$ (global)' if ax is axes[0] else None,
               xlabel='input noise σ (% of signal s.d.)', title=ARM_LABEL.get(arm, arm))
    fig.suptitle('Robustness to measurement noise on q, q̇, q̈',
                 x=0.02, ha='left', fontsize=11, color=INK_2)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, 'fig5_noise_robustness')


# ------------------------------------------------------------------ 图 6 闭环控制
CTRL_LABEL = {'pd': 'PD only', 'pd_grav_mlp': 'PD + MLP static τ',
              'pd_grav_learn': 'PD + learned g(q)', 'pd_grav_true': 'PD + true g(q)'}


def fig_control(ctrl):
    order = ['pd', 'pd_grav_mlp', 'pd_grav_learn', 'pd_grav_true']
    order = [c for c in order if c in ctrl.get('summary', {})]
    if not order:
        return
    m = [ctrl['summary'][c]['rmse_mean'] * 1000 for c in order]
    s = [ctrl['summary'][c]['rmse_std'] * 1000 for c in order]
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    x = np.arange(len(order))
    ax.bar(x, m, yerr=s, width=0.6, color=[NEUTRAL, NEUTRAL, HIGHLIGHT, SERIES[1]],
           capsize=3, error_kw=dict(ecolor=INK_2, lw=1.2))
    for xi, mi, si in zip(x, m, s):
        ax.text(xi, mi + si + 1, f'{mi:.1f}', ha='center', va='bottom', fontsize=9, color=INK_2)
    ax.set_xticks(x)
    ax.set_xticklabels([CTRL_LABEL[c] for c in order], rotation=15, ha='right', fontsize=9)
    ax.set_ylim(0, max(m) * 1.25)
    _clean(ax, ylabel='steady-state error RMSE (mrad)',
           title=f'Gravity-compensated setpoint regulation (Kp={ctrl["kp"]:g}, '
                 f'{ctrl["n_postures"]} postures)')
    fig.tight_layout()
    save(fig, 'fig6_closed_loop_control')


# ------------------------------------------------------------------ 图 7 物理性质
def fig_physics(reps):
    reps = [r for r in reps if 'A_inertia' in r]
    if not reps:
        return
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))

    ax = axes[0]
    for i, r in enumerate(reps[:4]):
        cond = np.array(r['A_inertia']['cond_values'])
        xs = np.sort(cond)
        ax.plot(xs, np.linspace(0, 1, len(xs)), color=SERIES[i], lw=2.0,
                label=f"{DISPLAY.get(r['variant'], r['variant'])} / {r['arm']}")
    ax.set_xscale('log')
    ax.legend(fontsize=7)
    _clean(ax, ylabel='empirical CDF', xlabel='cond $H(q)$', title='Inertia-matrix conditioning')

    ax = axes[1]
    r = next((x for x in reps if x.get('D_vs_mujoco', {}).get('available')), None)
    if r:
        d = r['D_vs_mujoco']
        names = ARMS[r['arm']].joint_names
        true_rms = np.array(d['g_true_rms_per_joint'])
        pred_rms = np.array(d.get('g_pred_rms_per_joint', [np.nan] * 7))
        x = np.arange(len(names))
        ax.bar(x - 0.2, true_rms, width=0.38, color=NEUTRAL, label='true g(q) RMS')
        ax.bar(x + 0.2, pred_rms, width=0.38, color=HIGHLIGHT, label='learned g(q) RMS')
        ax.set_xticks(x); ax.set_xticklabels(names)
        ax.legend(fontsize=8)
        _clean(ax, ylabel='g(q) RMS (N·m)',
               title=f"Learned vs true gravity\n(rel. err. {d['g_rel_err']:.3f})")
    else:
        ax.axis('off')

    ax = axes[2]
    for i, r in enumerate(reps[:4]):
        e = r.get('E_lambda', {})
        if not e.get('available'):
            continue
        h = np.array(e['hist'], float)
        centers = np.linspace(0, 0.1, len(h) + 1)[:-1] + 0.1 / len(h) / 2
        ax.plot(centers, h / h.sum(), color=SERIES[i], lw=2.0,
                label=f"{DISPLAY.get(r['variant'], r['variant'])} / {r['arm']}")
    ax.legend(fontsize=7)
    _clean(ax, xlabel='adaptive λ', ylabel='fraction of samples',
           title='Adaptive physics weight (ceiling 0.1)')
    fig.tight_layout()
    save(fig, 'fig7_physics_checks')


# ------------------------------------------------------------------ 图 8 覆盖度
def fig_coverage():
    from scipy.spatial import cKDTree
    from paper.core import load_arm
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for i, arm in enumerate(['franka', 'franka_ex', 'baxter']):
        try:
            d = load_arm(arm)
        except Exception:
            continue
        qtr, qte = d['train'][0], d['test'][0]
        lo, hi = qtr.min(0), qtr.max(0)
        rng = np.maximum(hi - lo, 1e-6)
        t = cKDTree(((qtr - lo) / rng)[::5])
        dte, _ = t.query(((qte - lo) / rng)[::5])
        xs = np.sort(dte)
        ax.plot(xs, np.linspace(0, 1, len(xs)), color=SERIES[i], lw=2.0,
                label=ARM_LABEL.get(arm, arm))
    ax.legend(fontsize=8, loc='lower right')
    _clean(ax, xlabel='normalised distance from a test point to the nearest training point',
           ylabel='empirical CDF', title='Configuration-space coverage of the test set')
    fig.tight_layout()
    save(fig, 'fig8_coverage')


# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*', default=None)
    a = ap.parse_args()
    want = set(a.only) if a.only else None

    def run(name, fn, *args):
        if want and name not in want:
            return
        if any(x is None for x in args):
            print(f'  [skip] {name}: 数据还没生成')
            return
        try:
            fn(*args)
        except Exception as e:
            import traceback
            print(f'  [fail] {name}: {e}')
            traceback.print_exc()

    t1 = load_json('experimental_results/v2_tables/table1_main.json')
    pj = load_json('experimental_results/v2_tables/table3_per_joint.json')
    runs = [json.load(open(f, encoding='utf-8'))
            for f in sorted((ROOT / 'experimental_results/v2').glob('*.json'))]
    de_dir = ROOT / 'experimental_results/v2_data_eff'
    runs_de = [json.load(open(f, encoding='utf-8')) for f in sorted(de_dir.glob('*.json'))] \
        if de_dir.exists() else []
    noise = {}
    nd = ROOT / 'experimental_results/v2_noise'
    if nd.exists():
        for f in sorted(nd.glob('noise_*.json')):
            d = json.load(open(f, encoding='utf-8'))
            noise[d['arm']] = d
    ctrl = load_json('experimental_results/v2_control/control.json')
    pd_dir = ROOT / 'experimental_results/v2_physics'
    reps = [json.load(open(f, encoding='utf-8')) for f in sorted(pd_dir.glob('*.json'))] \
        if pd_dir.exists() else []

    run('main', fig_main, t1)
    run('per_joint', fig_per_joint, pj, t1)
    run('curves', fig_curves, runs or None)
    run('data_eff', fig_data_eff, runs_de or None, t1)
    run('noise', fig_noise, noise or None)
    run('control', fig_control, ctrl)
    run('physics', fig_physics, reps or None)
    run('coverage', fig_coverage)
    print(f'\n图件目录: {FIG_DIR}')


if __name__ == '__main__':
    main()
