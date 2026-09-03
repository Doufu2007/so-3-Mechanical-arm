"""审计三支柱的三张图 —— Fig.10/11/12。

数据全部来自 `experimental_results/identifiability/` 下的 JSON，由

    python paper/identifiability.py   --xml tool/panda_real.xml --json ...
    python paper/identifiability.py   --xml tool/panda_real.xml --static --json ...
    python paper/baseline_ls_exact.py --xml tool/panda_real.xml --json ...
    python paper/gravity_subspace.py  --xml tool/panda_real.xml --json ...

产出。本脚本只画图，不做任何计算，也不重新拟合 —— 图里的每个数都能在 JSON 里查到。

    python paper/fig_identifiability.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / 'experimental_results' / 'identifiability'
OUT = ROOT / 'experimental_results' / 'v2_figures'

plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
    'legend.fontsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'figure.dpi': 150, 'savefig.bbox': 'tight', 'axes.grid': True,
    'grid.alpha': 0.3, 'grid.linewidth': 0.5,
})

C_ID, C_NULL, C_REF = '#1f77b4', '#d62728', '#7f7f7f'


def load(name):
    return json.loads((SRC / name).read_text(encoding='utf-8'))


def save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'{stem}.{ext}')
    plt.close(fig)
    print(f'  写入 {OUT.name}/{stem}.png / .pdf')


# ---------------------------------------------------------------------------
# Fig 10 —— 奇异值谱：秩断崖，两臂同秩
# ---------------------------------------------------------------------------
def fig_spectrum():
    real = load('identifiability_panda_real.json')
    toy = load('identifiability_panda.json')
    bax = load('identifiability_baxter_left.json')
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.9))

    # 三条谱线：两条 zyzyzyz（Panda 真臂 + 8 连杆玩具）同秩 43；Baxter 左臂 zyxyxyx
    # 是第一个真正不同的运动学结构，给出 43 强 + 18 弱 + 9 精确零的三层谱。
    for d, lab, mk, c in ((real, f'Franka Panda · zyzyzyz (K={real["K"]})', 'o', C_ID),
                          (toy, f'8-link toy · zyzyzyz (K={toy["K"]})', 's', '#ff7f0e'),
                          (bax, f'Baxter left · zyxyxyx (K={bax["K"]})', '^', '#9467bd')):
        s = np.asarray(d['singular_values_rel'])
        s = np.maximum(s, 1e-18)
        ax.semilogy(np.arange(1, len(s) + 1), s, mk + '-', color=c, ms=2.5,
                    lw=0.9, label=lab)
        ax.axvline(d['rank'] + 0.5, color=c, ls='--', lw=0.9, alpha=0.7)

    ax.axhline(1e-9, color=C_REF, ls=':', lw=0.9)
    ax.text(1.5, 1.6e-9, 'rank tolerance $10^{-9}$', color=C_REF, fontsize=7)
    ax.annotate(f'rank = {real["rank"]}', xy=(real['rank'] + 0.5, 1e-8),
                xytext=(48, 4e-5), fontsize=8, color=C_ID,
                arrowprops=dict(arrowstyle='->', color=C_ID, lw=0.8))
    # Baxter 的弱近零空间带：非机器精度零、也非强可辨识，位于 1e-8..1e-10
    ax.annotate('weak near-nullspace\n(18 dirs, $10^{-8}$–$10^{-10}$)',
                xy=(52, 2e-9), xytext=(53, 2e-4),
                fontsize=7, color='#9467bd',
                arrowprops=dict(arrowstyle='->', color='#9467bd', lw=0.8))
    ax.set_xlabel('singular-value index')
    ax.set_ylabel(r'$\sigma_k / \sigma_1$')
    ax.set_title('(a) Inertial-regressor spectrum')
    ax.set_ylim(1e-18, 3)
    ax.legend(loc='lower left')

    # (b) 完整 vs 静态
    gs = load('gravity_subspace_real.json')
    for s, lab, c in ((gs['sv_full'], f'full dynamics — rank {gs["dim_dynamic"]}', C_ID),
                      (gs['sv_static'], f'static / gravity only — rank {gs["dim_gravity"]}', '#2ca02c')):
        s = np.asarray(s)
        s = np.maximum(s / s[0], 1e-18)
        ax2.semilogy(np.arange(1, len(s) + 1), s, '-', color=c, lw=1.2, label=lab)
    ax2.axvline(gs['dim_dynamic'] + 0.5, color=C_ID, ls='--', lw=0.9, alpha=0.7)
    ax2.axvline(gs['dim_gravity'] + 0.5, color='#2ca02c', ls='--', lw=0.9, alpha=0.7)
    ax2.axhline(1e-9, color=C_REF, ls=':', lw=0.9)
    ax2.set_xlabel('singular-value index')
    ax2.set_ylabel(r'$\sigma_k / \sigma_1$')
    ax2.set_title('(b) Dynamic vs static excitation')
    ax2.set_ylim(1e-18, 3)
    ax2.legend(loc='lower left')

    fig.suptitle('Identifiability is a property of the kinematic chain, not the inertia values: '
                 'zyzyzyz → rank 43; zyxyxyx → 43 strong + 18 weak + 9 null',
                 fontsize=10, y=1.04)
    save(fig, 'fig10_identifiability_spectrum')


# ---------------------------------------------------------------------------
# Fig 11 —— 最小二乘的参数误差 100% 落在零空间
# ---------------------------------------------------------------------------
def fig_ls_error():
    d = load('ls_exact_real.json')
    fig, (ax, ax2, ax3) = plt.subplots(1, 3, figsize=(10.4, 2.9),
                                       gridspec_kw={'width_ratios': [1, 1.5, 1.2]})

    vals = [d['e_id_norm'], d['e_null_norm']]
    labs = [f'$\\|e_{{\\rm id}}\\|$\n(rank-{d["rank"]} subspace)',
            f'$\\|e_{{\\rm null}}\\|$\n({d["nullspace"]}-dim nullspace)']
    bars = ax.bar(labs, np.maximum(vals, 1e-18), color=[C_ID, C_NULL], width=0.6)
    ax.set_yscale('log')
    ax.set_ylim(1e-17, 1e3)
    ax.set_ylabel('parameter-error norm')
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, max(v, 1e-18) * 1.8,
                f'{v:.2e}', ha='center', fontsize=8)
    ax.axhline(np.finfo(float).eps * d['e_norm'], color=C_REF, ls=':', lw=0.9)
    ax.text(-0.42, np.finfo(float).eps * d['e_norm'] * 1.6, 'machine precision',
            color=C_REF, fontsize=7)
    ax.set_title(f'(a) Least squares: $R^2_\\tau$ = {d["r2_torque"]:.8f}')

    # 逐参数误差，按类型
    names = ['m', 'hx', 'hy', 'hz', 'Ixx', 'Iyy', 'Izz', 'Ixy', 'Ixz', 'Iyz']
    n_body = d['K'] // 10
    P_hat = np.asarray(d['pi_hat']).reshape(n_body, 10)
    P_tru = np.asarray(d['pi_true']).reshape(n_body, 10)
    err = np.abs(P_hat - P_tru)
    x = np.arange(10)
    for i in range(n_body):
        ax2.scatter(x + (i - n_body / 2) * 0.06, np.maximum(err[i], 1e-18),
                    s=12, alpha=0.75, edgecolors='none',
                    color=plt.cm.viridis(i / max(n_body - 1, 1)))
    ax2.set_yscale('log')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names)
    ax2.set_ylabel(r'$|\hat\pi - \pi_{\rm true}|$')
    ax2.set_title('(b) Absolute error per parameter, per link')
    sm = plt.cm.ScalarMappable(cmap='viridis',
                               norm=plt.Normalize(1, n_body))
    cb = fig.colorbar(sm, ax=ax2, pad=0.02, fraction=0.045)
    cb.set_label('link index', fontsize=8)

    # (c) 噪声扫描：e_id 随噪声线性增长，e_null 纹丝不动
    sweep = [(0.0, d)]
    for s in ('0.01', '0.05'):
        try:
            sweep.append((float(s), load(f'ls_exact_real_noise{s}.json')))
        except FileNotFoundError:
            pass
    x = np.array([s for s, _ in sweep])
    e_id = np.array([max(r['e_id_norm'], 1e-18) for _, r in sweep])
    e_nl = np.array([r['e_null_norm'] for _, r in sweep])
    xp = np.where(x == 0, 10 ** (np.log10(x[x > 0].min()) - 1.0) if (x > 0).any() else 1e-4, x)

    ax3.loglog(xp, e_id, 'o-', color=C_ID, ms=5, lw=1.2,
               label=r'$\|e_{\rm id}\|$  (identifiable)')
    ax3.loglog(xp, e_nl, 's-', color=C_NULL, ms=5, lw=1.2,
               label=r'$\|e_{\rm null}\|$  (nullspace)')
    if (x > 0).sum() >= 2:
        i, j = np.where(x > 0)[0][:2]
        ratio = e_id[j] / e_id[i]
        ax3.annotate(f'{ratio:.2f}× for {x[j]/x[i]:.0f}× noise\n(linear in $\\sigma$)',
                     xy=(xp[j], e_id[j]), xytext=(0.35, 0.30),
                     textcoords='axes fraction', fontsize=7.5, color=C_ID,
                     arrowprops=dict(arrowstyle='->', color=C_ID, lw=0.8))
    ax3.set_xticks(xp)
    ax3.set_xticklabels(['0'] + [f'{v:.0%}' for v in x[1:]])
    ax3.set_xlabel(r'relative Gaussian noise on $\tau$')
    ax3.set_ylabel('error norm')
    ax3.set_ylim(1e-17, 1e3)
    ax3.set_title('(c) Noise separates the components')
    ax3.legend(loc='center right', fontsize=7.5)

    fig.suptitle('Unbiased least squares fits the torque exactly and puts '
                 f'{d["null_energy_frac"]:.2%} of its parameter error in the nullspace',
                 fontsize=10, y=1.04)
    save(fig, 'fig11_ls_nullspace_error')


# ---------------------------------------------------------------------------
# Fig 12 —— 重力子空间含于可辨识子空间
# ---------------------------------------------------------------------------
def fig_gravity_subspace():
    d = load('gravity_subspace_real.json')
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.9),
                                  gridspec_kw={'width_ratios': [1.3, 1]})

    ang = np.asarray(d['principal_angles_deg'])
    ax.plot(np.arange(1, len(ang) + 1), np.maximum(ang, 1e-12), 'o-',
            color=C_ID, ms=4, lw=1.0)
    ax.set_yscale('log')
    ax.axhline(90, color=C_NULL, ls='--', lw=1.0)
    ax.text(0.6, 55, 'orthogonal (90°) — no containment', color=C_NULL, fontsize=7)
    ax.set_ylim(1e-12, 400)
    ax.set_xlabel(r'principal-angle index between $\mathcal{G}$ and $\mathcal{D}$')
    ax.set_ylabel('angle (degrees)')
    ax.set_title(f'(a) All {d["dim_gravity"]} principal angles $< '
                 f'{ang.max():.1e}^\\circ$')

    resid = [d['resid_G_in_D'], d['resid_D_in_G']]
    labs = [r'$\mathcal{G}\subseteq\mathcal{D}$' + f'\n({d["dim_gravity"]}→{d["dim_dynamic"]})',
            r'$\mathcal{D}\subseteq\mathcal{G}$' + f'\n({d["dim_dynamic"]}→{d["dim_gravity"]})']
    bars = ax2.bar(labs, np.maximum(resid, 1e-18), color=[C_ID, C_NULL], width=0.55)
    ax2.set_yscale('log')
    ax2.set_ylim(1e-17, 30)
    ax2.set_ylabel(r'$\max_v \|(I-P)\,v\|$')
    for b, v in zip(bars, resid):
        ax2.text(b.get_x() + b.get_width() / 2, v * 2.2, f'{v:.2e}',
                 ha='center', fontsize=8)
    ax2.text(0, 4e-13, 'holds', ha='center', color=C_ID, fontsize=8, weight='bold')
    ax2.text(1, 4.0, 'fails', ha='center', color=C_NULL, fontsize=8, weight='bold')
    ax2.set_title('(b) Containment residuals')

    fig.suptitle('The gravity subspace lies inside the identifiable subspace — '
                 'why control succeeds while parameters fail',
                 fontsize=10, y=1.04)
    save(fig, 'fig12_gravity_subspace')


if __name__ == '__main__':
    print('绘制审计三图：')
    fig_spectrum()
    fig_ls_error()
    fig_gravity_subspace()
    print('完成。')
