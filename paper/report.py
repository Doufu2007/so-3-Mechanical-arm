"""
把所有实验结果汇编成一份 RESULTS.md —— 论文正文的事实底稿。

规矩：**报告里出现的每个数字都从 json 读出来**，不允许手写。
这样 README / 稿件里的说法永远与最新一次实验一致，不会再出现
「README 写 +22.3%、实测 +0.023」这种对不上的情况。

    python paper/report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paper import core
from paper.core import ARMS

ROOT = core.ROOT
OUT = ROOT / 'RESULTS.md'

ARM_LABEL = {'baxter': 'Baxter（真实机械臂实测数据）',
             'franka_ex': '7 轴仿真臂（激励充分数据集 Dataset_Franka_Ex）',
             'franka': '7 轴仿真臂（原始窄覆盖数据集 Dataset_Franka）'}
DISPLAY = {
    'full': 'Ours（rotmat 编码，7 维瓶颈）',
    'enc_rotmat_wide': 'Ours（rotmat 编码，无瓶颈）',
    'enc_sincos': 'Ours（sin/cos 编码，7 维瓶颈）',
    'enc_sincos_wide': 'Ours（sin/cos 编码，无瓶颈）',
    'enc_none': 'Ours（关节角直接输入）',
    'no_physics': '去掉物理分支（纯黑箱 FFNN）',
    'no_uncertainty': '去掉残差分支（纯 DeLaN）',
    'no_adaptive_lam': '固定 λ（去掉自适应 λ）',
    'no_multiscale': '普通 Huber（去掉多尺度加权）',
    'dual_lam': '对偶上升 λ（约束式残差预算）',
    'dual_lam_wide': '对偶上升 λ（sin/cos、无瓶颈）',
    'mlp': 'MLP', 'hnn': 'HNN', 'lnn': 'LNN', 'symoden': 'SymODEN',
    'swevers': 'Swevers 刚体辨识（经典最小二乘）',
    'se3': 'SE(3) 结构化惯量（M=ΣJᵀGJ，核心）',
    'se3_no_uncertainty': 'SE(3) 结构单独（无残差）',
}
ORDER = list(DISPLAY)


def jload(p):
    p = ROOT / p if not str(p).startswith('/') else Path(p)
    if not Path(p).exists():
        return None
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def sec(title, level=2):
    return f"\n{'#' * level} {title}\n"


def main():
    L = ['# 实验结果底稿（自动生成，请勿手改）', '',
         f'生成时间：{core.provenance()["timestamp"]}　·　'
         f'git commit：`{core.git_commit()[:10]}`　·　'
         f'GPU：{core.provenance()["gpu"]}', '',
         '> 本文件由 `python paper/report.py` 从 `experimental_results/` 下的 json 汇编而成。',
         '> 所有数字都是读出来的，没有一个是手写的。']

    # ---------------- 数据集 ----------------
    L.append(sec('0. 数据集与评测协议'))
    L.append('| 数据集 | 轨迹数(train/val/test) | 样本数 | 说明 |')
    L.append('|---|---|---|---|')
    for arm in ['baxter', 'franka_ex', 'franka']:
        try:
            d = core.load_arm(arm)
        except Exception:
            continue
        m = d['manifest']
        L.append(f"| `{arm}` | {len(m['train'])}/{len(m['val'])}/{len(m['test'])} | "
                 f"{len(d['train'][0])}/{len(d['val'][0])}/{len(d['test'][0])} | "
                 f"{ARMS[arm].note or '—'} |")
    L += ['',
          '评测协议：轨迹级三划分；归一化统计量只从 train 估计；**选模只看验证集，'
          '测试集用最优权重评一次**。主指标 `R²(global)` 为所有关节残差平方和与总平方和'
          '合并统计（等价于按扭矩量级加权），不受数值退化关节支配。']

    # ---------------- 主表 ----------------
    t1 = jload('experimental_results/v2_tables/table1_main.json')
    if t1:
        L.append(sec('1. 主对比结果'))
        arms = sorted({v['arm'] for v in t1.values()},
                      key=lambda a: ['baxter', 'franka_ex', 'franka'].index(a))
        for arm in arms:
            L.append(f'**{ARM_LABEL.get(arm, arm)}**\n')
            L.append('| 方法 | R²(global) | R²(活跃关节) | RMSE (N·m) | 参数量 | seeds |')
            L.append('|---|---|---|---|---|---|')
            rows = [(v, t1[f'{arm}/{v}']) for v in ORDER if f'{arm}/{v}' in t1]
            for v, e in rows:
                L.append(f"| {DISPLAY[v]} | {e['r2_global_mean']:.4f} ± {e['r2_global_std']:.4f} "
                         f"| {e['r2_active_joints_mean']:.4f} ± {e['r2_active_joints_std']:.4f} "
                         f"| {e['rmse_overall_mean']:.4f} | {e['n_params']/1e6:.2f}M "
                         f"| {e['n_seeds']} |")
            if rows:
                best = max(rows, key=lambda r: r[1]['r2_global_mean'])
                L.append(f"\n该臂上最优配置：**{DISPLAY[best[0]]}**"
                         f"（R²(global) = {best[1]['r2_global_mean']:.4f}）。\n")

    # ---------------- 显著性 ----------------
    st = jload('experimental_results/v2_tables/stats.json')
    if st:
        L.append(sec('2. 消融与显著性检验'))
        L.append('Welch t 检验（不假设等方差），Holm–Bonferroni 多重比较校正，'
                 '指标为测试集 R²(global)。\n')
        L.append('| 臂 | 对比 | ΔR² | p (raw) | p (Holm) | Cohen d | 结论 |')
        L.append('|---|---|---|---|---|---|---|')
        for k, s in st.items():
            ph = s.get('p_holm', float('nan'))
            verdict = ('**显著**' if ph < 0.05 else '不显著')
            if np.isfinite(ph) and ph < 0.05 and s['delta'] < 0:
                verdict = '**显著为负**'
            L.append(f"| {s['arm']} | {s['desc']} | {s['delta']:+.4f} | {s['p']:.2e} "
                     f"| {ph:.2e} | {s['cohen_d']:+.2f} | {verdict} |")

    # ---------------- 逐关节 ----------------
    pj = jload('experimental_results/v2_tables/table3_per_joint.json')
    if pj:
        L.append(sec('3. 逐关节 R²'))
        L.append('把「逐关节 R² 取平均」当主指标是不成立的：扭矩标准差接近 0 的关节'
                 '（下表最后一行）其 R² 是纯数值噪声。\n')
        for arm in ['baxter', 'franka_ex', 'franka']:
            keys = [k for k in pj if k.startswith(f'{arm}/')]
            if not keys:
                continue
            names = ARMS[arm].joint_names
            L.append(f'**{ARM_LABEL.get(arm, arm)}**\n')
            L.append('| 方法 | ' + ' | '.join(names) + ' |')
            L.append('|---' * (len(names) + 1) + '|')
            for v in ORDER:
                k = f'{arm}/{v}'
                if k in pj:
                    L.append(f'| {DISPLAY[v]} | ' +
                             ' | '.join(f'{m:.3f}' for m in pj[k]['mean']) + ' |')
            L.append('| *τ 标准差 (N·m)* | ' +
                     ' | '.join(f'{s:.4f}' for s in pj[keys[0]]['tau_std']) + ' |\n')

    # ---------------- 物理核验 ----------------
    pdir = ROOT / 'experimental_results/v2_physics'
    reps = [jload(f) for f in sorted(pdir.glob('*.json'))] if pdir.exists() else []
    reps = [r for r in reps if r and 'A_inertia' in r]
    if reps:
        L.append(sec('4. 物理正确性核验'))
        L.append('| 模型 | H 对称误差 | H 最小特征值 | 正定比例 | cond(H) 中位数 | '
                 '能量恒等式相对误差 | JVP vs 显式 |')
        L.append('|---|---|---|---|---|---|---|')
        for r in reps:
            a, b, c = r['A_inertia'], r['B_energy_identity'], r['C_jvp_consistency']
            L.append(f"| {r['variant']}/{r['arm']} | {a['max_asymmetry']:.1e} "
                     f"| {a['min_eigenvalue']:.2e} | {a['frac_positive_definite']:.3f} "
                     f"| {a['cond_median']:.3g} | {b['rel_err_mean']:.2e} "
                     f"| {c['rel_err_mean']:.2e} |")
        mj = [r for r in reps if r.get('D_vs_mujoco', {}).get('available')]
        if mj:
            L.append('\n**与 MuJoCo 解析真值对比**（仅仿真臂有真值）\n')
            L.append('| 模型 | ‖ΔH‖/‖H‖ | ‖ΔH·q̈‖/‖H·q̈‖ | ‖Δg‖/‖g‖ | '
                     '真值 cond(H) 均值 | 学到的 cond(H) 均值 |')
            L.append('|---|---|---|---|---|---|')
            for r in mj:
                d = r['D_vs_mujoco']
                L.append(f"| {r['variant']}/{r['arm']} | {d['H_rel_fro_err']:.3f} "
                         f"| {d['Hddq_rel_err']:.3f} | {d['g_rel_err']:.3f} "
                         f"| {d['true_cond_mean']:.3g} | {r['A_inertia']['cond_mean']:.3g} |")
        lam = [r for r in reps if r.get('E_lambda', {}).get('available')]
        if lam:
            L.append('\n**自适应 λ 分布**（上限 0.1；若塌缩到 0 说明物理约束项失效）\n')
            L.append('| 模型 | 均值 | 标准差 | 最小 | 最大 | <1e-3 的比例 |')
            L.append('|---|---|---|---|---|---|')
            for r in lam:
                e = r['E_lambda']
                L.append(f"| {r['variant']}/{r['arm']} | {e['mean']:.4f} | {e['std']:.4f} "
                         f"| {e['min']:.4f} | {e['max']:.4f} | {e['frac_below_1e-3']:.3f} |")
        L.append('\n**分支贡献分解**\n')
        L.append('| 模型 | 整体 R²(global) | 仅物理分支 R²(global) | 残差分支 RMS (N·m) | 占 τ RMS |')
        L.append('|---|---|---|---|---|')
        for r in reps:
            f = r['F_branch_split']
            po = f.get('physics_only', {}).get('r2_global')
            L.append(f"| {r['variant']}/{r['arm']} | {f['total']['r2_global']:.4f} "
                     f"| {po:.4f} |" if po is not None else
                     f"| {r['variant']}/{r['arm']} | {f['total']['r2_global']:.4f} | — |")
            L[-1] += (f" {f.get('uncertainty_rms_Nm', float('nan')):.4f} "
                      f"| {f.get('uncertainty_share', float('nan'))*100:.1f}% |")

    # ---------------- 数据效率 ----------------
    de_dir = ROOT / 'experimental_results/v2_data_eff'
    de = [jload(f) for f in sorted(de_dir.glob('*.json'))] if de_dir.exists() else []
    if de and t1:
        L.append(sec('5. 数据效率'))
        for arm in ['baxter', 'franka_ex', 'franka']:
            rs = [r for r in de if r['arm'] == arm]
            if not rs:
                continue
            fracs = sorted({r['frac'] for r in rs}) + [1.0]
            vs = [v for v in ['full', 'enc_rotmat_wide', 'no_physics', 'mlp']
                  if any(r['variant'] == v for r in rs)]
            L.append(f'**{ARM_LABEL.get(arm, arm)}** — 测试集 R²(global)\n')
            L.append('| 方法 | ' + ' | '.join(f'{f*100:g}%' for f in fracs) + ' |')
            L.append('|---' * (len(fracs) + 1) + '|')
            for v in vs:
                cells = []
                for fr in fracs:
                    if fr == 1.0:
                        e = t1.get(f'{arm}/{v}')
                        cells.append(f"{e['r2_global_mean']:.4f}" if e else '—')
                    else:
                        vals = [r['test']['r2_global'] for r in rs
                                if r['variant'] == v and abs(r['frac'] - fr) < 1e-9]
                        cells.append(f'{np.mean(vals):.4f}' if vals else '—')
                L.append(f'| {DISPLAY[v]} | ' + ' | '.join(cells) + ' |')
            L.append('')

    # ---------------- 噪声 ----------------
    nd = ROOT / 'experimental_results/v2_noise'
    if nd.exists():
        L.append(sec('6. 噪声鲁棒性'))
        for f in sorted(nd.glob('noise_*.json')):
            d = jload(f)
            L.append(f"**{ARM_LABEL.get(d['arm'], d['arm'])}** — {d['noise_definition']}\n")
            lv = d['noise_levels']
            L.append('| 方法 | ' + ' | '.join(f'σ={s:g}' for s in lv) + ' |')
            L.append('|---' * (len(lv) + 1) + '|')
            for v, r in d['results'].items():
                L.append(f"| {DISPLAY.get(v, v)} | " +
                         ' | '.join(f'{x:.4f}' for x in r['r2_mean']) + ' |')
            L.append('')

    # ---------------- 控制 ----------------
    ct = jload('experimental_results/v2_control/control.json')
    if ct:
        L.append(sec('7. 闭环控制（重力补偿定点调节）'))
        L.append(f"被控对象：{ct['plant']}；{ct['n_postures']} 个目标姿态，"
                 f"Kp={ct['kp']:g}、Kd={ct['kd']:.1f}，从零位形拉到位并在重力下保持，"
                 f"报稳态跟踪误差 RMSE。\n")
        L.append('| 控制器 | 稳态 RMSE (mrad) |')
        L.append('|---|---|')
        names = {'pd': 'PD 无补偿', 'pd_grav_mlp': 'PD + MLP 静平衡输出',
                 'pd_grav_learn': 'PD + 学到的 g(q)', 'pd_grav_true': 'PD + 真值 g(q)'}
        for c in ['pd', 'pd_grav_mlp', 'pd_grav_learn', 'pd_grav_true']:
            s = ct['summary'].get(c)
            if s:
                L.append(f"| {names[c]} | {s['rmse_mean']*1000:.2f} ± {s['rmse_std']*1000:.2f} |")
        L.append('')

    # ---------------- 复杂度 ----------------
    cb = jload('experimental_results/v2_tables/coriolis_benchmark.json')
    if cb:
        L.append(sec('8. 科里奥利项计算的复杂度'))
        L.append('同一个 H 网络，只改 Ḣ·q̇ 的求导方式。batch = '
                 f"{cb['rows'][0]['batch']}，设备 {cb['device']}。\n")
        L.append('| n | 显式 ∂H/∂q (ms) | reverse ×n (ms) | JVP (ms) | JVP 相对显式加速 | '
                 '显式峰值显存 (MiB) | JVP 峰值显存 (MiB) |')
        L.append('|---|---|---|---|---|---|---|')
        for r in cb['rows']:
            sp = r.get('speedup_explicit_over_jvp', float('nan'))
            ex = 'OOM' if r.get('explicit_oom') else f"{r['explicit_ms']:.1f}"
            exm = 'OOM' if r.get('explicit_oom') else f"{r['explicit_peak_MiB']:.0f}"
            L.append(f"| {r['n']} | {ex} | {r['reverse_ms']:.1f} | {r['jvp_ms']:.1f} "
                     f"| {'—' if r.get('explicit_oom') else f'{sp:.1f}×'} | {exm} "
                     f"| {r['jvp_peak_MiB']:.0f} |")
        L.append('\n注：这是**科里奥利项本身**的开销对比。在 n=7 的真实机械臂上，'
                 '整步训练时间由网络前向/反向主导，端到端加速远小于此处的比值 —— '
                 '该方法真正的意义是把 O(n³) 的显式 Christoffel 存储彻底去掉。')

    # ---------------- λ 权衡 ----------------
    lt = jload('experimental_results/v2_tables/lam_tradeoff.json')
    if lt and any(r['r2_global_mean'] is not None for r in lt.get('rows', [])):
        L.append(sec('9. 物理权重 λ 的精度–可用性权衡'))
        L.append('固定 λ 的完整模型（rotmat 编码）；R² 为 3 seeds 聚合，g(q) 误差与闭环'
                 ' RMSE 用 seed0（物理核验与闭环只跑 seed0）。闭环任务同 §7。\n')
        L.append('| λ | R²(global) | ‖Δg‖/‖g‖ | 闭环稳态 RMSE (mrad) |')
        L.append('|---|---|---|---|')
        for r in lt['rows']:
            r2 = (f"{r['r2_global_mean']:.4f} ± {r['r2_global_std']:.4f}"
                  if r['r2_global_mean'] is not None else '—')
            g = f"{r['g_rel_err']:.3f}" if r['g_rel_err'] is not None else '—'
            c = f"{r['ctrl_rmse_mrad']:.2f}" if r['ctrl_rmse_mrad'] is not None else '—'
            L.append(f"| {r['lam']:g} | {r2} | {g} | {c} |")
        refs = lt.get('references', {})
        if refs:
            L.append(f"\n参考：自适应 λ→0 闭环 RMSE = "
                     f"{refs.get('ctrl_rmse_mrad_adaptive_lam0', float('nan')):.2f} mrad、"
                     f"MLP 静平衡 = {refs.get('ctrl_rmse_mrad_mlp', float('nan')):.2f} mrad、"
                     f"真值 g(q) = {refs.get('ctrl_rmse_mrad_true', float('nan')):.2e} mrad、"
                     f"无补偿 = {refs.get('ctrl_rmse_mrad_none', float('nan')):.2f} mrad。")
        L.append('')

    # ---------------- 超参敏感性 ----------------
    hd = ROOT / 'experimental_results/v2_hparam'
    if hd.exists() and t1:
        def _hp_vals(prefix, seeds=(0, 1)):
            vs = []
            for s in seeds:
                j = jload(hd / f'{prefix}_franka_ex_seed{s}.json')
                if j and 'test' in j and 'r2_global' in j['test']:
                    vs.append(j['test']['r2_global'])
            return vs
        ref = t1.get('franka_ex/full', {}).get('r2_global_mean')
        rows = []
        for p, label, refval in [('width0.5', 'hidden width ×0.5', ref),
                                 ('width2.0', 'hidden width ×2.0', ref),
                                 ('bs256', 'batch size 256', ref),
                                 ('bs4096', 'batch size 4096', ref)]:
            vs = _hp_vals(p)
            if vs:
                rows.append((label, np.mean(vs), refval))
        if rows:
            L.append(sec('10. 超参数敏感性（franka_ex，seed 0/1）'))
            L.append('完整模型其余超参不变，只改一项；参考值为默认配置'
                     f'（width ×1.0、batch 1024，R²(global) = {ref:.4f}）。\n')
            L.append('| 超参 | R²(global) | 与默认差值 |')
            L.append('|---|---|---|')
            for label, m, rv in rows:
                L.append(f'| {label} | {m:.4f} | {m - rv:+.4f} |')
            L.append('')

    OUT.write_text('\n'.join(L), encoding='utf-8')
    print(f'写入 {OUT}（{len(L)} 行）')


if __name__ == '__main__':
    main()
