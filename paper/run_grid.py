"""
并行网格执行器 —— 单卡多进程。

实测该模型训练是 Python/kernel-launch 开销受限（单进程 GPU 利用率约 7%），
5 个并发进程的墙钟时间与单进程几乎相同，因此用进程池把网格跑满。

    python paper/run_grid.py --grid main --workers 8
    python paper/run_grid.py --grid main --workers 8 --dry-run
"""
from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
LOG_DIR = ROOT / 'logs'

ABLATION_VARIANTS = ['full', 'enc_sincos', 'enc_none', 'enc_rotmat_wide',
                     'enc_sincos_wide', 'no_physics',
                     'no_uncertainty', 'no_adaptive_lam', 'no_multiscale',
                     'dual_lam', 'dual_lam_wide',
                     'se3', 'se3_no_uncertainty']
BASELINES = ['mlp', 'hnn', 'lnn', 'symoden']
ARMS = ['baxter', 'franka_ex', 'franka']
SEEDS = [0, 1, 2, 3, 4]

# SE(3) 分支需要 MuJoCo 解析模型（tool/panda.xml）提取几何，只能跑仿真臂；
# Baxter 是实机数据、无解析模型，se3 变体自动跳过。
SE3_ONLY = {'se3', 'se3_no_uncertainty'}


def job(arm, variant, seed, frac=1.0, out_dir='experimental_results/v2', extra=None):
    name = f'{variant}_{arm}_seed{seed}' + (f'_frac{frac:g}' if frac < 1.0 else '')
    cmd = [PY, str(ROOT / 'paper' / 'train.py'), '--arm', arm, '--variant', variant,
           '--seed', str(seed), '--out-dir', out_dir, '--skip-existing']
    if frac < 1.0:
        cmd += ['--frac', str(frac)]
    if extra:
        cmd += extra
    return name, cmd, out_dir, arm


def build_grid(which, seeds=None, quick=False):
    seeds = seeds or SEEDS
    jobs = []
    if which in ('main', 'all'):
        for arm, v, s in itertools.product(ARMS, ABLATION_VARIANTS + BASELINES, seeds):
            if arm == 'baxter' and v in SE3_ONLY:
                continue
            jobs.append(job(arm, v, s))
    if which in ('lam',):
        # λ 扫描：固定 λ 的完整模型（physics + uncertainty），只做关键权衡验证
        # λ 上限从 0.5 扩到 5.0：实测 λ=0.05 时 g(q) 仍差（闭环 32mrad），
        # 需更大 λ 才能逼物理分支学准重力项，看到「精度↔物理保真度」权衡。
        hp_dir = 'experimental_results/v2_hparam'
        for arm, s in itertools.product(['franka_ex'], [0, 1, 2]):
            for lam in [0.0, 0.1, 0.5, 1.0, 5.0]:
                tag = f'lam{lam}_{arm}_seed{s}'
                _, cmd, _, _ = job(arm, 'no_adaptive_lam', s, out_dir=hp_dir,
                                   extra=['--lambda-reg', str(lam), '--tag', tag])
                jobs.append((tag, cmd, hp_dir, arm))
    if which in ('dual',):
        # 对偶上升残差预算 δ 扫描：δ_frac 控制残差分支允许占 τ RMS 的比例。
        # δ 越大残差越自由（越像 λ→0），δ 越小物理分支越被逼着承担扭矩（越像大 λ）。
        hp_dir = 'experimental_results/v2_hparam'
        for arm, s in itertools.product(['franka_ex', 'baxter'], [0, 1, 2]):
            for dfrac in [0.1, 0.2, 0.3, 0.5]:
                tag = f'dual_d{dfrac}_{arm}_seed{s}'
                _, cmd, _, _ = job(arm, 'dual_lam', s, out_dir=hp_dir,
                                   extra=['--dual-delta-frac', str(dfrac), '--tag', tag])
                jobs.append((tag, cmd, hp_dir, arm))
    if which in ('data_eff', 'all'):
        for arm, v, s, fr in itertools.product(
                ARMS, ['full', 'no_physics', 'mlp'], [0, 1, 2], [0.05, 0.1, 0.25, 0.5]):
            jobs.append(job(arm, v, s, frac=fr, out_dir='experimental_results/v2_data_eff'))
    if which in ('hparam', 'all'):
        # 超参数敏感性：固定 λ / 网络宽度 / batch size，各扫几个点，seed 0/1
        hp_dir = 'experimental_results/v2_hparam'
        for arm, s in itertools.product(['baxter', 'franka_ex'], [0, 1]):
            specs = ([('no_adaptive_lam', ['--lambda-reg', str(v)], f'lam{v}') for v in
                      [0.0, 0.01, 0.1, 0.5]] +
                     [('full', ['--hidden-scale', str(v)], f'width{v}') for v in
                      [0.25, 0.5, 2.0]] +
                     [('full', ['--batch-size', str(v)], f'bs{v}') for v in [256, 4096]])
            for variant, extra, label in specs:
                tag = f'{label}_{arm}_seed{s}'
                _, cmd, _, _ = job(arm, variant, s, out_dir=hp_dir,
                                   extra=extra + ['--tag', tag])
                jobs.append((tag, cmd, hp_dir, arm))
    if quick:
        for j in jobs:
            j[1].extend(['--epochs', '20'])
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', default='main')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--seeds', type=int, nargs='*', default=None)
    ap.add_argument('--variants', nargs='*', default=None)
    ap.add_argument('--arms', nargs='*', default=None)
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    jobs = build_grid(a.grid, a.seeds, a.quick)
    if a.variants:
        jobs = [j for j in jobs if any(j[0].startswith(v + '_') for v in a.variants)]
    if a.arms:
        jobs = [j for j in jobs if j[3] in a.arms]

    # 已有结果的直接跳过（train.py 也有 --skip-existing，这里先过滤省得起进程）
    pending = [j for j in jobs if not (ROOT / j[2] / f'{j[0]}.json').exists()]
    print(f'总任务 {len(jobs)}，待跑 {len(pending)}，并发 {a.workers}')
    if a.dry_run:
        for n, c, _, _ in pending:
            print('  ', n)
        return

    LOG_DIR.mkdir(exist_ok=True)
    running, done, failed = [], [], []
    t0 = time.time()
    queue = list(pending)

    while queue or running:
        while queue and len(running) < a.workers:
            name, cmd, _, _ = queue.pop(0)
            log = open(LOG_DIR / f'{name}.log', 'w')
            p = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            running.append((name, p, log, time.time()))
            print(f'[start] {name}  ({len(done)+len(failed)}/{len(pending)} 完成, '
                  f'{len(queue)} 排队)', flush=True)
        time.sleep(5)
        still = []
        for name, p, log, ts in running:
            if p.poll() is None:
                still.append((name, p, log, ts))
            else:
                log.close()
                (done if p.returncode == 0 else failed).append(name)
                flag = 'ok' if p.returncode == 0 else f'FAIL rc={p.returncode}'
                print(f'[{flag}] {name}  {(time.time()-ts)/60:.1f} min', flush=True)
        running = still

    print(f'\n完成 {len(done)}，失败 {len(failed)}，总耗时 {(time.time()-t0)/60:.1f} min')
    if failed:
        print('失败任务:', failed)
        print(f'日志见 {LOG_DIR}')
        sys.exit(1)


if __name__ == '__main__':
    main()
