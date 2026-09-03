"""
批量物理核验 —— 对 results_v2/ 下每个 checkpoint 跑一次 verify_physics，多进程并行。

产出 experimental_results/v2_physics/<name>_physics.json。

**这是论文招牌图的唯一数据源**：横轴 R²(global)、纵轴 ‖Δg‖/‖g‖（或 cond(H)、
闭环 mrad），每个点是一个 (变体, 臂, seed)。现在 v2_physics 里只有 6 份、且全是
franka_ex 的 seed0，散点图画不出来。跑完这一步才有几百个点。

verify_physics.py 自己 catch 掉每个 checkpoint 的异常并照常退出 0，所以这里
不看返回码，改为**检查输出 json 是否真的生成**来判定成败。

    python paper/verify_all.py --workers 4              # 全部（跳过已有）
    python paper/verify_all.py --workers 4 --dry-run    # 只列清单
    python paper/verify_all.py --pattern '*franka_ex*.pth'
    python paper/verify_all.py --redo                   # 已有的也重跑
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
CKPT_DIR = ROOT / 'results_v2'
OUT_DIR = ROOT / 'experimental_results' / 'v2_physics'
LOG_DIR = ROOT / 'logs_verify'


def out_json(ckpt: Path) -> Path:
    return OUT_DIR / f'{ckpt.stem}_physics.json'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--pattern', default='*.pth')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--redo', action='store_true', help='已有结果的也重跑')
    a = ap.parse_args()

    ckpts = sorted(CKPT_DIR.glob(a.pattern))
    if not ckpts:
        print(f'!! {CKPT_DIR} 下没有匹配 {a.pattern} 的 checkpoint。')
        print('   物理核验依赖训练产出的 .pth —— 请先把主网格跑完（run_server_v2.sh 步骤 3）。')
        sys.exit(1)

    pending = ckpts if a.redo else [p for p in ckpts if not out_json(p).exists()]
    print(f'checkpoint {len(ckpts)} 个，待核验 {len(pending)}，并发 {a.workers}')
    if a.dry_run:
        for p in pending:
            print('  ', p.name)
        return
    if not pending:
        print('全部已有结果，无事可做。')
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)

    queue = list(pending)
    running, done, failed = [], [], []
    t0 = time.time()

    while queue or running:
        while queue and len(running) < a.workers:
            ck = queue.pop(0)
            log = open(LOG_DIR / f'{ck.stem}.log', 'w', encoding='utf-8')
            cmd = [PY, str(ROOT / 'paper' / 'verify_physics.py'), '--ckpt', str(ck)]
            p = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            running.append((ck, p, log, time.time()))
            print(f'[start] {ck.stem}  ({len(done)+len(failed)}/{len(pending)} 完成, '
                  f'{len(queue)} 排队)', flush=True)
        time.sleep(3)
        still = []
        for ck, p, log, ts in running:
            if p.poll() is None:
                still.append((ck, p, log, ts))
                continue
            log.close()
            # verify_physics 内部吞异常，只能靠产物判定
            ok = out_json(ck).exists()
            (done if ok else failed).append(ck.stem)
            print(f'[{"ok" if ok else "FAIL"}] {ck.stem}  {time.time()-ts:.0f}s', flush=True)
        running = still

    print(f'\n核验完成 {len(done)}，失败 {len(failed)}，总耗时 {(time.time()-t0)/60:.1f} min')
    if failed:
        print('失败清单:', failed[:20], '...' if len(failed) > 20 else '')
        print(f'逐个日志见 {LOG_DIR}')
        # 部分失败不阻断后续步骤：散点图少几个点无所谓，全失败才是问题
        if len(failed) == len(pending):
            sys.exit(1)


if __name__ == '__main__':
    main()
