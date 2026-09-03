"""把 franka_fric / franka_fn 两个新臂注册进 core.py 和 run_grid.py。

**为什么必须同时改两个文件**：run_grid.build_grid() 用模块级的 ARMS 列表
（run_grid.py:53）来"生成"任务列表，而 --arms 只是事后"过滤"（run_grid.py:116）。
如果新臂名不在 run_grid.ARMS 里，`--arms franka_fric` 会把所有任务都过滤掉，
最后跑出 0 个任务却不报错。core.py 那边则是 train.py 子进程真正查表的地方
（run_grid 是 subprocess 调 train.py，所以运行时 monkeypatch 不管用）。

幂等：重复执行不会重复插入。

    python paper/add_arms.py
    python paper/add_arms.py --check     # 只检查，不写
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORE = ROOT / 'paper' / 'core.py'
GRID = ROOT / 'paper' / 'run_grid.py'

MARKER = '# --- friction / noise arms (add_arms.py) ---'

CORE_APPEND = f'''

{MARKER}
# 破自证循环用的两个数据集，几何与 franka_ex 完全相同（同一 tool/panda.xml），
# 唯一区别是标签里多了关节摩擦（Fric）以及测量噪声（FN）。
# 见 paper/make_dataset_fric.py。
ARMS['franka_fric'] = ArmSpec(
    name='franka_fric',
    train_dir='Dataset_Franka_Fric/Training',
    test_dir='Dataset_Franka_Fric/Test',
    val_dir='Dataset_Franka_Fric/Validation',
    has_ddq=True, dt=0.005,
    joint_names=['J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7'],
    axes='zyzyzyz',
    n_val_traj=0,
    note='与 franka_ex 同激励同几何，标签追加库仑+粘滞关节摩擦。'
         'SE(3) 结构分支无法表示摩擦项，用于检验 R²=1.0 是否为无摩擦数据的产物。',
)
ARMS['franka_fn'] = ArmSpec(
    name='franka_fn',
    train_dir='Dataset_Franka_FN/Training',
    test_dir='Dataset_Franka_FN/Test',
    val_dir='Dataset_Franka_FN/Validation',
    has_ddq=True, dt=0.005,
    joint_names=['J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7'],
    axes='zyzyzyz',
    n_val_traj=0,
    note='在 franka_fric 基础上再加 q/dq/ddq/tau 的测量噪声，最接近真实采集条件。',
)
'''

GRID_OLD = "ARMS = ['baxter', 'franka_ex', 'franka']"
GRID_NEW = ("ARMS = ['baxter', 'franka_ex', 'franka', 'franka_fric', 'franka_fn']"
            f"  {MARKER}")


def patch_core(check):
    txt = CORE.read_text(encoding='utf-8')
    if MARKER in txt:
        print('  core.py      已注册，跳过')
        return True
    if not check:
        CORE.write_text(txt + CORE_APPEND, encoding='utf-8')
    print('  core.py      + franka_fric, franka_fn')
    return True


def patch_grid(check):
    txt = GRID.read_text(encoding='utf-8')
    if MARKER in txt:
        print('  run_grid.py  已注册，跳过')
        return True
    if GRID_OLD not in txt:
        print(f'  run_grid.py  !! 找不到这一行，需手工改：\n      {GRID_OLD}')
        return False
    if not check:
        GRID.write_text(txt.replace(GRID_OLD, GRID_NEW, 1), encoding='utf-8')
    print('  run_grid.py  ARMS 列表 + franka_fric, franka_fn')
    return True


def verify():
    """确认 train.py 子进程真的能查到新臂。"""
    sys.path.insert(0, str(ROOT))
    from paper import core
    missing = [a for a in ('franka_fric', 'franka_fn') if a not in core.ARMS]
    if missing:
        print(f'  !! core.ARMS 里仍缺 {missing}')
        return False
    for a in ('franka_fric', 'franka_fn'):
        d = ROOT / core.ARMS[a].train_dir
        n = len(list(d.glob('*.csv'))) if d.exists() else 0
        flag = 'OK ' if n else '!! 数据缺失，先跑 make_dataset_fric.py'
        print(f'  {flag} {a:12s} {core.ARMS[a].train_dir}  ({n} 条轨迹)')
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true', help='只检查不写入')
    a = ap.parse_args()

    print('注册新臂：')
    ok = patch_core(a.check) and patch_grid(a.check)
    if not ok:
        sys.exit(1)
    if not a.check:
        print('\n校验：')
        verify()
        print('\n下一步：bash run_server.sh  （脚本会先 --dry-run 让你确认任务数）')


if __name__ == '__main__':
    main()
