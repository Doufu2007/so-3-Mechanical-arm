"""从 robosuite 的 Baxter robot.xml 提取**左臂**，剥成无网格最小模型。

Baxter 全模型是双 7 轴臂 + 躯干 + 头 + 手的完整机器人。可辨识性分析只关心
「一条 7 轴串联链」的惯性参数秩，所以本脚本：

  1. 剥掉 <asset>/<default>/所有 geom/site/camera/light（同 strip_meshes.py）；
  2. 剥掉关节的 range/limited/damping/armature（限位约束力会污染 mj_inverse，见
     strip_meshes.py 的 docstring）；
  3. 删掉执行器/传动/等式约束/接触/关键帧/传感器；
  4. 把 left_arm_mount 重挂到 worldbody，并用 MuJoCo 在 q=0 读出的世界位姿覆盖它
     原来的父系相对位姿 —— 这样 base→torso→left_arm_mount 的固定链被「折叠」成一个
     固定的世界系根 body，_moving_bodies 只数到 7 个带关节的连杆；
  5. 删掉零质量的末端 left_hand（robosuite 里 gripper 是后加的，这里只是挂点），
     与 panda_real.xml 的 7-body 口径一致（K = 10×7 = 70）。

输出 tool/baxter_left.xml，结构同 panda_real.xml：worldbody → 固定根 → 7 个带铰链
关节的连杆。动力学与官方 Baxter 一致（<inertial> 逐字保留）。

用法：
    python paper/make_baxter_left.py --src /path/to/baxter_robot.xml
"""
from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def mat2quat(R: np.ndarray) -> np.ndarray:
    """3x3 旋转矩阵 -> (w,x,y,z) 四元数（MuJoCo 约定）。"""
    R = np.asarray(R, dtype=np.float64)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def strip(root: ET.Element) -> None:
    """就地剥掉 asset/default/geom/site/camera/light/actuator 等，去关节限位。"""
    for tag in ('asset', 'default', 'actuator', 'tendon', 'equality',
                'contact', 'keyframe', 'sensor'):
        for e in root.findall(tag):
            root.remove(e)

    def scrub(parent):
        for child in list(parent):
            if child.tag in ('geom', 'site', 'camera', 'light'):
                parent.remove(child)
                continue
            scrub(child)
    scrub(root)

    for j in root.iter('joint'):
        for k in ('armature', 'damping', 'frictionloss', 'range', 'limited', 'ref'):
            j.attrib.pop(k, None)
    for c in root.findall('compiler'):
        c.set('autolimits', 'false')


def drop_massless_leaves(parent: ET.Element) -> None:
    for child in list(parent):
        if child.tag != 'body':
            continue
        drop_massless_leaves(child)
        if any(g.tag == 'body' for g in child):
            continue
        inert = child.find('inertial')
        if inert is None or float(inert.get('mass', '1')) == 0.0:
            print(f'  drop massless leaf body: {child.get("name")}')
            parent.remove(child)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='robosuite 的 baxter robot.xml 路径')
    ap.add_argument('--dst', default=str(Path(__file__).resolve().parent.parent
                                         / 'tool' / 'baxter_left.xml'))
    a = ap.parse_args()

    tree = ET.parse(a.src)
    root = tree.getroot()
    strip(root)

    # 剥净后的全模型 -> 用 from_xml_string 算 left_arm_mount 的世界位姿。
    # 不能用 from_xml_path：MuJoCo 的 C 文件 I/O 打不开非 ASCII 绝对路径（见
    # RESULTS_MAP.md「需 ASCII 路径」），而本机项目目录含中文。剥净后已无 mesh 引用，
    # 所以 from_xml_string 是安全的（无外部资源依赖）。
    import mujoco
    m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    mid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'left_arm_mount')
    pos = np.asarray(d.xpos[mid], dtype=np.float64)
    xmat = np.asarray(d.xmat[mid], dtype=np.float64).reshape(3, 3)
    quat = mat2quat(xmat)

    # 重挂：left_arm_mount 设为世界位姿，取代 base 成为 worldbody 唯一子节点
    wb = root.find('worldbody')
    lam = None
    for b in root.iter('body'):
        if b.get('name') == 'left_arm_mount':
            lam = b
            break
    assert lam is not None, '找不到 left_arm_mount'
    lam.set('pos', ' '.join(f'{v:.10g}' for v in pos))
    lam.set('quat', ' '.join(f'{v:.10g}' for v in quat))
    for c in list(wb):
        wb.remove(c)
    wb.append(lam)
    drop_massless_leaves(wb)

    # 顶层 <mujoco> 的 model 名改掉，避免与全模型混淆
    root.set('model', 'baxter_left')

    Path(a.dst).parent.mkdir(parents=True, exist_ok=True)
    tree.write(a.dst, encoding='utf-8', xml_declaration=False)
    print(f'{a.src} -> {a.dst}')

    # 自检：能加载、7 个铰链关节、7 个运动连杆、qpos0==0
    m2 = mujoco.MjModel.from_xml_string(Path(a.dst).read_text(encoding='utf-8'))
    d2 = mujoco.MjData(m2)
    mujoco.mj_forward(m2, d2)
    print(f'  nbody={m2.nbody}  nv={m2.nv}  nq={m2.nq}')
    assert m2.nq == m2.nv == 7, '应恰好 7 个铰链关节'
    assert np.allclose(m2.qpos0, 0.0, atol=1e-12), 'qpos0 非零'
    for i in range(1, m2.nbody):
        name = mujoco.mj_id2name(m2, mujoco.mjtObj.mjOBJ_BODY, i)
        print(f'    body{i} {name:20s} m={m2.body_mass[i]:.4f} '
              f'parent={m2.body_parentid[i]}')


if __name__ == '__main__':
    main()
