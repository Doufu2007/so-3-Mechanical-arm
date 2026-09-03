"""把 menagerie 的 franka_emika_panda 剥成无网格模型。

可辨识性分析只用到运动学（关节轴、连杆偏置）和惯量（mass/pos/diaginertia/fullinertia），
不需要碰撞或视觉几何。剥掉 <asset> 和所有 geom 后，模型不再依赖 20+ 个 STL/OBJ，
而 <inertial> 是显式写在 XML 里的，逐字保留 —— 所以动力学与官方模型完全一致。

同时移除 armature 和 damping：它们是关节级的传动惯量与阻尼，不属于连杆刚体参数
pi=(m,h,I_o)，留着会让 mj_inverse 的输出偏离 Y·pi，破坏线性性自检。
"""
import sys
import xml.etree.ElementTree as ET

src, dst = sys.argv[1], sys.argv[2]
tree = ET.parse(src)
root = tree.getroot()

# 1. 删掉 asset 整节，以及 compiler 的 meshdir
for tag in ('asset',):
    for e in root.findall(tag):
        root.remove(e)
for c in root.findall('compiler'):
    c.attrib.pop('meshdir', None)

# 2. 递归删掉所有 geom / site / camera / light，以及 default 里的 geom 样式
def scrub(parent):
    for child in list(parent):
        if child.tag in ('geom', 'site', 'camera', 'light'):
            parent.remove(child)
            continue
        scrub(child)

scrub(root)

# 3. 关节：去掉 armature / damping / frictionloss（非刚体参数，见模块 docstring）
#
#    也去掉 range/limited：Panda 的 joint4 范围是 [-3.072, -0.070]，连 q=0 都在范围外，
#    而可辨识性分析按 |q|<=1.4 均匀采样。限位被违反时 mj_inverse 会把约束力
#    （实测可达 1.5e3 N·m）算进 qfrc_inverse，那不属于 tau = Y·pi 的张成空间，
#    会让线性性自检直接失败。可辨识性是刚体惯性参数的结构性质，与限位无关。
for j in root.iter('joint'):
    for k in ('armature', 'damping', 'frictionloss', 'range', 'limited'):
        j.attrib.pop(k, None)
for c in root.findall('compiler'):
    c.set('autolimits', 'false')

# 4. 去掉执行器/传动/接触/等式约束，它们对 mj_inverse 的刚体项没有贡献但会引入额外力
for tag in ('actuator', 'tendon', 'equality', 'contact', 'keyframe', 'sensor'):
    for e in root.findall(tag):
        root.remove(e)

# 5. 删掉零质量的叶子 body（menagerie 的 "attachment" 只是个安装点）。
#    留着的话它那 10 个惯性参数会整体落进零空间，把不可辨识占比灌水 10/90。
def drop_massless_leaves(parent):
    for child in list(parent):
        if child.tag != 'body':
            continue
        drop_massless_leaves(child)
        if any(g.tag == 'body' for g in child):
            continue                                    # 不是叶子
        inert = child.find('inertial')
        # 没有 <inertial> 标签且已被剥掉 geom 的叶子 body，质量由 MuJoCo 补成 0
        if inert is None or float(inert.get('mass', '1')) == 0.0:
            print(f'  drop massless leaf body: {child.get("name")}')
            parent.remove(child)

for wb in root.findall('worldbody'):
    drop_massless_leaves(wb)

tree.write(dst, encoding='utf-8', xml_declaration=False)
print(f'{src} -> {dst}')

import mujoco
m = mujoco.MjModel.from_xml_path(dst)
print(f'  nbody={m.nbody}  nv={m.nv}  nq={m.nq}')
for i in range(1, m.nbody):
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)
    print(f'    body{i} {name:12s} m={m.body_mass[i]:.4f} '
          f'ipos={m.body_ipos[i].round(4)} diagI={m.body_inertia[i].round(6)}')
