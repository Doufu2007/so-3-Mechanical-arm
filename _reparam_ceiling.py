"""诊断：重参数化的「理论上限」——训练收敛后 π̂=B·θ̂=B·Bᵀ·π_true=π_id。

本脚本不依赖训练结果，直接算 π_id 的逐连杆 (m,c,I) 误差，从而预测
reparam_param_check.py 训练后报告的数字。若误差大，说明「重参数化能学回
逐连杆物理参数」的说法不成立，只能学回 base 参数（可辨识组合）。
"""
import numpy as np
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / 'paper'))
from se3_physics import extract_geometry, skew
from identifiability import sym_from_vec, SYM_IDX

xml = 'tool/panda.xml'
geom = extract_geometry(xml)
d = np.load('base_basis.npz')
pi_true = d['pi_true']            # (80,)
pi_id = d['pi_identifiable']      # (80,) = B·Bᵀ·π_true
n_body = len(geom['mass'])

def to_mcI(pi):
    pi = pi.reshape(n_body, 10)
    m = pi[:, 0]
    h = pi[:, 1:4]
    I_o = sym_from_vec(torch.tensor(pi[:, 4:10], dtype=torch.float64))
    c = h / np.maximum(m, 1e-9)[:, None]
    cx = skew(torch.tensor(c))
    I_c = I_o - torch.tensor(m)[:, None, None] * (cx @ cx.transpose(-1, -2))
    return m, c, I_c.detach().numpy()

m_t, c_t, I_t = to_mcI(pi_true)
m_i, c_i, I_i = to_mcI(pi_id)

mass_t = np.asarray(geom['mass'])
com_t = np.asarray(geom['com'])
I_t_g = np.asarray(geom['I'])

print(f'n_body={n_body}  K=80  rank={d["rank"]}')
print(f'{"body":<8} {"m_true":>8} {"m_id":>8} {"m_rel":>9} {"com_true":>8} {"com_id":>8} {"com_err":>9} {"I_rel":>9}')
rows = []
for k in range(n_body):
    m_rel = abs(m_i[k] - mass_t[k]) / max(abs(mass_t[k]), 1e-6)
    c_err = float(np.linalg.norm(c_i[k] - com_t[k]))
    i_rel = float(np.linalg.norm(I_i[k] - I_t_g[k]) / max(np.linalg.norm(I_t_g[k]), 1e-12))
    rows.append((m_rel, c_err, i_rel))
    print(f'body{k:<4} {mass_t[k]:>8.3f} {m_i[k]:>8.3f} {m_rel:>9.4f} '
          f'{np.linalg.norm(com_t[k]):>8.4f} {np.linalg.norm(c_i[k]):>8.4f} {c_err:>9.4f} {i_rel:>9.4f}')

# link1-link6 口径（固定基座 body0 + 末端 hand 排除）
sel = rows[1:7]
print(f'\nlink1-link6 口径（正文 §4.2 用的）:')
print(f'  mass_rel_err_mean = {np.mean([r[0] for r in sel]):.4f}')
print(f'  com_err_m_mean    = {np.mean([r[1] for r in sel]):.4f}')
print(f'  I_rel_err_mean    = {np.mean([r[2] for r in sel]):.4f}')
# 全 8 body 口径（reparam_param_check 的 check_params 默认）
print(f'全 body 口径（check_params 默认）:')
print(f'  mass_rel_err_mean = {np.mean([r[0] for r in rows]):.4f}')
print(f'  com_err_m_mean    = {np.mean([r[1] for r in rows]):.4f}')
print(f'  I_rel_err_mean    = {np.mean([r[2] for r in rows]):.4f}')
