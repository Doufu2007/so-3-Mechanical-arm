"""
SE(3) 物理参数恢复核验 —— 学到的质量/质心/惯量 vs MuJoCo 真值。

这是 se3 变体**独有**的证据：DeLaN 学的是任意 H(q) 矩阵（无法对比真值物理量），
而 se3 学的就是 10·n_body 个物理惯量参数（质量 m、质心 c、质心系惯量张量 I，
SPD 由 Cholesky 保证），可以逐项和 MuJoCo 真值对比。本脚本做这个对比，
产出论文「参数可恢复性 / 可解释性」的核心证据。

诚实性说明：不是所有参数都能被训练数据唯一确定（可辨识性问题）。末端 hand 等
低激励连杆的误差可能偏大——这是可辨识性，不是实现 bug，报告时如实呈现、不掩盖。
（Swevers 基线 `baseline_swevers.py` 已给出干净数据下的可辨识性天花板作对照。）

    python paper/se3_param_check.py --ckpt results_v2/se3_franka_ex_seed0.pth
    python paper/se3_param_check.py --pattern 'se3*_seed0.pth'     # 全部 se3 变体
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paper import core
from paper.core import ARMS, Normalizer, load_arm, provenance, save_json
from paper.models import ModelCfg, build_model
from se3_physics import extract_geometry

ROOT = core.ROOT
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_se3(path):
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    cfg = ModelCfg(**ck['cfg'])
    if cfg.physics_kind != 'se3':
        raise ValueError(f'{path} 不是 se3 变体（physics_kind={cfg.physics_kind}）')
    arm = ck['arm']
    data = load_arm(arm)
    norm = Normalizer(*data['train'])
    model = build_model(cfg, 7, norm, ARMS[arm].axes).to(DEVICE)
    model.load_state_dict(ck['state_dict'])
    model.eval()
    return model, cfg, arm, ck


def _body_names(xml, body_ids):
    """连杆名。新版 mujoco 绑定没有 MjModel.body_id2name，须用 mj_id2name。"""
    import mujoco
    m = mujoco.MjModel.from_xml_path(xml)
    out = []
    for b in body_ids:
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(b))
        out.append(nm if nm else f'body{int(b)}')   # 未命名 body 返回 None
    return out


def check_params(model, cfg):
    """逐连杆对比学到的 (m, c, I) 与 MuJoCo 真值，返回 per_link + summary。"""
    xml = cfg.se3_xml or 'tool/panda.xml'   # 相对路径：MuJoCo 绝对中文路径会 ValueError
    geom = extract_geometry(xml)

    with torch.no_grad():
        mass, com, I = model.phys.inertial_params()
        mass = mass.detach().cpu().numpy().astype(np.float64)
        com = com.detach().cpu().numpy().astype(np.float64)
        I = I.detach().cpu().numpy().astype(np.float64)

    mass_t = np.asarray(geom['mass'], dtype=np.float64)
    com_t = np.asarray(geom['com'], dtype=np.float64)
    I_t = np.asarray(geom['I'], dtype=np.float64)
    names = _body_names(xml, geom['body_ids'])

    per = []
    for k in range(len(names)):
        m_rel = abs(mass[k] - mass_t[k]) / max(abs(mass_t[k]), 1e-6)
        c_abs = float(np.linalg.norm(com[k] - com_t[k]))
        i_rel = float(np.linalg.norm(I[k] - I_t[k]) / max(np.linalg.norm(I_t[k]), 1e-12))
        per.append({'body': names[k],
                    'mass_true': float(mass_t[k]), 'mass_learned': float(mass[k]),
                    'mass_rel_err': float(m_rel), 'com_err_m': c_abs, 'I_rel_err': i_rel})

    summary = {
        'mass_rel_err_mean': float(np.mean([p['mass_rel_err'] for p in per])),
        'com_err_m_mean': float(np.mean([p['com_err_m'] for p in per])),
        'I_rel_err_mean': float(np.mean([p['I_rel_err'] for p in per])),
        'mass_rel_err_median': float(np.median([p['mass_rel_err'] for p in per])),
    }
    return {'per_link': per, 'summary': summary, 'xml': xml}


def summarize(result, variant, arm, seed):
    s = result['summary']
    print(f"\n=== se3 参数恢复 / {variant} / {arm} seed{seed} ===")
    print(f"  整体: 质量相对误差均值={s['mass_rel_err_mean']:.4f}  "
          f"质心误差均值={s['com_err_m_mean']:.4f} m  "
          f"惯量 Frobenius 相对误差均值={s['I_rel_err_mean']:.4f}")
    print(f"  {'body':<12} {'m_true':>8} {'m_learned':>8} {'m_rel':>8} {'com_m':>8} {'I_rel':>8}")
    for p in result['per_link']:
        print(f"  {p['body']:<12} {p['mass_true']:>8.3f} {p['mass_learned']:>8.3f} "
              f"{p['mass_rel_err']:>8.4f} {p['com_err_m']:>8.4f} {p['I_rel_err']:>8.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--pattern', default='se3*_seed0.pth')
    a = ap.parse_args()
    paths = ([Path(a.ckpt)] if a.ckpt else
             sorted((ROOT / 'results_v2').glob(a.pattern)))
    if not paths:
        print(f'[se3_param_check] 无匹配 checkpoint: {a.pattern}（先跑 run_grid 训练 se3）')
        return
    for p in paths:
        try:
            model, cfg, arm, ck = load_se3(p)
            seed = ck.get('seed', 0)
            variant = ck.get('variant', 'se3')
            res = check_params(model, cfg)
            summarize(res, variant, arm, seed)
            out = {'checkpoint': str(p), 'variant': variant, 'arm': arm, 'seed': seed,
                   'summary': res['summary'], 'per_link': res['per_link'],
                   'provenance': provenance()}
            # 输出文件名带 variant：--pattern 'se3*_seed0.pth' 会同时匹配 se3 与
            # se3_no_uncertainty，不带 variant 会互相覆盖、只留最后一个。
            save_json(ROOT / 'experimental_results' / 'v2_physics' /
                      f'se3_param_{variant}_{arm}_seed{seed}.json', out)
        except Exception as ex:
            import traceback
            print(f'[FAIL] {p}: {ex}')
            traceback.print_exc()


if __name__ == '__main__':
    main()
