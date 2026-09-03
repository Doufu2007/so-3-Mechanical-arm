# 交付清单 —— 机械臂物理信息神经网络论文

本包是论文的全部可交付产物（约 18MB）。所有数字来自自动生成的 `RESULTS.md`，禁止手写。

## 目录
- `RESULTS.md`          —— 事实底稿（§0–§10），所有表格数字的单一来源
- `paper/manuscript.md` —— 完整英文草稿（占位符已填好，正文引用 Fig.1–12 / Table 1–5）
- `paper/RESULTS_MAP.md` —— 结果→图/表对照表（写论文队友的交接单，含三个口径注意事项）
- `paper/P0_positioning.md` —— 定位、四条贡献、目标期刊
- `paper/dual_ascent.md` —— λ 塌缩证明 + 对偶上升方法（本次核心增量）
- `paper/se3_physics.py` —— **SE(3)/SO(3) 结构化惯量 M(q)=ΣJᵀGJ**（本次核心，含 mj_fullM 强自检）
- `paper/se3_param_check.py` —— **SE(3) 物理参数恢复核验**（学到的质量/质心/惯量 vs 真值，se3 独有证据）
- `paper/identifiability.py` —— **精确惯性回归量 + 数值秩**（--static 只激励重力）
- `paper/baseline_ls_exact.py` —— **无偏最小二乘天花板**，误差按可辨识/零空间分解
- `paper/gravity_subspace.py` —— **重力子空间 ⊆ 可辨识子空间**（主角度检验）
- `paper/fig_identifiability.py` —— Fig.10–12
- `paper/strip_meshes.py` —— 从 MuJoCo Menagerie 生成 `tool/panda_real.xml`（真实 Franka Panda）
- `paper/baseline_swevers.py` —— **已废弃**（MuJoCo body_inertia 只有 3 列，必然 IndexError，
  从未产出过结果），保留仅为记录手稿旧版那句错误断言的来源
- `paper/*.py`          —— 全部实验脚本（训练/分析/出图/物理核验/闭环/λ权衡）
- `experimental_results/v2/`          —— 主网格训练结果 json（每 seed 一份）
- `experimental_results/v2_tables/`   —— 主对比表 + 显著性(stats.json) + 逐关节 + λ权衡表 + 科里奥利复杂度表
- `experimental_results/v2_figures/`  —— 12 张图（png + pdf），Fig.10–12 为可辨识性审计
- `experimental_results/identifiability/` —— **可辨识性审计结果**（秩 43/70、最小二乘天花板、G⊆D）
- `experimental_results/v2_physics/`  —— 物理正确性核验（H 正定/条件数/能量恒等式/g(q)误差）
- `experimental_results/v2_control/`  —— 闭环重力补偿（含 λ 扫描 control_lam*.json）
- `experimental_results/v2_noise/`    —— 噪声鲁棒
- `experimental_results/v2_data_eff/` —— 数据效率（frac 5%–50%）
- `experimental_results/v2_hparam/`   —— λ 扫描 + 超参敏感性（width/batch）
- `docs/README.md`      —— 项目诚实版 README

## 复现
见 `docs/README.md` 与 `paper/README.md`。核心：
  python paper/se3_physics.py --xml tool/panda.xml   # SE(3) 几何/数学强自检（训练前先跑）
  python paper/run_grid.py --grid main --workers 6
  python paper/se3_param_check.py --pattern 'se3*_seed0.pth'   # SE(3) 物理参数 vs 真值
  python paper/analyze.py && python paper/report.py && python paper/figures.py
  # 可辨识性审计（真实 Panda；须在 ASCII 路径下跑，中文路径 MuJoCo 直接 ValueError）
  python paper/identifiability.py   --xml tool/panda_real.xml --n 200
  python paper/baseline_ls_exact.py --xml tool/panda_real.xml --n 1500
  python paper/gravity_subspace.py  --xml tool/panda_real.xml --n 200
  python paper/fig_identifiability.py

## 未包含（体积大，需要时另取）
- `results_v2/*.pth` —— 304 个 checkpoint（4.8G），只在想直接加载模型不重训时有用；
  论文所有数字都在上面的 json 里，不依赖 checkpoint。
- `Dataset*/`        —— 原始数据（85M），franka_ex 可用 paper/make_franka_dataset.py 重生成。
