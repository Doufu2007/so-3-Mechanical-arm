# Physics-informed inverse dynamics for a 7-DoF manipulator: SE(3)-structured inertia

基于物理信息神经网络（PINN）的工业机械臂逆动力学建模，核心落点是 **SE(3) 李群结构放在惯性矩阵里**，不是放在输入编码里。

---

## 一句话定位

**精确逆动力学是模型控制的核心，但解析参数难标定、纯数据驱动缺物理保证。**
本工作回答四个问题：李群（SO(3)/SE(3）该放在哪？物理结构值不值？哪种输入表示好？学到的模型能不能用来控？
**核心答案：李群该放在惯性矩阵的结构里（M(q)=ΣJᵀGJ），不是输入编码里。**

---

## 核心贡献（2026-08-27 定位改为**审计**，以 `paper/manuscript.md` 为准）

> SE(3) 结构本身**不是本文贡献**：M(q)=ΣJᵀGJ 的李群形式是 Park et al. IJRR 1995
> （`park1995lie`），10 参数可微 Newton–Euler 参数化是 Sutanto et al. L4DC 2020
> （`sutanto2020encoding`），与本文做法逐字重合。本文问的是那篇没回答的问题：
> **这类模型到底能不能学回物理参数？** 答案是不能，而且这不妨碍它好用。

1. **可辨识性上界**：构造精确惯性回归量（线性性自检 6.5e-16，与 `mj_inverse` 对齐 1.3e-06），
   真实 Franka Panda 的 70 个惯性参数里**只有 43 个方向可辨识**，27 维在任何轨迹上都不可观测。
   换一个质量差 10 倍、惯量各向同性的 7 轴臂，**秩仍是 43** —— 秩只由运动学结构决定。
2. **拟合精度 ≠ 物理保真度**：2 变体 × 2 臂 × 5 seed = 20 次运行全部 global R²=1.0000，
   而质量误差 0.244–0.362、质心 8.0–9.6 cm、惯量 Frobenius 0.850–5.738（link1–link6 口径）。
   **无偏最小二乘撞同一堵墙**：力矩 R²=1.00000000，100.0000% 参数误差落在零空间
   （‖e_id‖=4.7e-15）—— 不是训练问题，是问题的结构性质。
3. **解释闭环为何仍然成功**：重力子空间 G（12 维）⊆ 动力学可辨识子空间 D（43 维），
   残差 1.60e-15、最大主角度 2.09e-06 度。所以 g(q) 完全由可辨识部分决定。含摩擦+噪声训练的
   SE(3) 分支闭环稳态 **0.46±0.10 mrad**，比最强基线 8.38±0.59 好 18 倍，尽管参数是错的。
4. **全局 R² 是不安全的指标**：加摩擦后纯结构分支全局 R² 只从 1.0000 掉到 0.9871，
   而 J5 掉到 0.320、J7 掉到 **−0.008**；`dual_lam_wide` 在**干净数据**上全局 0.9979 而 J7 = **−4.76**。
   建议报告规范：逐关节 R² + 回归量秩 + 明确的连杆口径 + 闭环验证。

仍然成立的旧结论（作为背景而非卖点）：李群价值在动力学结构不在输入编码；物理结构价值有条件
（窄覆盖下不如黑箱）；自适应 λ 结构上必塌缩、是被支配点（对偶上升作为训练装置，不作为贡献）；
无矩阵科里奥利项用 JVP，n=7 快 4.6×、n=30 快 103×，能量恒等式保持到 ~1e-5。

---

## 核心结果（5-seed 均值，来自 `RESULTS.md`，勿手改）

### 主对比 — 7 轴仿真臂（激励充分 Dataset_Franka_Ex，R²(global)）

| 方法 | R²(global) | RMSE (N·m) |
|---|---:|---:|
| **SE(3) 结构化惯量 M=ΣJᵀGJ（核心）** | **1.0000** | **0.0012** |
| SE(3) 结构单独（无残差，80 参数） | 1.0000 | 0.0012 |
| 对偶上升 λ（sin/cos、无瓶颈） | 0.9979 | 0.4163 |
| 纯 DeLaN（无残差） | 0.9746 | 1.4465 |
| 去掉物理分支（纯黑箱 FFNN） | 0.9187 | 2.5899 |
| LNN | 0.8988 | 2.8889 |
| MLP | 0.8768 | 3.1867 |
| SymODEN | 0.7559 | 4.4805 |
| HNN | 0.5067 | 6.3786 |

外推（窄覆盖 Dataset_Franka）：SE(3) 结构 R²(global)=0.9999，同样碾压全部基线。

### 诚实负结果（论文正文同款口径）

- **参数不可辨识**：SE(3) 分支把扭矩复现到机器精度（franka_ex 全关节 R²>0.999999），但 80 个物理参数仍偏离真值。
  激励充分的 franka_ex 上 link1–link6 口径质量相对误差 ≈0.27、质心误差 ≈0.09–0.10 m、惯量相对误差 ≈0.85；
  窄覆盖的 franka 上惯量误差 >1（≈5.7，比取零还远离真值），质量 ≈0.36。这是可辨识性极限
  （多组 (m,c,I) 给出同一 M(q)），不是结构 bug。**结构是正确归纳偏置（对 map），参数级可解释性仍受可辨识性限制。**
- **真机无优势**：Baxter（真实臂）上物理结构 vs 黑箱无显著差异（+0.0014，不显著），SE(3) 分支未在真机评测。
  物理结构的价值在富数据下的精度/数据效率，**不买外推**。
- **物理权重的价值有条件**：自适应 λ 会塌缩到 0（严格被支配点），固定 λ 或对偶上升才能无代价换回控制级 g(q)。

---

## 复现

见 `paper/README.md` 与 `MANIFEST.md`。核心：

```bash
python paper/se3_physics.py --xml tool/panda.xml        # SE(3) 几何/数学强自检（训练前先跑）
python paper/run_grid.py --grid main --workers 6
python paper/se3_param_check.py --pattern 'se3*_seed*.pth'   # SE(3) 物理参数 vs 真值
python paper/analyze.py && python paper/report.py && python paper/figures.py
```

所有数字由 `python paper/report.py` 从 `experimental_results/` 的 json 汇编进 `RESULTS.md`，禁止手写。

---

## 项目结构

```
paper/                          # 当前规范代码（本包唯一权威）
  se3_physics.py                # SE(3) 结构化惯量 M=ΣJᵀGJ（核心，含 mj_fullM 强自检）
  se3_param_check.py            # SE(3) 物理参数恢复核验（质量/质心/惯量 vs 真值）
  identifiability.py            # 精确惯性回归量 + 数值秩（--static 只激励重力）
  baseline_ls_exact.py          # 无偏最小二乘天花板；误差按可辨识/零空间分解
  gravity_subspace.py           # 重力子空间 ⊆ 可辨识子空间（主角度检验）
  fig_identifiability.py        # Fig.10–12
  strip_meshes.py               # 生成 tool/panda_real.xml（真实 Franka Panda）
  baseline_swevers.py           # 【已废弃】从未跑通（body_inertia 只有 3 列），
                                #   且其结论被 baseline_ls_exact.py 证伪，仅留作记录
  dual_ascent.md                # λ 塌缩证明 + 对偶上升方法（训练装置，非贡献）
  manuscript.md                 # 完整英文草稿
  P0_positioning.md             # 定位 + 目标期刊
  *.py                          # 训练/分析/出图/物理核验/闭环/λ权衡
experimental_results/
  v2/                           # 主网格训练结果 json（每 seed 一份）
  v2_tables/                    # 主对比表 + 显著性 + 逐关节 + λ权衡 + 科里奥利复杂度
  v2_figures/                   # 12 张图（png + pdf），Fig.10–12 为可辨识性审计
  identifiability/              # 可辨识性审计结果（秩 43/70、最小二乘天花板、G⊆D）
  v2_physics/ v2_control/ v2_noise/ v2_data_eff/ v2_hparam/
RESULTS.md                      # 事实底稿（§0–§10），所有数字的单一来源
MANIFEST.md                     # 交付清单
```

## 环境

Python 3.10+ / PyTorch 2.0+（训练需 CUDA GPU，CPU 仅能跑分析脚本）；MuJoCo（仿真数据与控制闭环）。

## 许可证

MIT
