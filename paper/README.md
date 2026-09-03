# `paper/` — 论文实验的唯一真相来源

仓库根目录下的 `pinn_delan_master*.py`、`run_unified.py`、`run_data_efficiency.py`、
`run_noise_robustness.py`、`analysis/*` 属于历史实现，**论文里的数字一律不要用它们**：
复核时在那套代码里发现了三处会改变结论的实质错误（见 `pinn_delan_master.py` 顶部说明）。
本目录是重写后的实现，所有表格与图件都由这里的脚本从头产出。

---

## 相对旧实现修正了什么

| # | 问题 | 影响 | 修正 |
|---|---|---|---|
| 1 | `autograd.grad(lagrangian.mean(), q)` | 逐样本标量对自身输入求导应当用 `sum()`。用 `mean()` 把 `g(q)` 与科里奥利第二项整体缩小 batch_size 倍（≈10³），等于删掉重力项；且训练/评估 batch 不同 ⇒ 同一权重给出不同预测 | `paper/models.py` 用 `sum()` |
| 2 | 编码作用在归一化关节角上 | `sin((q−μ)/σ)` 与 `sin(q)` 无关，「真实连杆旋转矩阵」不成立 | 编码前先还原成物理弧度 |
| 3 | `dq`/`ddq` 各自减均值除自身 std | 网络看到的「速度」不是「位置」的时间导数，归一化坐标下拉格朗日结构不成立；逐关节缩放使网络内 `H` 的正定性 ≠ 物理惯性矩阵正定 | 物理分支全程物理单位；黑箱分支单独做 z-score |
| 4 | 用**测试集**上的最好 epoch 选模 | 测试集泄漏，指标系统性偏乐观 | 轨迹级 train/val/test 三划分，**验证集选模，测试集只评一次** |
| 5 | 两个 master 文件编码方式不同，但 `run_unified.py` 两个数据集都只 import 其中一个 | README 声称 Franka 用旋转矩阵编码，实际跑的是 sin/cos | 单一 `JointEncoder`，`none` / `sincos` / `rotmat` 三选一，作为消融因素 |
| 6 | 指标口径不一致（一处全关节平均、一处剔除静态关节） | 同一模型报出不同数字 | 统一同时报 `r2_global`（主指标）/ 全关节平均 / 活跃关节平均 / 逐关节 |

## 数据集

| 名字 | 来源 | 说明 |
|---|---|---|
| `baxter` | 真实 Baxter 左臂实测 | 13 条轨迹，96k 样本；无解析真值 |
| `franka` | MuJoCo `tool/panda.xml` | **原始数据，覆盖度极差**：15 条轨迹全部以 q=0 为中心、\|q̇\|≤0.85 rad/s，在 7 维位形空间里只是 15 条一维曲线。测试点到训练集的最近邻距离是训练点彼此距离的 130 倍 ⇒ 测试集是纯外推。保留作为「窄覆盖」工况 |
| `franka_ex` | 同一 MuJoCo 模型，重新激励 | 有限傅里叶级数激励轨迹（Swevers 式），中心位形随机覆盖关节量程，\|q̇\|≤3 rad/s，动力学项 RMS / 重力项 RMS = 1.26。40/8/8 条轨迹，112k 样本。生成脚本 `make_franka_dataset.py` |

`tool/panda.xml` 是**简化**的 7 轴串联臂（盒/柱几何、Franka 式 z-y-z-y-z-y-z 轴序），
不是真实 Panda 的惯量参数，论文里要照此表述。

## 复现步骤

```bash
PY=/home/claudeuser/so3-env/bin/python      # torch 2.13 + CUDA，见根目录 requirements.txt

# 0) 重新生成激励充分的仿真数据集（可选，仓库里已带）
$PY paper/make_franka_dataset.py

# 1) 主网格：11 个变体 × 3 个臂 × 5 seeds（单卡多进程，--workers 按显存调）
$PY paper/run_grid.py --grid main     --workers 6
$PY paper/run_grid.py --grid data_eff --workers 6      # 数据效率
$PY paper/run_grid.py --grid hparam   --workers 6      # 超参敏感性

# 2) 汇总表格 + 显著性检验
$PY paper/analyze.py

# 3) 物理正确性核验（H 正定性/条件数、能量恒等式、与 MuJoCo 真值对比、λ 分布）
$PY paper/verify_physics.py --all

# 3b) SE(3) 物理参数恢复核验（学到的质量/质心/惯量 vs MuJoCo 真值，se3 独有证据）
$PY paper/se3_param_check.py --pattern 'se3*_seed0.pth'

# 4) 噪声鲁棒性
$PY paper/noise_robustness.py --arm franka_ex
$PY paper/noise_robustness.py --arm baxter

# 5) 闭环控制
$PY paper/control_sim.py --ckpt-full results_v2/full_franka_ex_seed0.pth \
                         --ckpt-mlp  results_v2/mlp_franka_ex_seed0.pth

# 6) 科里奥利项复杂度基准
$PY paper/benchmark_coriolis.py

# 7) 出图
$PY paper/figures.py
```

## 实验规范

* **划分**：轨迹级三划分。同一条轨迹的相邻帧高度相关，若按帧随机划分，验证/测试
  集里会混进训练轨迹的邻帧，指标会明显偏乐观。
* **选模**：验证集 global R²。测试集只在训练结束后用最优权重评一次。
* **随机性**：`seed` 同时控制权重初始化、数据打乱、（数据效率实验里的）子采样。
  每个结果 json 都记录 seed、git commit、GPU 型号、torch 版本、耗时与完整复现命令。
* **主指标 `r2_global`**：把所有关节的残差平方和与总平方和合并统计。不用「逐关节 R²
  取平均」当主指标，是因为 Franka 的 J7 扭矩标准差只有 9e-4 N·m（末端只挂一个小球，
  绕自身轴既无重力矩也几乎无惯量），它的 R² 是纯数值噪声、能到 −10⁴，放进任何
  平均值里都会把整张表变成噪声表。逐关节结果在 `table3_per_joint.md` 里全列。

## 变体登记表（`paper/train.py: VARIANTS`）

| 变体 | 含义 |
|---|---|
| `full` | 完整模型：rotmat 编码 + 物理分支 + 残差分支 + 自适应 λ + 多尺度损失 |
| `enc_sincos` / `enc_none` | 只改编码方式 |
| `no_physics` | 去掉物理分支 ⇒ 纯黑箱 FFNN（架构容量与完整模型的残差分支相同）|
| `no_uncertainty` | 去掉残差分支 ⇒ 纯 DeLaN |
| `no_adaptive_lam` | 自适应 λ 换成固定 λ |
| `no_multiscale` | 多尺度关节加权换成普通 Huber |
| `dual_lam` / `dual_lam_wide` | 自适应 λ 换成对偶上升约束式 λ（不自塌缩）|
| `se3` | **SE(3)/SO(3) 结构化惯量**：物理分支换成 M(q)=Σ JᵀGJ（学 10·n_body 个物理惯量参数），其余同 `dual_lam_wide` |
| `se3_no_uncertainty` | SE(3) 结构单独（无残差），证明结构化惯量本身是正确归纳偏置 |
| `mlp` / `hnn` / `lnn` / `symoden` | 外部基线 |
