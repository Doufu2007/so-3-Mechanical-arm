# SO(3)/SE(3) 李群机械臂 —— 基座姿态等变逆动力学（工程类版）

> **一句话卖点**：黑箱模型把重力学进权重，换安装姿态就得重训；我们的结构化模型把重力方向做成显式输入 `g(q; g_dir) = g_dirᵀ·p(q)`，换姿态**零重训**。

---

## 这是什么

工业机械臂逆动力学 `τ = M(q)q̈ + c(q,q̇) + g(q)` 是模型控制的核心，但解析参数难标定、纯数据驱动缺物理保证。更关键的一个工程痛点：**一旦机械臂改变安装姿态**（地面 → 墙面 / 倒挂 / 倾斜、或装到移动底盘），重力项 `g(q)` 随基座方向改变，原有模型失效，必须重新辨识。

本工作的增量（**应用级，不是范式级**）：把"基座旋转等变"做成学习逆动力学模型的**构造性约束** —— 重力项写成对重力方向的显式线性 `g(q; g_dir) = g_dirᵀ·p(q)`，其中 `p(q) ∈ R³` 是学的"势能方向"。于是换安装姿态 = 换 `g_dir`，模型自动正确，**等变是构造出来的，不是学出来的**。

---

## 核心结果（四组生死门对照，全局 R²，越高越好）

| 模型 | 训练姿态 | @0° | @30° | @60°（留出） |
|---|---|---|---|---|
| **结构模型**（g 线性于 g_dir） | 仅 0° | **0.9997** | **0.9997** | **≈0.9997** |
| 黑箱 MLP（喂 g_dir 输入） | 仅 0° | 0.9234 | 0.7879 | — |
| 黑箱 MLP（喂 g_dir 输入） | 0/15/30/45 | 0.9287 | — | 0.8983 |

- **① 结构模型**换姿态 ΔR² ≈ 0 —— 单姿态训练、全姿态零重训，构造性成立；
- **③ 黑箱**即使拿到同样的 `g_dir` 输入，换姿态崩（0.92 → 0.79）—— 证明"线性结构约束"值钱，不是"喂个输入谁都会"；
- **④ 黑箱**给 4 倍姿态数据，留出姿态仍只有 0.90 < 结构 1 姿态的 0.9997 —— **数据效率 + 零重训**双重优势。

判据与完整实验设计见 `选项B_立项任务清单.md` §1；逐关节分析见 `experimental_results/equivar/`。

---

## 定位与诚实边界

- **novelty 是应用级**：落在"等变 × 逆动力学 × 串链机械臂 × 基座姿态"这个四轴交点，不是新范式（"SE(3) 等变动力学学习"是 Duong & Atanasov 一行）。
- **"重力线性于重力方向"是教科书**（牛顿-欧拉 / RNE 常识）；本工作的增量在于**把它做成学习模型的结构约束 + 零重训的工程验证 + 黑箱对照**，不吹"发现等变"。
- 查重与差异化见 `选项B_基座姿态等变_查重与定位.md`。

---

## 复现

```bash
# 1) 生成水平 + 倾斜数据集（MuJoCo 仿真臂，有限傅里叶激励轨迹）
python paper/make_franka_dataset.py
python paper/make_franka_dataset.py --out Dataset_Franka_Ex_Tilt30 --tilt-deg 30 --tilt-axis x

# 2) 四组生死门（①②③，核心 go/no-go）
python paper/equivar_experiment.py --tilts 30 --axis x

# 3) 多姿态黑箱（组④，审稿人必问）
python paper/equivar_experiment.py --train-tilts 15,30,45 --test-tilts 60 --axis x

# 4) 闭环跨姿态（把 R² 升级成"换姿态后控制器还能不能用"）
python paper/control_crosspose.py --axis x --tilts 30,60,90 --epochs 20
```

训练需 CUDA GPU；数据集、模型权重（.pth/.npz/.csv）均被 `.gitignore` 排除、可重新生成。

---

## 项目结构

```
paper/                           # 当前规范代码（本包唯一权威）
  equivar_experiment.py          # 四组生死门（①②③④）—— 核心实验
  control_crosspose.py           # 闭环跨姿态验证（重力补偿定点调节）
  make_franka_dataset.py         # 有限傅里叶激励轨迹数据生成（--tilt-deg/--tilt-axis）
  models.py                      # 模型定义（gravity_input 黑箱对照 / SE(3) 物理分支）
  se3_physics.py                 # SE(3) 结构化物理分支（含 mj_fullM 强自检）
  core.py                        # 归一化 / 指标 / 重力旋转 / 数据加载
  train.py                       # 训练循环 / 种子 / 指标
  baseline_ls_exact.py           # 经典最小二乘辨识天花板（零空间分解）
  baseline_swevers.py            # 【已废弃】勿用，仅留作记录
  *.py                           # 其余：核验 / 出图 / 可辨识性 / 摩擦 / 噪声 等
tool/                            # MuJoCo XML（panda.xml / panda_real.xml / baxter_left.xml）
experimental_results/
  equivar/                       # 选项B 生死门 + 多姿态结果 json
  v2/ v2_tables/ v2_figures/ ... # 旧审计线结果
选项B_立项任务清单.md             # 立项：生死门判据 / 方法 / 实验矩阵 / 真机故事 / 风险
选项B_基座姿态等变_查重与定位.md   # 逐项查重 + 500 字定位
```

> 仓库里 `docs/README.md` 与 `RESULTS.md` 是**上一阶段（SE(3) 可辨识性审计）**的文档，已降级为背景/动机，不作为当前工程类版卖点。

---

## 环境

Python 3.10+ / PyTorch 2.0+（训练需 CUDA GPU，CPU 仅能跑分析脚本）；MuJoCo（仿真数据与控制闭环）。

## 许可证

MIT
