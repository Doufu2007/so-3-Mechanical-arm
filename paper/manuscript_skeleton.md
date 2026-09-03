# 论文骨架（诚实叙事版）

> **!! 已废弃，勿用 —— 请改用 `paper/manuscript.md`。!!**
> 本骨架是 2026-08-25 之前「系统评估」定位的旧稿：题目不含强制要求的「SE(3) 李群」、
> 关键词缺 identifiability、Abstract 要点没有审计主线。现在的定稿定位是**可辨识性审计**，
> 全部内容已并入 `manuscript.md`（题目含 "SE(3) Lie-group"，正文四段式审计）。写论文只从
> `manuscript.md` 取，不要用本骨架。

> 所有 `{…}` 占位符由 `python paper/report.py` 生成的结果填充，禁止手写数字。
> 正文引用 `RESULTS.md` 的编号结论。投稿前把本骨架改成目标期刊的模板。

---

## 题目（候选，避免 "SO(3) 新方法" 假创新）

1. **Physics-informed inverse-dynamics learning for a 7-DoF manipulator: a systematic
   evaluation of inductive bias, input representation, and closed-loop usability**
2. 备选：*Learning manipulator inverse dynamics with physical structure: representation,
   generalization, and gravity-compensation control*

关键词：inverse dynamics; physics-informed neural network; manipulator; gravity
compensation; input representation; directional derivative

## Abstract 要点（一句话一坨，用结论填）

1. 背景：解析逆动力学是模型控制核心，但难标定；纯数据驱动需海量数据、缺物理保证。
2. 方法：双分支物理信息模型（拉格朗日结构 H(q)q̈+c+g + 残差补偿），输入表示三种
   （旋转矩阵/周期/关节角直入），科里奥利项用方向导数（JVP）无矩阵实现。
3. 结果：物理结构在精度/泛化上优于等容量黑箱（{表1}）；输入表示的收益主要来自
   **去掉压缩瓶颈**而非编码本身；JVP 相对显式 Christoffel 快 {4.6×@n=7, 103×@n=30}、
   省 {4–25×} 显存。
4. 关键发现：拟合精度 ≠ 物理可解释性——自适应物理约束会塌缩到 0，此时 R² 最高但
   g(q) 幅度错 {87%}、闭环重力补偿无效；加物理约束能把精度换回物理保真度。
5. 一句话价值：不是又一个 SO(3) 编码，而是一份关于"物理先验在工业臂逆动力学里
   到底值不值、值在哪"的系统工程评估。

## 1. Introduction

- 逆动力学 τ=H(q)q̈+c(q,q̇)+g(q) 之于模型控制（计算力矩、重力补偿、力控）。
- 解析参数难标定（摩擦、柔度、载荷变化）；纯数据驱动 MLP 缺物理保证、泛化差。
- 物理信息方法（DeLaN 等）理论上更好，但 **7 自由度工业臂上没人系统测过**：
  好在哪、值不值、能不能用于控制。这是 gap。
- 三条贡献（见 `P0_positioning.md`，与结果一致）。

## 2. Related work

- 解析动力学 / 回归辨识（Swevers 傅里叶激励）
- 神经网络逆动力学（MLP、HNN、LNN、SymODEN、DeLaN）
- 输入表示（关节角、sin/cos、SO(3)/四元数）
- 本工作的位置：**工程评估 + 消融 + 闭环**，不做新方法。

## 3. Method

- 3.1 问题定义与数据（`Dataset_Franka_Ex` 的生成、`baxter` 实机、`franka` 窄覆盖对照）
- 3.2 双分支模型：拉格朗日物理分支（全程物理单位、H=MMᵀ+εI 严格正定）+
  残差分支；损失 = Huber + λ‖ε‖²，λ 可自适应可固定。
- 3.3 输入表示与**压缩瓶颈**这个被旧实现忽略的混淆因素。
- 3.4 无矩阵科里奥利项（方向导数/JVP），复杂度与能量恒等式。
- 3.5 评测协议：轨迹级三划分、验证集选模、global R² 主指标 + 逐关节。

## 4. Experiments

- 4.1 主对比（`RESULTS.md §1`）
- 4.2 消融（`RESULTS.md §2`）
- 4.3 数据效率 / 噪声鲁棒（`RESULTS.md §5/§6`）
- 4.4 物理正确性核验（`RESULTS.md §4`）
- 4.5 闭环重力补偿（`RESULTS.md §7`）
- 4.6 复杂度（`RESULTS.md §8`）

## 5. Discussion

- **正结果**：物理结构全面优于黑箱（尤其外推/低数据）。
- **负结果（坦白）**：旋转矩阵编码不是卖点；自适应 λ 塌缩；λ=0 时物理量不可信。
- **权衡**：精度 vs 物理保真度/可控性，λ 是旋钮。
- 局限：仿真臂是简化模型、无真机闭环、Baxter 无解析真值、单卡复现。

## 6. Conclusion

诚实的三条结论 + 未来工作（真机验证、λ 的自监督调度、载荷变化下的辨识）。
