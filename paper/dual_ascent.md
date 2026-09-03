# 对偶上升约束式物理权重 —— 机制、命题与证据

> ## ⚠️ 2026-08-27：本文档描述的是**训练装置**，不再作为论文贡献
>
> 论文定位已改为审计（见 `manuscript.md`），dual ascent 在正文里只作为「下面所有实验
> 用的权重调度」出现，§3.2 描述、§5 给负结果证据；两条命题降格为标准约束优化事实，
> 不再编号为 Prop.1/Prop.2。理由有两条：
>
> 1. 「把罚项权重当拉格朗日乘子做投影梯度上升」是标准约束优化做法，不构成方法贡献；
> 2. 数据不支持把它写成卖点 —— 3-seed 闭环批里 `dual_lam_wide` 是 8.38±0.59 mrad，
>    只比朴素 MLP 的 8.68±1.69 好一点点，而 SE(3) 分支是 0.46±0.10 mrad。真正起作用的
>    是惯性矩阵的结构，不是权重调度。
>
> 本文档的**机制说明和 λ 塌缩的负结果证据仍然有效**，正文继续引用。

> 本文档把论文里那条「自适应 λ 塌缩」的负结果，升级成一个**可发表的方法**。
> 定位一句话：**自适应 λ 之所以塌缩，是因为它更新方向错了（梯度下降）；把 λ 从
> 「最小化的变量」改成「对偶变量做投影梯度上升」，物理分支就不会被关掉，且无需手工调 λ。**

---

## 1. 为什么自适应 λ 必然塌缩到 0（一个干净的反向论证）

设双分支模型输出归一化扭矩 $\hat\tau = \tau_{\mathrm{phys}} + \varepsilon$，
其中 $\varepsilon$ 是残差分支（黑箱）输出，$\tau_{\mathrm{phys}}$ 是物理分支输出。
总损失

$$
L(\theta, \phi) = \ell(\hat\tau) + \lambda_\phi \cdot \|\varepsilon\|^2,
\qquad \lambda_\phi = \sigma(g_\phi(\cdot)) \in (0, 0.1),
$$

其中 $\theta$ 是物理分支 + 残差分支参数，$\phi$ 是 λ 网络（`AdaptiveLambda`）的参数。
$\lambda_\phi$ 只出现在正则项里，与数据项 $\ell(\hat\tau)$ 无关（$\varepsilon$ 不依赖 $\phi$）。

**命题 1（塌缩）**：设 $\|\varepsilon\|^2 > 0$（残差分支非平凡激活）。则对任何满足
$\partial\lambda_\phi/\partial\phi \ne 0$ 的 $\phi$，

$$
\frac{\partial L}{\partial\phi} = \|\varepsilon\|^2 \cdot \frac{\partial\lambda_\phi}{\partial\phi}
\ \Rightarrow\ \phi \gets \phi - \eta\,\|\varepsilon\|^2\,\frac{\partial\lambda_\phi}{\partial\phi}
$$

因为 $\lambda_\phi$ 是 sigmoid 有界且 $\partial\lambda_\phi/\partial\phi$ 的方向在梯度下降下
只会把 $\lambda_\phi$ 压向 0（$\lambda$ 单调减 ⇒ $L$ 单调减），所以**梯度下降在 $\lambda_\phi$ 上的
不动点就是 $\lambda_\phi\to 0$**。换成人话：λ 只会给损失「加钱」，把 λ 调小永远能降损失，
优化器没有理由不这么做。这不是实现 bug，是**目标函数结构本身决定的**。

> 这个反向论证解释了旧实现里 `reg = (lam * eps.pow(2).mean(dim=1)).mean()` 为什么
> 在训练中一路塌缩到 $\lambda\approx 2\times10^{-9}$——它不是超参没调好，是必然的。

**代价**：$\lambda\to 0$ 意味着 $\lambda\|\varepsilon\|^2$ 这一项消失，物理分支失去约束，
被容量更大的残差分支「架空」；但物理分支的参数 $\theta_{\mathrm{phys}}$ 仍在求导，
于是学到的 $g(q)$、$H(q)$ 全部失真（实测 $g(q)$ 相对误差 0.67、$H$ 条件数比真值大两个
数量级），闭环重力补偿反而差于纯 MLP。而 $\lambda=0$ **连 R² 都不是最优**（0.9643 vs
$\lambda=1$ 的 0.9735）——即自适应 λ 收敛到一个**严格被支配**的点。

---

## 2. 方法：把 λ 从「罚项权重」改成「对偶变量」

把「物理保真度」从软惩罚改成**硬约束**：

$$
\min_{\theta}\ \ell(\hat\tau)
\quad\text{s.t.}\quad
\|\varepsilon\|^2 \le \delta,
$$

其中 $\delta$ 是残差预算。用 Lagrange 乘子 $\lambda\ge 0$ 松弛：

$$
\mathcal{L}(\theta, \lambda) = \ell(\hat\tau) + \lambda\,(\|\varepsilon\|^2 - \delta).
$$

- **原问题（θ）**：$\theta \gets \theta - \eta_\theta \nabla_\theta\big[\ell + \lambda\|\varepsilon\|^2\big]$
  （$-\lambda\delta$ 是常数，不进 θ 梯度）。
- **对偶问题（λ）**：$\lambda \gets \max(0,\ \lambda + \eta_\lambda\,(\|\varepsilon\|^2 - \delta))$，
  即**投影梯度上升**。

代码对应 `train.py` 里的两行：

```python
loss, parts = compute_loss(pred, taub, comps, cfg, lam_dual=lam_dual)
...
lam_dual = max(0.0, lam_dual + cfg.dual_lr * (parts['residual_sq_mean'] - parts['dual_delta']))
```

### 为什么这样不会塌缩

**命题 2（非塌缩）**：设约束有效（$\|\varepsilon\|^2 > \delta$）。则在投影梯度上升下
$\lambda$ 严格递增，直到 $\|\varepsilon\|^2\approx\delta$ 或 $\lambda$ 满足 KKT 条件。
只要残差分支超过预算，$\lambda$ 就上升去惩罚它——**λ 的方向由「约束违反量」驱动，
而不是由「降低总损失」驱动**，因此不存在 λ→0 的退化不动点。收敛后 $\lambda^\star>0$
当且仅当约束是紧的（active），这正是我们想要的：物理分支被强制承担扭矩，直到残差
被压回预算内。

### δ 的可解释性

$\delta = \delta_{\mathrm{frac}}^2 \cdot \mathrm{mean}(\|\tau\|^2)$（归一化扭矩空间），
所以 $\delta_{\mathrm{frac}}$ 就是「残差分支 RMS 允许占扭矩 RMS 的上限比例」。
- $\delta_{\mathrm{frac}}=0.3$ ⇒ 残差最多解释约 30% 的扭矩 RMS，物理分支必须承担其余大部分；
- $\delta_{\mathrm{frac}}\to\infty$ ⇒ 退化为自适应 λ→0（物理被关掉）；
- $\delta_{\mathrm{frac}}\to 0$ ⇒ 退化为纯 DeLaN（残差被禁）。

它把一个「旋钮」从**不可解释的手工 λ 扫描**变成了**有物理含义的预算**，这是本方法
相对「固定 λ=1」的真正增量：**自调、可解释、且有收敛保证**。

---

## 3. 期望结果与读取方式

训练产出（`experimental_results/v2/dual_lam_*.json` / `dual_lam_wide_*.json`）里多了：

| 字段 | 含义 | 期望 |
|---|---|---|
| `dual_final_lambda` | 收敛后的对偶 λ | **非零**（如 0.5–5），证明未塌缩 |
| `dual_delta` | 实际使用的残差预算 δ | = $\delta_{\mathrm{frac}}^2\cdot\mathrm{mean}(\|\tau\|^2)$ |
| `curve.residual_sq_mean` | 每 epoch 的 $\|\varepsilon\|^2$ | 收敛到 ≈ δ |
| `curve.lambda_mean` | 每 epoch 的 λ | 上升后平稳，**不单调趋零** |

配套证据（用现有脚本，无需改）：

```bash
# 物理核验：g(q) 误差应接近固定 λ=1 的 0.11，而不是自适应 λ→0 的 0.67
python paper/verify_physics.py --ckpt results_v2/dual_lam_franka_ex_seed0.pth
# 闭环：稳态 RMSE 应 ≈ 10 mrad（≈MLP），而不是自适应 λ→0 的 36 mrad
python paper/control_sim.py --ckpt-phys results_v2/dual_lam_franka_ex_seed0.pth \
                            --ckpt-mlp  results_v2/mlp_franka_ex_seed0.pth
# δ 扫描：δ_frac ∈ {0.1,0.2,0.3,0.5} 的 R²–可用性权衡（先跑 run_grid --grid dual）
```

**三个待验证的卖点**（跑完把数字填进 manuscript）：

1. `dual_lam` 的 R² ≥ `full`（自适应塌缩）——即「修法无精度代价，甚至更高」；
2. `dual_lam` 的 `dual_final_lambda` 显著 > 0，而 `full` 的 λ≈2×10⁻⁹——「塌缩 vs 不塌缩」的对照；
3. `dual_lam` 的 g(q) 误差与闭环 RMSE 接近 `no_adaptive_lam`（固定 λ=1），
   但**无需人工扫 λ**——这是对偶上升相对固定 λ 的核心优势。

---

## 4. 一句论文话术（写进 Methods/Results）

> We reformulate the physics weight not as a trainable penalty coefficient — which we prove
> collapses to zero by construction (Prop. 1) — but as the **dual variable of a
> residual-magnitude constraint**, updated by projected gradient ascent. The weight is then
> driven by the constraint violation $\|\varepsilon\|^2-\delta$ rather than by total-loss
> descent, so it cannot collapse while the physics branch is under-utilised (Prop. 2), and it
> converges to a positive, interpretable value without any manual $\lambda$ search.
