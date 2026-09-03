# 结果 → 图/表 对照表（给写论文队友的交接单）

> 所有数字唯一真相源：根目录 `RESULTS.md`（`python paper/report.py` 自动生成，勿手改）。
> 本文件只做「正文引用 ↔ 图/表文件 ↔ RESULTS §N」的映射，方便写论文时取数。

## 图（`experimental_results/v2_figures/`，png + pdf 各一份）

| Fig | 文件名 | 内容 | 对应 RESULTS |
|---|---|---|---|
| Fig. 1 | `fig1_main_comparison` | 主对比（三臂 × 各方法的 R²(global)） | §1 |
| Fig. 2 | `fig2_per_joint` | 逐关节 R² | §3 |
| Fig. 3 | `fig3_convergence` | 训练收敛曲线 | — |
| Fig. 4 | `fig4_data_efficiency` | 数据效率（5%–100% 训练数据） | §5 |
| Fig. 5 | `fig5_noise_robustness` | 噪声鲁棒（σ=0–30%） | §6 |
| Fig. 6 | `fig6_closed_loop_control` | 闭环重力补偿（PD/MLP/学到的 g(q)/真值 g(q)） | §7 |
| Fig. 7 | `fig7_physics_checks` | 物理核验（H 正定/条件数/能量恒等式/JVP 一致性） | §4 |
| Fig. 8 | `fig8_coverage` | 位形空间覆盖（franka_ex vs franka） | §0 |
| Fig. 9 | `fig9_lambda_tradeoff` | λ 扫描（R² / ‖Δg‖/‖g‖ / 闭环 RMSE vs λ） | §9 |
| Fig. 10 | `fig10_identifiability_spectrum` | 惯性回归量奇异值谱：同 zyzyzyz 两臂同秩 43/70；zyxyxyx (Baxter) 三层谱 43 强+18 弱+9 零；完整 vs 静态激励 | 可辨识性 |
| Fig. 11 | `fig11_ls_nullspace_error` | 最小二乘 R²=1 而 100% 参数误差落在零空间；逐参数绝对误差；**第三面板：噪声扫描（e_id 随噪声线性×5.00、e_null 七位不变）** | 可辨识性 |
| Fig. 12 | `fig12_gravity_subspace` | 重力子空间 G ⊆ 动力学可辨识子空间 D（主角度 + 包含残差） | 可辨识性 |

Fig. 10–12 由 `python paper/fig_identifiability.py` 生成，数据源是
`experimental_results/identifiability/*.json`（**不**在 `RESULTS.md` 里，见下表）。

## 表（`experimental_results/v2_tables/`）

| Table | 文件名 | 内容 | 对应 RESULTS |
|---|---|---|---|
| Table 1 | `table1_main.md` | 主对比（R² global/全关节/活跃关节、RMSE、参数量、训练时长） | §1 |
| Table 2 | `table2_ablation.md` | 消融 + 显著性（Welch t / Holm / Cohen d） | §2 |
| Table 3 | `table3_per_joint.md` | 逐关节 R²（含 τ 标准差） | §3 |
| Table 4 | `coriolis_benchmark.md` | 科里奥利复杂度（explicit / reverse×n / JVP） | §8 |
| Table 5 | `lam_tradeoff.md` | λ 权衡（R² / ‖Δg‖/‖g‖ / 闭环 RMSE） | §9 |

## 其余结果位置（正文若需成表，从这里取数）

| 内容 | 数据位置 | RESULTS |
|---|---|---|
| 物理核验表（H 对称/正定/条件数/能量恒等式；vs MuJoCo 真值；自适应 λ 分布；分支贡献） | `v2_physics/*.json` | §4 |
| 闭环控制表（4 控制器 × 8 姿态稳态 RMSE） | `v2_control/control.json`（`summary` 字段） | §7 |
| 数据效率（三臂 × 三方法 × 5 档 frac） | `v2_data_eff/*.json` | §5 |
| 噪声鲁棒（两臂 × 5 方法 × 7 档 σ） | `v2_noise/*.json` | §6 |
| 超参敏感性（width/batch） | `v2_hparam/*.json` | §10 |
| **可辨识秩 43/70、奇异值断崖、逐连杆零空间能量** | `identifiability/identifiability_panda_real.json`（真实 Panda）、`identifiability_panda.json`（8 连杆对照） | 无（新增） |
| **Baxter 左臂 (zyxyxyx) 三层谱：43 强 + 18 弱 + 9 精确零** | `identifiability/identifiability_baxter_left.json`（robosuite 真臂，`tool/baxter_left.xml`） | 无（新增） |
| **静态/重力秩 12** | `identifiability/ident_static_panda_real.json`、`ident_static_panda.json` | 无（新增） |
| **最小二乘天花板：R²=1、‖e_null‖/‖e‖=100%** | `identifiability/ls_exact_real.json` | 无（新增） |
| **G ⊆ D 包含关系与主角度** | `identifiability/gravity_subspace_real.json` | 无（新增） |

### 可辨识性结果的复现命令（需 ASCII 路径，见下节「踩过的坑」）

```bash
python paper/identifiability.py   --xml tool/panda_real.xml --n 200  --json out/identifiability_panda_real.json
# Baxter 左臂（zyxyxyx，第一个真正不同的运动学结构）。模型由 robosuite 的 robot.xml 提取：
python paper/make_baxter_left.py  --src /path/to/robosuite/baxter/robot.xml   # 生成 tool/baxter_left.xml
python paper/identifiability.py   --xml tool/baxter_left.xml --n 200 --json experimental_results/identifiability/identifiability_baxter_left.json
python paper/identifiability.py   --xml tool/panda_real.xml --n 200 --static --json out/ident_static_panda_real.json
python paper/baseline_ls_exact.py --xml tool/panda_real.xml --n 1500 --json out/ls_exact_real.json
# 两个稳健性对照（§4.2 / Limitations）：
python paper/baseline_ls_exact.py --xml tool/panda_real.xml --arm franka_ex --n 4000 --json out/ls_exact_real_trajdata.json   # 用真实训练轨迹，排除「数据比随机采样差」
python paper/baseline_ls_exact.py --xml tool/panda_real.xml --n 1500 --noise 0.01 --json out/ls_exact_real_noise0.01.json       # e_id 随噪声增长
python paper/baseline_ls_exact.py --xml tool/panda_real.xml --n 1500 --noise 0.05 --json out/ls_exact_real_noise0.05.json       # e_null 纹丝不动
python paper/gravity_subspace.py  --xml tool/panda_real.xml --n 200  --json out/gravity_subspace_real.json
python paper/fig_identifiability.py
# 重参数化（工程类主线）：
python paper/base_reparam.py --xml tool/panda.xml --n 1500 --out base_basis.npz          # 生成可辨识子空间基 B
python paper/train.py --arm franka_ex --variant se3_reparam_no_uncertainty --seed 0      # 训练 θ
python paper/reparam_param_check.py --selfcheck                                         # B/几何自检（M err 8e-7）
python paper/reparam_param_check.py --ckpt results_v2/se3_reparam_no_uncertainty_franka_ex_seed0.pth   # θ 恢复报告
```

`tool/panda_real.xml` 由 `paper/strip_meshes.py` 从 MuJoCo Menagerie 的 `franka_emika_panda`
剥离网格资源生成。**必须去掉 joint range/limited**：Panda 的 joint4 范围 `[-3.072, -0.070]`
连 q=0 都在界外，而分析按 |q|≤1.4 采样，限位约束力（可达 1.5e3 N·m）会混进 `mj_inverse`，
让线性性自检直接失败（曾见 8011 N·m 的误差）。零质量叶子 body 也要删，否则白送 10 维零空间。

## 写论文前必须核对的口径（重要）

0. **参数误差必须用 link1–link6 口径**：固定基座对 τ 无贡献、末端 hand 的 m_true=0.1 kg
   让相对误差分母失去意义。全 8-body 口径把质量误差从 **24–36% 抬到 72–92%**，是算术假象，
   手稿旧版误用过。两种口径都写进了 json，正文必须显式声明用的哪种。
   另：`v2_physics/` 只用主目录 20 份；`_backup_zero_init/` 那 20 份的 `I_rel_err_mean`
   全等于常数 0.9978611142（08-25 修复前的 bug 版），**不可用**。

0b. **闭环有两个批次，跨批次数字不可比**：
   - **3-seed 批** `v2_control/control_*_seed*.json` —— 正文所有主张都用这批，sd 用 ddof=1。
     PD 40.99 / MLP 8.68±1.69 / dual_lam_wide 8.38±0.59 / **se3-on-franka_fn 0.46±0.10** /
     se3_no_unc-on-fn 0.92±0.11 / enc_none 35.84±0.58 / no_adaptive_lam 27.88±3.88。
   - **单 seed λ 扫描批** `v2_control/control_lam*.json`（Fig.9 / Table 5）—— 该批**自己的**
     MLP = 10.63 mrad，λ=0 30.87、λ=1 10.57、λ=5 8.19。λ=1 与该批 MLP 的相等纯属巧合。
   - se3 训练于**干净** franka_ex 的 0.009 mrad **不可作为结果报告**：干净标签来自与 rollout
     同一个刚体仿真器，模型只需反演自己的数据生成器，是自证循环。手稿已明确弃用。

1. **旗舰模型不是 `full`**：`full`（rotmat + 7 维瓶颈 + 自适应 λ）= 0.9593，是最弱的「默认配置」；
   最优是 **sin/cos + 无瓶颈 + 固定 λ** = 0.9936。别把 full 当「Ours」写。

2. **「物理权重塌缩到 0」跨了两个编码**：
   - 自适应 λ 塌缩（λ̄≈2×10⁻⁹）在 **enc_none（原始关节角）** 模型上测（`v2_physics/enc_none_..._physics.json` 的 `E_lambda`，RESULTS §4 只显示到 0.0000，精确值看 json）。
   - 固定 λ 扫描（R² / g 误差 / 闭环 RMSE）在 **rotmat** 模型上测（RESULTS §9）。
   趋势单调、结论一致，但「λ=0 的 0.9643 / 30.9 mrad」和「塌缩到 2×10⁻⁹」不是同一个模型。正文二选一 + 脚注交代。

3. **闭环 36.0 vs 30.9 mrad**：RESULTS §7「学到的 g(q)」= 36.00 mrad（enc_none 自适应模型），
   RESULTS §9 λ=0 = 30.9 mrad（rotmat 固定 λ）。同为「物理被关掉」，编码不同。manuscript.md §5 已加口径注。

4. **已删除的错误断言（别再写回去）**：手稿旧版 §4.1 曾写「经典最小二乘能把真值恢复到
   浮点精度，因此神经网络的参数误差是训练缺陷」。该句引用的 `baseline_swevers.py` **从未
   跑通过**（MuJoCo `body_inertia` 只有 3 列，必然 IndexError），且结论**被证伪**：
   `baseline_ls_exact.py` 用精确回归量做无偏最小二乘，力矩 R²=1.00000000，而
   **100.0000% 的参数误差落在零空间**（‖e_id‖=4.7e-15）。参数恢复失败是问题的结构性质，
   不是训练问题——这正是本文的主命题。

5. **不得宣称 SE(3) 结构为本文贡献**：M(q)=ΣJᵀGJ 的李群形式是 Park et al. IJRR 1995
   （`park1995lie`），10 参数可微 Newton–Euler 参数化是 Sutanto et al. L4DC 2020
   （`sutanto2020encoding`）。本文定位是**工程类**：把 SE(3) 分支重参数化到 43 维可辨识
   子空间（π=Bθ，base parameters），使辨识良定、可复现；审计（秩 43/80）作为动机。dual
   ascent 只作为训练装置描述，不作为贡献。

5b. **重参数化学回的是 base 参数（可辨识组合），不是逐连杆物理参数**：π=Bθ 收敛到
   π_id=B·Bᵀ·π_true（可辨识投影），逐连杆 (m,c,I) 仍差真值一个零空间分量，且 π_id 把近端
   连杆质量压到 ≈0（非物理；质量相对误差 0.61，vs 朴素分支 0.244）。**正文绝不能写「学回
   真质量/质心/惯量」**——这是错的（经典可辨识性定理）。能学回的是 θ=Bᵀπ_true（base 参数，
   机器精度）+ 动力学不变（自检 8e-7）。卖点是「良定 + 可复现 + base 参数恰好是控制所需」，
   不是「逐连杆物理参数恢复」。恢复逐连杆参数需再加物理可行性约束（选观测等价解），留作
   后续工作。

6. **§4.2 重力子空间的叙述重心要押在 D⊄G，不是 G⊆D**：静态回归量是完整回归量在
   q̇=q̈=0 处的切片，所以 G⊆D（残差 1.60e-15）**按构造**成立，只是数值自检，不是发现。
   真正非平凡的结论是 **D⊄G（残差 0.984）**——「要学惯量必须有运动，光靠重力激励不够」；
   以及由此推出的「g(q) 恰好落在可辨识子空间里，所以参数错但控制对」。写正文时把这三层
   关系讲清楚：包含关系一笔带过，把笔墨放在 D⊄G 和「悖论消解」上，别把平凡的那一半当卖点。

7. **Baxter 左臂 (zyxyxyx) 的可辨识谱是「三层」的，不是单一秩**：Panda (zyzyzyz) 的谱是
   「43 强 + 27 精确零」的干净两段（秩断崖 ~17 个数量级）；Baxter 左臂是「**43 强 +
   18 弱(1e-8..1e-10) + 9 精确零**」。所谓「rank 58/59」落在弱带内，随采样点数(n=200→59、
   n=300/400→59) 和 seed 抖动，**不是干净秩，别当单一数字报**。正确表述：可辨识性是**运动学
   链**的性质——同 zyzyzyz 的两臂同秩 43（与惯量值无关，Fig.10 旧结论仍成立）；换 zyxyxyx 则
   **零空间形状改变**：精确零从 27 减到 9，多出 18 维弱近零空间（相对奇异值 1e-8..1e-10，
   实际力矩噪声下同样不可辨识）。**别把「58 vs 43」当卖点**；把「43 强方向两者共有 + 零空间
   形状由运动学决定」当卖点。三个采样点/两个 seed 都复核过：strong=43、weak=18、exact_null=9
   稳定；`--verify` 对 Baxter 的 SE(3) 自检也通过（max err 6.8e-6 < 1e-5）。

## 复现

```bash
python paper/run_grid.py --grid main --workers 6        # 主网格（11 变体 × 3 臂 × 5 seeds）
python paper/run_grid.py --grid data_eff --workers 6
python paper/run_grid.py --grid hparam --workers 6
python paper/run_grid.py --grid lam --workers 6
python paper/verify_physics.py --all
python paper/noise_robustness.py --arm franka_ex
python paper/control_sim.py --ckpt-phys results_v2/enc_none_franka_ex_seed0.pth --ckpt-mlp results_v2/mlp_franka_ex_seed0.pth
python paper/benchmark_coriolis.py
python paper/lam_tradeoff.py
python paper/analyze.py && python paper/report.py && python paper/figures.py
```
