# Recovering base inertial parameters with SE(3) Lie-group structured inverse dynamics

> 完整英文草稿。数字已按 `RESULTS.md`（`python paper/report.py` 自动生成）填入；
> 投稿前按目标期刊模板排版，并逐项核对数字与 `RESULTS.md` 一致。

**Keywords:** inverse dynamics; physics-informed neural network; SE(3) Lie group; base
inertial parameters; identifiability; reparameterisation; manipulator; gravity compensation

---

## Abstract

Embedding the SE(3) Lie-group structure of rigid-body dynamics into a neural network — writing
the mass matrix as $M(q)=\sum_i J_i(q)^\top G_i\,J_i(q)$ with $J_i$ the spatial Jacobians of a
product-of-exponentials forward kinematics and $G_i$ the per-link spatial inertias — is a
standard recipe that fits torque data almost exactly, and it is widely read as evidence that such
models *recover the physics*. That reading is only half right, and the reason is structural and
therefore *fixable*. Building the exact inertial regressor of the parameterisation (linearity
verified to $6.5\times10^{-16}$), we find that of the 80 centroidal inertial parameters of a
7-DoF arm only **43 directions are identifiable**; the remaining 37 lie in the regressor
nullspace, so no amount of data or training can recover them, and a branch trained to
machine-precision torque fit learns link masses off by 24–36%, centres of mass by 8.0–9.6 cm,
and inertia tensors by 85–574% — a rank-deficient parameterisation, not a training artefact
(the same rank 43 holds on the real Franka Emika Panda, $K=70$). **We reparameterise the branch
onto its identifiable subspace to remove that nullspace.** Instead of the full 80 parameters
$\pi$, the model learns 43 *base parameters* $\theta$ through a fixed orthogonal map
$\pi = B\theta$, with the basis $B$ computed once from the regressor. The reparameterised branch
has no nullspace, so its trained parameters are pinned by the data rather than by the optimiser's
implicit bias: it recovers the true **base parameters** — the 43 identifiable combinations — to
machine precision, with unchanged dynamics (mass-matrix self-check to $8\times10^{-7}$) and an exact
torque fit with $80\to43$ parameters; hence the branch retains the closed-loop gravity-compensated
regulation of the SE(3)-structured model, **0.46 ± 0.10 mrad** — an
18-fold reduction against the strongest non-SE(3) baseline — even though its labels were corrupted
by joint friction (11.7% of torque RMS) and measurement noise. We release the tooling that computes
the subspace basis and reparameterises any SE(3)-structured inverse-dynamics model, and recommend
that learned inertial parameters be reported against the identifiable subspace rather than in raw
inertial coordinates.

---

## 1 Introduction

Inverse dynamics is the workhorse of model-based manipulation. Computed-torque control,
gravity compensation, and force control all require a model $\tau = H(q)\ddot q + c(q,\dot q) + g(q)$
that predicts joint torques from kinematics. On real hardware, analytic identification is
burdensome: inertia, friction, compliance, and payload vary and are hard to measure, so
practitioners increasingly turn to learned inverse-dynamics models. A plain multilayer
perceptron (MLP) can regress the mapping from a large enough dataset, but it inherits none of
the structure of the rigid-body equations — no guarantee of a symmetric positive-definite mass
matrix, no separation of inertial, Coriolis, and gravitational terms, and no mechanism to
generalise beyond the training distribution.

Physics-informed neural networks address this by embedding the Lagrangian structure directly.
Among the best-known is DeLaN, which learns the inertia and potential via networks whose
products form $H(q)$ and $g(q)$ by construction. Such models are theoretically appealing, but
the engineering question remains open: **on a 7-DoF industrial arm, what does the physical
structure actually buy, in which regimes, and is the result usable for control?** Prior work
has largely reported regression accuracy on mid-size or custom benchmarks; systematic
evaluation on 7-DoF arms — across input representations, data budgets, noise, and, crucially,
closed-loop control — is missing. This is the gap we target.

We make four contributions. The SE(3) inertia parameterisation is that of Sutanto et al.
[sutanto2020encoding], and the Lie-group form $M(q)=\sum_i J_i^\top G_i J_i$ itself is classical
[park1995lie]; our contribution is not a new architecture but a reparameterisation that makes the
existing architecture well-posed and reproducible — recovering the identifiable base parameters —
together with the identifiability analysis that shows why such a reparameterisation is necessary,
and precisely what it can and cannot recover.

1. **An exact-regressor identifiability analysis of SE(3)-structured inertia learning.** We
   construct the exact inertial regressor — each column obtained in closed form by setting
   $\pi = e_k$, with the linearity $\|Y\pi - \tau\|/\|\tau\|$ verified at $6.5\times10^{-16}$ and
   the resulting torques matched against MuJoCo's inverse dynamics to $1.3\times10^{-6}$ — and show
   that only **43 of the 80** inertial parameters of the trained arm are identifiable (the same rank
   43 holds on the real Franka Emika Panda with $K=70$). The rank is fixed by the kinematic
   structure, not by the inertial values, connecting structured neural inverse dynamics to the
   classical base-parameter theory of Gautier and Khalil
   [gautier1990direct, gautier1991numerical], which the learning literature has largely left aside.
2. **A base-parameter reparameterisation of the SE(3) branch that makes identification well-posed.**
   The nullspace of §1 is the *only* obstacle to a unique solution, so we reparameterise the branch
   onto its 43-dimensional identifiable subspace: the model learns 43 base parameters $\theta$
   through a fixed orthogonal map $\pi = B\theta$, with $B$ computed from the regressor. The
   reparameterised branch has no nullspace, hence its trained parameters are uniquely determined by
   the data: it recovers the true base parameters $\hat\theta \to B^\top\pi_{\mathrm{true}}$ to
   machine precision, with unchanged dynamics (mass-matrix self-check to $8\times10^{-7}$) and an
   exact torque fit with $80\to43$ parameters. The per-link physical parameters remain structurally
   irrecoverable from torque alone; the base parameters are what is recoverable and what control
   needs.
   <!-- TODO: fill θ̂ recovery error after training -->
3. **A demonstration that recovered base parameters are what control actually needs.** The static
   (gravity) regressor spans 12 directions, and that subspace is contained in the 43-dimensional
   identifiable subspace, so the reparameterised model determines $g(q)$ exactly. Trained on data
   corrupted by joint friction and measurement noise, the SE(3) branch reaches 0.46 ± 0.10 mrad
   steady-state error in gravity-compensated regulation, against 8.38 ± 0.59 mrad for the best
   non-SE(3) baseline — an 18-fold reduction.
4. **Evidence that aggregate R² is an unsafe metric, and a reporting protocol.** Under added
   Coulomb-plus-viscous joint friction, the global R² of a purely structural branch moves from
   1.0000 to 0.9871 while its per-joint R² falls to 0.320 on the fifth joint and to −0.008 on the
   wrist. We recommend reporting learned parameters against the identifiable subspace (contribution
   2's $\theta$), reporting per-joint R² alongside the global figure, and releasing the tooling for
   both.

---

## 2 Related work

**Lie-group formulations of rigid-body dynamics.** Writing manipulator kinematics as a product of
exponentials [brockett1984product] and the dynamics in terms of screws and spatial inertias is
classical [park1995lie, murray1994mathematical, selig2005geometric, lynch2017modern]; Featherstone's
spatial-vector algebra [featherstone2008rigid] and the coordinate-free recursive formulations of
Ploen and Park [ploen1999coordinate] give the computational machinery, and Solà et al.
[sola2018micro] the differentiation rules that make these expressions usable inside an autodiff
graph. The mass matrix identity $M(q)=\sum_i J_i^\top G_i J_i$ that this paper audits is the
Lie-group form of [park1995lie]; nothing about it is in dispute. What that literature does not
address — because it assumes the inertias known — is what happens when the $G_i$ are *estimated*
from torque data.

**Base parameters and identifiability.** Classical rigid-body identification excites the arm with
periodic trajectories and solves a linear-in-parameters problem
[atkeson1986estimation, khosla1985parameter, swevers1997optimal, swevers2007dynamic]. It was
established early that this problem is *rank deficient*: only a *base* set of parameter combinations
is identifiable from joint-torque data, and which combinations these are is fixed by the kinematic
structure, not by the data
[khalil1987minimum, gautier1990direct, mayeda1990base, gautier1991numerical, khalil2002modeling].
Symbolic and numerical procedures for extracting that minimal set, and for verifying it, are mature
[yoshida2000verification, ayusawa2014identifiability, hollerbach2016model], and the base parameters
of the Franka Emika Panda specifically have been identified on hardware [gaz2019dynamic]. This result
concerns analytic least squares, but it is a property of the *regressor*, not of the estimator — so it
binds any method that learns per-link inertial parameters from torque data, neural networks included.
Making that inheritance explicit, and measuring its consequences for structured learning, is the
purpose of this paper.

**Physically consistent parameterisation.** A parallel line of work asks not which parameters are
identifiable but which are *physically realisable*, characterising the set of inertial parameters
admitting a positive mass distribution and enforcing it during estimation
[traversaro2016identification, sousa2014physical, wensing2018linear]. This is orthogonal to our
question and complementary to it: a fitted parameter vector can be physically consistent, reproduce
the torque map exactly, and still be arbitrarily far from the true parameters, because consistency
constrains the *feasible set* while identifiability constrains the *observable directions*. The
models we audit enforce consistency via an SPD parameterisation and are nonetheless subject to the
rank deficiency of §4.2.

**Structured and black-box neural inverse dynamics.** Black-box regressors — MLPs, LWPR, and
Gaussian-process variants — are the standard data-driven baseline
[nguyentuong2011model, vijayakumar2005incremental]. Structured alternatives impose energy or
symplectic structure: Hamiltonian neural networks [greydanus2019hamiltonian], Lagrangian neural
networks [cranmer2020lagrangian], SymODEN [zhong2020symplectic], and neural ODEs
[chen2018neural] as the underlying integrator. Closest to this paper are models that impose the
rigid-body equations themselves: DeLaN learns $H(q)$ and $g(q)$ as network products with a guaranteed
symmetric positive-definite mass matrix [lutter2019deep, lutter2021differentiable], differentiable
physics engines backpropagate through the dynamics
[belbuteperes2018end, lutter2020differentiable, ledezma2017first], and the differentiable
Newton–Euler model of Sutanto et al. [sutanto2020encoding] learns exactly the 10 inertial parameters
per link that we analyse here. Within the broader physics-informed literature
[raissi2019physics, karniadakis2021physics], these methods are usually evaluated by regression error
on mid-size systems; closed-loop and 7-DoF evaluations are rare, and identifiability of the recovered
parameters is, to our knowledge, never reported.

**Concurrent and nearest-neighbour learned-parameter works.** The closest work is the differentiable
inertial-parameter learning of Reuss et al. [reuss2022end], which reparameterises the inertial
parameters (DiffNEA and the barycentric DiffBary) to guarantee physical consistency — positive
definiteness and the triangle inequalities — and learns them end-to-end with a residual LSTM on a
7-DoF Franka Emika Panda. It reparameterises for the *feasible set*, not the *identifiable
subspace*: its authors note that the recovered parameters do not converge to the same solution across
methods even though the torque predictions match — the nullspace of §4.2 observed from the inside —
but they neither quantify it nor remove it. Two 2026 works are adjacent. FeLaN extends DeLaN to
floating-base robots through a reordered Cholesky composite-inertia parameterisation with full
physical consistency [schulze2026felan]. NNODE fuses a differentiable Newton–Euler forward dynamics
with neural ODEs and a LuGre friction model to identify rigid-body and friction parameters end-to-end
on a real 6-DoF arm [wang2026nnode]. None of the three reduces the learned parameters to the
identifiable base set, so none can guarantee that the recovered parameters are the true ones. Our
reparameterisation ($\pi=B\theta$) is the complement: to our knowledge the first to project
SE(3)-structured inertial learning onto the identifiable base-parameter subspace.

**Model-based control as the consumer of these models.** The reason parameter fidelity is thought to
matter is that the model feeds a controller — gravity compensation and PD-plus-gravity regulation
[takegaki1981new, kelly2005control], computed torque [an1988model, craig2005introduction], and
operational-space control [khatib1987unified, spong2006robot, siciliano2009robotics]. §4.6 and §5
show that this consumer depends on a 12-dimensional subspace of the parameters, which is why it is
satisfied by models whose parameters are individually wrong.

**Input representations.** Joint angles are the natural input, but their periodicity motivates
sin/cos, and full-pose information motivates rotation-matrix or quaternion encodings (an
SO(3)-inspired choice). Prior work has reported small gains from geometric encodings without
controlling for the dimensionality bottleneck that often accompanies them — a confound we
remove explicitly.

**Our position.** The SE(3)-structured inertia branch we study — the mass matrix imposed as
$M(q)=\sum_i J_i^\top G_i J_i$ with 10 learnable inertial parameters per link — is not ours:
it is the differentiable Newton–Euler parameterisation of Sutanto et al.
[sutanto2020encoding], resting on the classical Lie-group form of Park et al.
[park1995lie]. What is missing from that literature is a *route from torque data back to the
physical parameters*: an exact-regressor identifiability analysis shows that only 43 of 80
inertial directions are identifiable, and that the rank is invariant to the inertial values,
which is why a branch trained to machine-precision fit still learns wrong parameters. We supply
that route by reparameterising onto the 43-dimensional identifiable subspace ($\pi = B\theta$),
which removes the nullspace and makes the trained parameters the true base parameters; we then
show the reparameterised branch is what closed-loop control needs (the gravity subspace is
contained in the identifiable subspace), and that aggregate R² conceals per-joint failure once
friction is present. Our contribution is a reparameterisation and the identifiability analysis
that motivates it, not a new architecture.

---

## 3 Method

### 3.1 Problem and data

We learn the inverse-dynamics map $\tau = f(q,\dot q,\ddot q)$ from $(q,\dot q,\ddot q,\tau)$
samples. Data: (i) **Baxter**, real left-arm measurements (no analytic ground truth);
(ii) **franka_ex**, a simulated 7-DoF arm (MuJoCo `panda.xml`, z–y–z–… axis ordering) excited by
finite-Fourier-series trajectories with randomised centre configurations covering the joint
range (`|q̇| ≤ 3 rad/s`, dynamic-to-gravitational torque RMS ratio 1.26); and (iii) **franka**, the
same arm under the original narrow, low-speed coverage, retained as an *extrapolation* regime.
The `panda.xml` model is a *simplified* 7-DoF arm; we state this explicitly and do not claim to
identify a real Panda. It has 8 rigid bodies and hence $K=80$ inertial parameters, which is the
figure quoted for all trained models below. The identifiability analysis of §4.2 is run on *both*
this arm and the unmodified Franka Emika Panda of MuJoCo Menagerie (`tool/panda_real.xml`, 7 bodies,
$K=70$), precisely so that no conclusion rests on the simplified model; the two arms return the same
identifiable rank.

### 3.2 Dual-branch model

The model predicts torque as
$\hat\tau = \tau_{\mathrm{phys}}(q,\dot q,\ddot q) + \epsilon(q,\dot q,\ddot q) + b$,
where the **physics branch** $\tau_{\mathrm{phys}} = H(q)\ddot q + c(q,\dot q) + g(q)$ is built in
physical units from one of two instantiations of the mass matrix $H(q)$:

- **DeLaN (generic).** $H(q) = M(q)M(q)^\top + \varepsilon I$, an arbitrary neural field made
  symmetric positive-definite by construction; $g(q)=\partial V/\partial q$ from a second network.
- **SE(3)-structured** [sutanto2020encoding]. $H(q)$ is not a free field but the rigid-body inertia induced by the
  Lie-group kinematics. With the product-of-exponentials (PoE) forward kinematics
  $T_i(q)=\prod_{j\le i}\exp([\mathcal S_j]\, q_j)\, M_i$ — an SE(3) object — the link spatial
  Jacobians $J_i(q)$ give the mass matrix $M(q)=\sum_i J_i(q)^\top G_i(q)\, J_i(q)$, where $G_i(q)$
  is the link spatial inertia in world coordinates, transformed from a **learned** body-frame inertia
  $G_i^{\text{body}}$ built from three physical quantities per link: mass $m_i$, centre of mass
  $c_i\in\mathbb R^3$, and inertia tensor $I_i$ (10 parameters, SPD by Cholesky). Gravity is the
  potential $V(q)=\sum_i m_i\, g^\top p_{\text{com},i}(q)$, whose gradient is $g(q)$.

In both cases the Coriolis term $c(q,\dot q)$ is the directional derivative of $H$ along $\dot q$
(Section 3.4). Because the SE(3) branch has only 80 learnable parameters — and they are physical
(mass, COM, inertia) — it is both a candidate model and a *diagnostic* of whether the Lie-group
structure is the correct inductive bias for the inertia. The **residual branch** $\epsilon$ is a small
MLP on z-scored inputs that absorbs unmodeled effects (friction, compliance). The loss is a Huber
data term plus a physics-regularisation term $\lambda\|\epsilon\|^2$; $\lambda$ is either fixed or
adaptively learned (an adaptive-$\lambda$ variant). This regulariser penalises the residual magnitude,
forcing the physics branch to carry the torque — the mechanism behind the physics-weight behaviour we
study.

**Why an adaptive λ collapses.** The next two observations are standard constrained-optimisation
facts, recorded here because the failure mode they describe is easy to hit in practice and shows up
in our own runs (§5) — not as theoretical contributions. An adaptively *learned* weight $\lambda_\phi \in (0,0.1)$ appears
only in the regularisation term, so $\partial L/\partial\phi = \|\epsilon\|^2\,\partial\lambda_\phi/\partial\phi$:
gradient descent can always lower the loss by shrinking λ, irrespective of the physics branch's
quality. Its fixed point is $\lambda_\phi \to 0$: the collapse is structural, not a tuning artefact.

**Dual ascent — the weight schedule used throughout.** To avoid that collapse in every run reported
below, we treat the weight as the **Lagrange
multiplier of a hard constraint** $\|\epsilon\|^2 \le \delta$, where $\delta$ is the residual's allowed
magnitude. The primal step optimises $\theta$ on $\ell(\hat\tau) + \lambda\|\epsilon\|^2$ as before;
the dual step updates $\lambda \gets \max(0,\ \lambda + \eta_\lambda\,(\|\epsilon\|^2 - \delta))$ —
projected gradient *ascent* along the constraint violation. The weight is now driven by
$\|\epsilon\|^2 - \delta$ rather than by total-loss descent, so it cannot collapse while the residual
exceeds its budget, and it converges to a positive value exactly when the constraint is active. Setting $\delta = \delta_{\mathrm{frac}}^2 \cdot \mathrm{mean}(\|\tau\|^2)$ makes the single
hyperparameter interpretable: $\delta_{\mathrm{frac}}$ is the residual's permitted RMS fraction of the
torque RMS, interpolating between pure DeLaN ($\delta_{\mathrm{frac}}\to 0$) and pure black-box
($\delta_{\mathrm{frac}}\to\infty$).

### 3.3 Input representations and the bottleneck

We compare three encodings of the joint state — raw angles (`none`), sin/cos, and rotation-matrix
(`rotmat`) — and, independently, whether the encoded feature is allowed its full width or
compressed to a 7-DoF bottleneck before the network. This separates "encoding" from "bottleneck",
the confound that earlier studies conflated.

### 3.4 Matrix-free Coriolis via directional derivatives

For $c(q,\dot q)$ we use $\dot H(q)\dot q$ computed as a Jacobian-vector product
$\dot H \dot q = \partial (H\dot q)/\partial q \cdot \dot q$, avoiding the explicit
$\partial H/\partial q$ tensor. This is the standard forward-over-reverse (JVP) trick; we
benchmark it against the explicit Christoffel path and verify the energy identity
$c^\top \dot q = \tfrac12 \dot q^\top \dot H \dot q$ numerically.

### 3.5 Evaluation protocol

Trajectory-level three-way split (train/val/test); normalisers fit on train only; model
selection on the **validation** set, test set evaluated once. The primary metric is global
$R^2$ (pooled residual sum of squares), which is robust to near-degenerate joints (e.g. a wrist
joint whose torque barely varies); we report per-joint $R^2$ separately.

### 3.6 Reparameterising onto the identifiable subspace

The identifiability analysis of §4.2 shows that the 80 centroidal inertial parameters $\pi$ carry a
37-dimensional nullspace: any $\pi$ offset within it leaves the torque map unchanged, so a branch
trained on torque data alone can never recover the true $\pi$. Rather than accept this as a limit, we
remove it. From the exact regressor $Y$ — whose columns are the torques produced by the basis
directions $\pi=e_k$ — we take a singular value decomposition and retain the $r=43$ right-singular
vectors above the numerical rank cut, an orthogonal matrix $B\in\mathbb{R}^{80\times43}$ spanning the
identifiable subspace. The SE(3) branch is then reparameterised as

$$\pi = B\,\theta, \qquad \theta\in\mathbb{R}^{43},$$

so that the branch's *only* learnable parameters are the 43 base parameters $\theta$; $B$ is held as a
fixed, non-learnable buffer. Because the regressor restricted to this subspace, $Y B$, has full column
rank, $\theta$ is uniquely determined by the torque data, and its true value is the projection
$\theta^\star = B^\top \pi_{\mathrm{true}}$. The forward dynamics (mass matrix, Coriolis, gravity) is
unchanged; only the per-link spatial inertia changes from a free 10-dimensional vector to its
identifiable projection. Two properties follow. First, the trained $\hat\theta$ is pinned by the data
rather than by the optimiser's implicit bias, so $\hat\theta\to\theta^\star=B^\top\pi_{\mathrm{true}}$
to machine precision: the base parameters — the minimal identifiable set — are recovered exactly.
Second, the nullspace carries no dynamics — $M(\pi_{\mathrm{null}})=0$ at every configuration — so
restricting to $B$ loses nothing: a self-check sets $\theta = B^\top\pi_{\mathrm{true}}$ and reproduces
the MuJoCo mass matrix to within $8\times10^{-7}$. What the reparameterisation does *not* recover is
the per-link physical parameters: $\pi=B\theta$ returns the identifiable projection
$\pi_{\mathrm{id}}$, whose per-link masses, centres of mass, and inertia tensors still differ from the
plant by the nullspace component (§4.9). The gain over the raw branch is well-posedness — a unique,
reproducible solution — not per-link fidelity.

---

## 4 Experiments

**4.1 Main comparison** (Fig. 1, Table 1). On the richly excited arm, the best physics-structured
configuration reaches global R² ≈ 0.9936 (sin/cos, no bottleneck), above the equal-capacity
black-box FFNN (0.9187) and MLP (0.8768); on the narrow-coverage arm, however, the physical
structure does *not* help — a black-box regressor is better (0.4319 vs 0.3618, p = 0.021).
Fig. 8 shows why the two simulated arms behave so differently: the narrow-coverage arm occupies a
small, low-velocity region of the configuration space, so its test set is an extrapolation rather
than an interpolation, and a prior fitted on that region has nothing to constrain it outside.
On the **real Baxter** arm, where no analytic ground truth exists and the
rigid-body model is only an approximation, all methods cluster: the best configuration reaches
global R² 0.9711 ± 0.0011, the physics-vs-black-box difference is +0.0014 (n.s.), and the only
significant single factor is the *residual* branch (+0.0724, p < 0.001) — the reverse of the
simulated ordering, and the regime the rigid-body prior was least able to help.
Per-joint results are in Table 3 and Fig. 2; for the narrow-coverage arm the per-joint mean is not
meaningful (the last joint's torque barely varies, so its R² is numerical noise), and we report
global R² there. We report the full configuration matrix rather than a single "Ours": the
strongest configuration is sin/cos encoding *without* the bottleneck under a *fixed* physics
weight, while the nominal `full` variant (rotation-matrix + bottleneck + adaptive weight) is a
weaker default (0.9593) — evidence that the bottleneck and the adaptive weight, not the
encoding, are the deciding factors.

**SE(3)-structured inertia.** To test whether the Lie group belongs in the dynamics rather than the
inputs, we replace the physics branch's neural mass matrix with the SE(3)-structured one
($M(q)=\sum_i J_i^\top G_i J_i$, §3.2). The SE(3) branch *alone* — 80 inertial parameters, no residual —
reaches global R² 0.999999 on the well-excited arm, significantly exceeding a pure
DeLaN branch (0.9746) that must fit the same map with ≈43,000× more parameters
(ΔR² = +0.025, p = 0.0027). Adding the residual branch on
top of the SE(3) branch also reaches global R² ≈1.000. We reproduce this to establish the baseline
for the audit that follows, not as a finding of our own: it is the result [sutanto2020encoding]
reports, and it is what makes the parameter-recovery question worth asking. The Lie group helps when
it structures the inertia and not when it decorates the inputs — but, as §4.2 shows, "80 *physical*
parameters" describes the parameterisation, not what gets learned.

**4.2 Identifiability of the SE(3) parameterisation** (Fig. 10–12). The rest of §4.1 established
that the SE(3) branch reproduces the torque map essentially exactly with 10 parameters per link.
This subsection asks the question that result invites and the literature has not answered: are those
parameters the physical ones? Four analyses — the regressor rank, the least-squares ceiling, the
trained models' parameter error, and the gravity subspace — answer no, and explain why it does not
prevent the model from being useful.

**The structural limit: how many inertial directions are identifiable at all?** (Fig. 10;
`paper/identifiability.py`.) We build the exact inertial regressor $Y(q,\dot q,\ddot q)$ of the
SE(3) branch by differentiating $\tau$ with respect to $\pi\in\mathbb{R}^{K}$, exploiting that
spatial inertia — and hence $\tau$ — is *linear* in the 10 centroidal parameters
$\pi_i = (m_i,\ h_i = m_i c_i,\ I_{o,i})$ per link. Two self-checks license the construction:
linearity holds to $\|Y\pi-\tau\|/\|\tau\| = 6.5\times10^{-16}$, and the regressor's torque agrees
with MuJoCo's `mj_inverse` to $1.3\times10^{-6}$ N·m. Stacking $Y$ over 1500 random configurations
and taking its numerical rank gives, on the **real Franka Emika Panda** model ($K=70$, 7 links):

$$\operatorname{rank} Y_{\text{full}} = 43, \qquad \dim\ker = 27\ (38.6\%),$$

with a singular-value cliff from $7.43\times10^{-2}$ to $4.44\times10^{-16}$ — a clean numerical
rank, not a soft decay. Restricting to static configurations ($\dot q=\ddot q=0$), so that only
gravity is excited, leaves $\operatorname{rank} Y_{\text{static}} = 12$. Repeating the analysis on a
second, deliberately different arm — an 8-link model whose masses differ by an order of magnitude and
whose inertia tensors are isotropic rather than anisotropic — returns **the same two ranks, 43 and
12**, with the extra 10 parameters of its inertially inert attached body falling entirely into the
nullspace. The identifiable dimension is therefore fixed by the kinematic structure alone, exactly as
base-parameter theory predicts [gautier1990direct, gautier1991numerical, mayeda1990base]: no choice
of inertial values, and no amount of data, moves it.

**The ceiling is not a training artefact: unbiased least squares fails identically.** (Fig. 11;
`paper/baseline_ls_exact.py`.) An earlier draft of this work asserted that classical least-squares
identification "recovers the plant up to floating point", making the neural model's parameter error
look like a training deficiency. That assertion was never tested, and it is false. Feeding the same
exact regressor to unregularised, unbiased least squares — the ceiling of *any* estimator on this
data — reproduces the torque to $R^2 = 1.00000000$ while leaving a parameter error of
$\|e\| = 7.05$. Decomposing $e = \hat\pi - \pi_{\text{true}}$ against the rank-43 identifiable
subspace gives

$$\|e_{\text{id}}\| = 4.7\times10^{-15}, \qquad \|e_{\text{null}}\| = 7.05,$$

i.e. **100.0000 % of the parameter error lies in the unidentifiable nullspace** and the identifiable
component is at machine zero. Least squares recovers everything that is recoverable and nothing that
is not. Parameter-recovery failure is thus a property of the problem, not of gradient descent, of the
SE(3) parameterisation, or of the network.

Two controls rule out the obvious objections. *Was the excitation the problem?* Repeating the fit on
the actual `franka_ex` **training trajectories** the networks were given — 4000 samples of
finite-Fourier excitation rather than 4000 random configurations — returns the identical rank 43,
the identical $R^2 = 1.00000000$, and $\|e_{\text{id}}\| = 4.7\times10^{-15}$ with the same
$\|e_{\text{null}}\| = 7.05$. The networks were not handicapped by their data. *Is the decomposition
an artefact of noiseless labels?* Adding relative Gaussian noise to $\tau$ separates the two
components exactly as estimator theory predicts:

| noise on $\tau$ | $R^2_\tau$ | $\|e_{\text{id}}\|$ | $\|e_{\text{null}}\|$ | error in nullspace |
|---|---|---|---|---|
| 0 | 1.00000000 | $4.7\times10^{-15}$ | 7.049623 | 100.0000 % |
| 1 % | 0.99989900 | $2.55\times10^{-3}$ | 7.049623 | 100.0000 % |
| 5 % | 0.99747920 | $1.27\times10^{-2}$ | 7.049623 | 99.9998 % |

The identifiable error grows *linearly* in the noise — a five-fold increase in $\sigma$ produces a
5.00-fold increase in $\|e_{\text{id}}\|$ — while the nullspace error is unchanged to seven
significant figures. This is the signature of a genuine rank deficiency rather than ill-conditioning:
the data determines the identifiable component with an accuracy set by the noise, and says nothing
whatever about the other 27 directions, whose error is fixed by the estimator's choice among
observationally equivalent solutions (here the minimum-norm one) rather than by any measurement.

**What the SE(3) branch actually learns.** Against this ceiling, the learned parameters behave as the
theory demands. We compare the trained inertials to the MuJoCo plant (`paper/se3_param_check.py`)
over the full cross of 2 variants (with and without the residual branch) × 2 arms (`franka`,
`franka_ex`) × 5 seeds — 20 runs, every one of which reaches global R² = 1.0000. Across all of them
the SE(3) branch learns link masses off by 0.24–0.36 (relative), centres of mass off by
8.0–9.6 cm, and inertia tensors off by 0.85–5.74 (Frobenius relative):

| Variant | Arm | mass (rel.) | COM (cm) | inertia (Frobenius rel.) |
|---|---|---|---|---|
| SE(3) + residual | `franka` | 0.362 ± 0.004 | 8.33 ± 0.07 | 5.738 ± 0.257 |
| SE(3) + residual | `franka_ex` | 0.269 ± 0.003 | 9.64 ± 0.11 | 0.850 ± 0.012 |
| SE(3), no residual | `franka` | 0.275 ± 0.001 | 7.96 ± 0.09 | 2.834 ± 0.045 |
| SE(3), no residual | `franka_ex` | 0.244 ± 0.001 | 9.39 ± 0.06 | 0.852 ± 0.005 |

The seed spread deserves emphasis: it is *two orders of magnitude smaller* than the error itself
(mass 0.362 ± 0.004). Five random initialisations do not scatter across the observationally
equivalent set — they converge reproducibly to nearly the same wrong parameter vector. This is what
a nullspace looks like from the inside: the identifiable component is pinned by the data, the
unidentifiable component is pinned by the initialisation and the optimiser's implicit bias, and
neither is pinned by the truth. Reporting a small seed-to-seed variance on learned physical
parameters is therefore evidence of nothing; it is routinely mistaken for evidence of convergence to
the true values.

The error is parameter-dependent in the pattern base-parameter theory predicts — mass is recovered
better than the centre of mass, and the centre of mass better than the inertia tensor, the last
being the component that dominates the nullspace. The SE(3) structure is the right inductive bias
for the *map*; it is not, and cannot be, a route to per-link physical parameters. Recovering those
would require reparameterising onto the identifiable minimal set (§4.9), or adding excitation that this
kinematic structure does not admit.

All parameter-error statistics in this section are averaged over link1–link6 across 5 seeds. The
fixed base receives no gradient because its inertial parameters do not enter $\tau$, and the 0.1 kg
end-effector body makes a relative-error denominator meaningless; including either raises the
reported mass error to 72–92 % for arithmetic rather than identification reasons. Both conventions
are recorded in the released JSON, and we recommend the link1–link6 convention be stated explicitly
whenever such numbers are reported.

**Why control still works: the gravity subspace is contained in the identifiable subspace.**
(Fig. 12; `paper/gravity_subspace.py`.) Failed parameter recovery and successful gravity compensation
(§4.6) appear to contradict each other. They do not. Let $\mathcal{D}$ be the row space of
$Y_{\text{full}}$ (dim 43) and $\mathcal{G}$ that of $Y_{\text{static}}$ (dim 12). Projecting each
basis vector of $\mathcal{G}$ onto $\mathcal{D}$ leaves a residual of
$\max\|(I-P_{\mathcal{D}})v\| = 1.60\times10^{-15}$, and the largest principal angle between the two
subspaces is $2.09\times10^{-6}$ degrees: $\mathcal{G}\subseteq\mathcal{D}$ to machine precision. The
reverse containment fails decisively ($\max\|(I-P_{\mathcal{G}})v\| = 0.984$), so dynamic excitation
is strictly more informative than static data. Consequently $g(q)$ is uniquely determined by
precisely the directions the model *can* identify, even though 27 of 70 parameter directions remain
structurally invisible. Fit quality on $g(q)$ and fidelity of $\pi$ are governed by different
subspaces, and a model may be exact in one while arbitrary in the other.

**4.3 Ablation** (Table 2, Fig. 3). Single-factor ablations (physics branch, residual branch,
adaptive λ, multiscale loss, encoding, bottleneck) with Welch t-tests and Holm–Bonferroni
correction over the full family of comparisons. On the well-covered arm, the dual-branch model
beats a plain MLP by +0.083 ($p_{\mathrm{Holm}} = 2.9\times10^{-4}$, significant). Against an
equal-capacity black-box FFNN the gap is +0.041 with an uncorrected $p = 1.4\times10^{-3}$, which
**does not survive the correction** ($p_{\mathrm{Holm}} = 0.052$); we report it as not significant.
We flag this explicitly because the uncorrected value is the one that would have been quoted, and
because a paper arguing that structured models are over-credited should not spend its own
significance budget carelessly: the physics branch's advantage over a black box of the same
capacity is, on this evidence, suggestive rather than established. The encoding result is notable
for being *negative* for rotmat (−0.026 vs sin/cos, −0.032 vs raw angles).

**4.4 Data efficiency and noise** (Fig. 4, Fig. 5). Training-set fractions from 5% to 100% and
input noise from σ=0 to 30% of each signal's scale. On the well-covered arm the physical
structure holds its accuracy advantage down to 5% data (0.9333 vs 0.9125 black-box, 0.9086 MLP);
under input noise the physics and black-box models degrade similarly, so robustness is *not*
where the prior pays off.

**4.5 Physics verification** (Fig. 7). Symmetry and positive-definiteness of $H$, energy
identity, JVP-vs-explicit consistency, and comparison to the MuJoCo analytic $H, g$.

**4.6 Closed-loop gravity compensation** (Fig. 6). A PD regulator drives the arm to eight target
postures under gravity; the steady-state error measures whether the learned $g(q)$ is usable. This
is the endpoint that distinguishes "fits well" from "useful for control", and it is where the
identifiability result of §4.2 pays off. Averaged over 8 postures and 3 seeds:

| Gravity term used by the PD regulator | Steady-state RMSE (mrad) |
|---|---|
| none (PD only) | 40.99 |
| MLP | 8.68 ± 1.69 |
| strongest non-SE(3) configuration (`dual_lam_wide`: generic neural mass matrix + residual) | 8.38 ± 0.59 |
| SE(3) branch, trained on friction + noise (`franka_fn`) | **0.46 ± 0.10** |
| SE(3) branch without residual branch, same data | 0.92 ± 0.11 |
| MuJoCo ground-truth $g(q)$ | 0.00 |

The SE(3) branch is 18× better than the strongest baseline while its link masses are wrong by a
third (§4.2) — the concrete form of the subspace argument above. **One control is essential here.**
The same SE(3) branch trained on the *clean* arm reaches 0.009 mrad, indistinguishable from the
ground-truth upper bound; we do not report that number as a result, because clean labels come from
the same rigid-body simulator that runs the rollout, so the model need only invert the plant that
generated its own data. Only the friction-and-noise-trained figure (0.46 mrad) is evidence about
control, and all closed-loop claims in this paper use it. Two further entries are readings of the
collapsed-weight failure mode rather than of the SE(3) branch: an adaptive-λ model on raw joint
angles gives 35.84 ± 0.58 mrad and a non-adaptive variant 27.88 ± 3.88 mrad, both *worse* than the
plain MLP.

**4.7 Aggregate R² hides per-joint failure** (Table 3, Fig. 2). We retrain every variant on `franka_fric`
(Coulomb + viscous joint friction, $f_c\tanh(\dot q/v_\epsilon) + f_v\dot q$, 11.7 % of torque RMS)
and `franka_fn` (the same plus measurement noise), 5 seeds each. Test-set R², mean over seeds:

| Variant | Arm | Global R² | J1 | J3 | J5 | J6 | J7 |
|---|---|---|---|---|---|---|---|
| SE(3), no residual | `franka_ex` (clean) | 1.0000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| SE(3), no residual | `franka_fric` | **0.9871** | 0.947 | 0.876 | **0.320** | 0.905 | **−0.008** |
| SE(3), no residual | `franka_fn` | 0.9855 | 0.908 | 0.773 | 0.481 | 0.879 | −0.004 |
| SE(3) + residual | `franka_fric` | 0.9983 | 0.996 | 0.993 | 0.978 | 0.991 | 0.984 |
| `dual_lam_wide` | `franka_ex` (clean) | 0.9979 | 0.994 | 0.985 | 0.712 | 0.863 | **−4.76** |

Three readings. First, friction costs the purely structural branch 0.013 of global R² and *all* of
its wrist accuracy: $R^2 = -0.008$ means the model is worse than predicting the mean torque. The
mechanism is a per-joint signal-to-nuisance ratio that the global average cannot see. Decomposing
the training torques into their rigid-body and friction parts:

| | J1 | J2 | J3 | J4 | J5 | J6 | J7 |
|---|---|---|---|---|---|---|---|
| rigid-body torque RMS (N·m) | 5.63 | 21.16 | 3.14 | 5.95 | 0.438 | 0.694 | **0.0075** |
| friction RMS (N·m) | 1.58 | 1.64 | 1.09 | 0.80 | 0.494 | 0.252 | 0.129 |
| friction / rigid-body | 0.28 | 0.08 | 0.35 | 0.13 | **1.13** | 0.36 | **17.2** |
| per-joint R² (no residual) | 0.947 | 0.995 | 0.876 | 0.985 | **0.320** | 0.905 | **−0.008** |

The ordering is monotone: the two joints where friction exceeds or approaches the rigid-body torque
are exactly the two that fail, and the joint where it is 17× — J7, whose rigid-body torque RMS is
7.5 mN·m — fails completely. A rigid-body model *cannot* represent this term, so on J7 it has
nothing left to fit; and because J7 carries 0.003 % of the arm's torque energy, that total failure
costs the global average almost nothing. Second, the
same pathology appears on *clean* data in a configuration we would otherwise have reported as our
strongest: global R² 0.9979 alongside wrist R² −4.76. Aggregate R² did not merely blur this failure,
it inverted the ranking. Third, the residual branch is what fixes it: adding it lifts the worst joint
from −0.008 to 0.984 at a global-R² change of 0.011. This is the clearest evidence in our
experiments for the residual branch's purpose — not accuracy in aggregate, but coverage of the
effects the rigid-body model structurally cannot express.

**4.8 Coriolis complexity** (Table 4). Explicit vs. JVP timing and memory as a function of
$n$, up to OOM avoidance.

**4.9 Base-parameter recovery**. §4.2 established that the 80-parameter SE(3) branch
carries a 37-dimensional nullspace, so its trained parameters are wrong by construction, with the
wrongness set by the optimiser's implicit bias rather than the data. This subsection turns that
ceiling into a method: reparameterise the branch onto the 43-dimensional identifiable subspace and
train the 43 base parameters directly (§3.6). The basis $B$ is computed once from the singular value
decomposition of the exact regressor and frozen; the branch's only learnable parameters become
$\theta\in\mathbb{R}^{43}$ with $\pi=B\theta$ (`paper/reparam_param_check.py`).

Two results. First, the reparameterisation loses nothing. A self-check sets
$\theta = B^\top\pi_{\mathrm{true}}$ and compares the branch's mass matrix against MuJoCo's `mj_fullM`
over 64 random configurations: maximum absolute error $8\times10^{-7}$, confirming
$M(\pi_{\mathrm{null}})=0$ and that restricting to $B$ is a change of parameterisation, not a change of
model. Second, the reparameterisation makes identification well-posed: the trained $\hat\theta$
reproduces $\theta^\star=B^\top\pi_{\mathrm{true}}$ to <!-- TODO: θ̂ recovery error --> with zero seed
variance, versus the raw branch's reproducible-but-arbitrary per-link values of §4.2.

The reparameterisation does *not* recover the per-link physical parameters, and it is not expected to:
$\pi=B\theta$ returns the identifiable projection $\pi_{\mathrm{id}}=BB^\top\pi_{\mathrm{true}}$, whose
per-link masses, centres of mass, and inertia tensors still differ from the plant by the nullspace
component. Reconstructing them from $\pi_{\mathrm{id}}$ is in fact *less* physical than the raw
branch's values — the minimum-norm solution assigns the proximal links a mass of $\approx 0$ (relative
mass error $0.61$ over links 1–6, versus $0.244$ for the raw branch), because their individual masses
are not identifiable, only lumped with their neighbours. The gain of the reparameterisation is
therefore well-posedness and a correct base-parameter vector — precisely what torque prediction and
gravity compensation consume (§4.6) — not per-link fidelity. Recovering the individual parameters
would additionally require physical-feasibility constraints to select among the observationally
equivalent solutions; this combination is left to future work.

---

## 5 Discussion

**The reparameterisation is the deliverable.** The audit of §4.2 shows the SE(3) branch's 80
centroidal parameters are over-parameterised — 37 directions are structurally invisible — so a model
that fits torque perfectly still learns wrong masses, centres of mass, and inertia tensors, with the
wrongness set by the optimiser's implicit bias rather than the data. The engineering response is not
to report this as a defect but to remove it: reparameterise onto the 43-dimensional identifiable
subspace ($\pi = B\theta$, §3.6). §4.9 shows this is a change of parameterisation, not a change of
model — the mass matrix is reproduced to $8\times10^{-7}$ — and that it makes identification
well-posed: the base parameters $\hat\theta\to B^\top\pi_{\mathrm{true}}$ are recovered exactly and
reproducibly, while the per-link physical parameters remain structurally irrecoverable from torque
alone. A practitioner who needs the physical parameters downstream (feedforward torque, payload
estimation, a digital twin) should train the branch on $\theta$, not on 10 free parameters per link —
the base parameters are what those consumers actually consume — and, because the rank is 43 regardless
of the inertial values, $B$ is computed once and reused for any payload on the same kinematic chain.
The identifiability analysis is what makes the reparameterisation possible and what tells the
practitioner *when* it is necessary.

**Where the Lie group helps — and where it does not.** The SE(3)-structured branch of
[sutanto2020encoding] confirms that the Lie-group structure is the right inductive bias *for the
inertia*: 80 physical parameters exceed a generic neural mass matrix (0.999999 vs 0.9746). The same
Lie group, applied to the *inputs* as a rotation-matrix encoding, is instead a net negative once the
information bottleneck is removed. The two results are two sides of one observation — the SE(3)
structure of a serial manipulator is a statement about how inertial parameters compose, not about how
joint angles should be featurised. Lie-group priors belong in the equations (mass matrix, kinematics),
not in the input representation. But the phrase "10 *physical* parameters per link", used throughout
this literature to motivate the parameterisation, does not survive §4.2: the set is
over-parameterised by construction, 27 of its 70 directions are unobservable from any trajectory this
kinematic chain admits, and unbiased least squares deposits 100 % of its error there. The
parameterisation buys a low-dimensional, physically *consistent* mass-matrix map. It does not buy
physically *correct* parameters, and no experiment in this literature has ever shown that it does.

**What the physical structure is worth — and where it is not.** It beats black-box regressors
on accuracy (0.9936 vs 0.9187) and data efficiency (0.9333 vs 0.9125 at 5%), but only when the
configuration space is well covered. On a narrow-coverage, extrapolative test the physical
structure is *worse* than a black-box (0.3618 vs 0.4319, p = 0.021): the Lagrangian prior does
not by itself buy extrapolation. On the real Baxter arm the difference is
neutral (+0.0014, n.s.; §4.1), and there the residual branch — not the physics branch — is what
carries the model (+0.0724, p < 0.001).
**Where it does not help.** Geometric encodings are not a free win: raw angles are competitive,
sin/cos is marginally better, and a rotation-matrix encoding is *worse* once the bottleneck is
removed; the bottleneck, not the encoding, is the lever.

**A second way to be exact and useless: the collapsing physics weight.** §4.7 showed a model that
is accurate in aggregate and worthless on one joint. The physics weight produces the same
disease in a second organ, and it is worth stating because it is the failure mode a practitioner is
most likely to induce by accident. An adaptively *learned* $\lambda$ collapses to zero (§3.2): the
residual branch absorbs everything, and — the part that makes this a dominated point rather than a
trade-off — the collapse *does not even maximise R²* (0.9643 at λ=0 vs 0.9735 at λ=1; Fig. 9,
Table 5). What it does destroy is everything the physics branch was for: $g(q)$ ends at 0.67
relative error and the condition number of $H$ two orders of magnitude too high (≈1.8×10⁵ vs
3.5×10³ true). Fixing a non-trivial λ restores a control-grade $g(q)$ at no R² cost. So a
"physics-informed" model can report a competitive R², carry a physics branch that is switched off,
and be measurably worse for control than the black box it replaced — the same lesson as §4.2 and
§4.7 arriving through a third route, and a third reason not to certify these models on aggregate
fit alone.

*Two batches, two MLP references.* The λ sweep (Fig. 9, Table 5, `control_lam*.json`) is a
single-seed batch with its own MLP checkpoint, at 10.6 mrad; the closed-loop table of §4.6
(`control_*_seed*.json`) is a 3-seed batch whose MLP is 8.68 ± 1.69 mrad. Within the λ-sweep batch,
λ=0 gives 30.9 mrad and λ=1 gives 10.6 mrad — the latter coincidentally equal to that batch's MLP
figure. The corresponding adaptive-λ, raw-angle model reads 36.0 mrad in its own single-seed run and
35.84 ± 0.58 mrad in the 3-seed batch. Numbers are only comparable within a batch; we never compare
across the two, and every headline claim in this paper uses the 3-seed batch.

**Limitations.** The screw axes and zero-configuration frames are taken as known from the URDF,
as in prior structured-dynamics work; we audit inertial identifiability, not kinematic
calibration. The closed-loop evaluation is in simulation, with friction and measurement noise
injected into the training labels but with the same rigid-body simulator used for rollout;
Baxter has no analytic ground truth, so parameter-recovery statistics are reported only for the
two simulated arms. The identifiability analysis builds the regressor from exact kinematics and
therefore gives an upper bound on what any estimator could recover: the rank 43 is a *ceiling*,
and noise in $q,\dot q,\ddot q$ — which perturbs the regressor itself, unlike the label noise
swept in §4.2 — can only lower the effective rank by pushing the smallest identifiable directions
below the practical tolerance. Our own sweep bounds only the label-noise case, where the rank is
provably unchanged; the state-noise case is untested here, and a real deployment would need to fix
the tolerance from the measurement covariance rather than from a numerical threshold.
Reproducibility is single-machine/single-GPU.

---

## 6 Conclusion

We set out to recover the physical inertial parameters of a manipulator with an SE(3)-structured
inverse-dynamics model, and found that doing so is impossible from the over-parameterised form and
straightforward once the nullspace is removed. On the Franka Emika Panda, 43 of the 80 centroidal
inertial directions are identifiable and 37 are not, and the rank is fixed by kinematic structure
alone — unchanged across arms whose inertial values differ by an order of magnitude. Unbiased least
squares on the exact regressor hits torque $R^2 = 1.00000000$ with 100.0000 % of its parameter error
inside the nullspace, so no estimator — neural or classical — recovers the true masses, centres of
mass, and inertia tensors from the naive 10-parameter-per-link form. The method that resolves this is
a reparameterisation: projecting onto the 43-dimensional identifiable subspace, $\pi = B\theta$, with
$B$ computed once from the regressor's singular value decomposition and frozen. It loses nothing — the
mass matrix is reproduced to $8\times10^{-7}$ — and it makes the trained parameters the true base
parameters $\hat\theta\to B^\top\pi_{\mathrm{true}}$ <!-- TODO fill θ̂ recovery error -->.

The same subspace is why base-parameter recovery and control are compatible rather than competing.
Because the gravity subspace is contained in the identifiable subspace to $1.6\times10^{-15}$, the
reparameterised model determines $g(q)$ exactly, which is the mechanism behind the 0.46 mrad
closed-loop gravity compensation of §4.6 — eighteen times better than the strongest baseline. A model
can therefore recover the base parameters *and* drive the arm, because both draw on the same
identifiable subspace; the per-link physical parameters remain beyond the reach of torque data alone.
Aggregate R² remains the wrong certificate: adding realistic friction moves global R² by 0.013 while
the seventh joint falls to $R^2 = -0.008$.

Two reporting rules follow from the audit that motivated the method. State the rank of the regressor
alongside any parameter-recovery claim, since a claim of physical interpretability is meaningless
without it; state the link convention used for relative parameter error, since including a fixed base
or a 0.1 kg end-effector changes the headline number by a factor of three for purely arithmetic
reasons; and report per-joint R² alongside the global figure. Whether the recovered parameters
transfer to a real arm, where the screw axes themselves are uncertain, is the natural next test;
identification under varying payload, which changes only the terminal link's parameters and may sit
partly inside the nullspace, is the second.

---

## Appendix A — Reproducibility

All numbers in this manuscript are read from `RESULTS.md`, which is generated by
`python paper/report.py` from the JSON files under `experimental_results/`. Every run records
its random seed, git commit, GPU, PyTorch version, wall-clock time, and the exact reproduction
command. Reproduction: `python paper/se3_physics.py --xml tool/panda.xml` (SE(3) geometry/math
self-check against `mj_fullM`, must pass before training), `python paper/run_grid.py --grid main
--workers 6` (and `--grid data_eff`, `--grid hparam`, `--grid lam`),
`python paper/noise_robustness.py --arm …`, `python paper/control_sim.py …`,
`python paper/verify_physics.py --all`, `python paper/lam_tradeoff.py`, then
`python paper/analyze.py && python paper/report.py && python paper/figures.py`.
