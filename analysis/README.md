# Analyses for the project questions (Burgers')

`run_all.jl` runs the conditioning, accuracy and stability studies from `../docs/project_plan.html`
on the default Burgers' setup (ν = 0.02, N = 201). It writes figures and every table
(`results.txt`) to `output/`.

```bash
julia analysis/run_all.jl      # ≈ 2 min, mostly Plots compiling
```

The model is trained on t ∈ [0, 2], the same data as `../burgers/output/burgers_data.jld2`. The full-order model
is also run to t = 4, so "prediction" below means 2 < t ≤ 4. `common.jl` holds the shared pieces:
- data, POD basis and the data matrix D (same `compact_kron` ordering as `../burgers/opInf.jl`);
- intrusive Galerkin operators;
- three least-squares solvers;
- RK4 for the ROM with a blow-up guard;
- the stability-based choice of λ.

## Results

| # | Question | Finding | Figures |
|---|---|---|---|
| Q1 | Recovery | With re-projected data, OpInf recovers the Galerkin operators to ~1e-14 at r = 2–3, degrading to 1e-5 by r = 6 as κ(D) grows; beyond that, rounding amplified by κ(D) takes over. Projected FOM derivatives give operators that differ from Galerkin by O(1) or more, yet the ROMs still track the data at small r. | `q1_*` |
| Q2 | Conditioning | κ(D) reaches 1e13 at r = 6 and 2e29 at r = 18. That's far more than (σ₁/σᵣ)², and column scaling helps only for r ≲ 5. The cause is that one trajectory makes all columns smooth functions of t, so they are nearly dependent. | `q2_*` |
| Q3 | Solvers | QR and SVD errors follow ε·κ(D). The normal equations lose about twice the digits and Cholesky fails for κ(D) ≳ 1e13 (r ≥ 6). All three are fast at this size. | `q3_*` |
| Q4 | Choosing λ (r = 15) | Stability-based λ = 1.8e-2 gives 7.7e-4 training / 4.1e-2 prediction error. The L-curve corner (1.8e-3) is close. GCV picks the smallest λ on the grid and predicts badly (4.7). The learned Â has eigenvalues ≈ 0, against −0.20 for Galerkin. | `q4_*` |
| Q5 | Accuracy, noise | With one trajectory, OpInf levels off at ~1e-3 training / 4e-2 prediction for r ≥ 6, while Galerkin keeps improving. Noise up to 1e-4 doesn't matter; at 1e-3 and above, prediction error rises to 0.1–0.2. | `q5_*` |
| Q6 | Fix: 4 trajectories | κ(D) drops by a factor of 1e3–1e11. Re-projected recovery holds to 5e-10 at r = 10. Prediction error at r = 15 falls from 4e-2 to 1e-5. | `q6_*` |

Main takeaway: the bottleneck is the **information content of the training data**, not the solver.
A single trajectory makes D numerically rank-deficient no matter how it is solved or regularized.
Several initial conditions fix that.

## Theory: accuracy of the least-squares solve (skeleton, to be filled in)

Goal: explain the Q3 results, where QR and SVD errors follow $\varepsilon\,\kappa(D)$, the normal
equations do worse and then fail, and nothing recovers the operators once $\kappa(D) \gtrsim 1/\varepsilon$.
The argument is **forward error ≲ condition number × backward error**.
References are to Trefethen & Bau, *Numerical Linear Algebra* (TB).

### Setup and notation

- $D \in \mathbb{R}^{K \times p}$ is the data matrix, with $K$ snapshots and $p = r + r(r+1)/2$
  columns. Assume $K \ge p$ and full column rank.
- The OpInf problem $\min_O \|D O^\top - R\|_F$ splits into $r$ independent problems
  $\min_x \|Dx - b\|_2$, one per row of $O$ (one per column of $R$). It's enough to analyze one.
- $\kappa = \kappa(D) = \sigma_{\max}(D)/\sigma_{\min}(D)$, and $\varepsilon \approx 1.1\times10^{-16}$ is the unit roundoff.
- $\theta$ is the angle between $b$ and $\operatorname{range}(D)$, and $\eta = \|D\|\,\|x\| / \|Dx\|$.

> TODO: state the full-rank assumption precisely, and note what it means in the experiments.
> Q2 shows $\kappa(D) > 1/\varepsilon$ for $r \ge 7$, so $D$ is *numerically* rank-deficient
> there, and the theorems below only describe $r \le 6$.

### Claim 1: conditioning of least squares (TB Lecture 18, Theorem 18.1)

**Statement.** The relative condition number of $x$ is
$\kappa_{b\to x} = \dfrac{\kappa}{\eta\cos\theta}$ for perturbations of $b$, and
$\kappa_{D\to x} = \kappa + \dfrac{\kappa^2\tan\theta}{\eta}$ for perturbations of $D$.

**Proof outline.**
1. > TODO: write $x = D^+ b$ and reduce to the SVD of $D$.
2. > TODO: perturbations of $b$: bound $\|\delta x\|/\|x\|$ in terms of $\|\delta b\|/\|b\|$.
3. > TODO: perturbations of $D$: split $\delta D$ into the part that tilts $\operatorname{range}(D)$
   > and the part that doesn't. The tilt is where the $\kappa^2 \tan\theta$ term comes from.

### Claim 2: the consistent case used in Q3

**Statement.** If $b \in \operatorname{range}(D)$, so $\theta = 0$, then $\kappa_{D\to x} = \kappa$
and $\kappa_{b\to x} = \kappa/\eta \le \kappa$.

**Proof outline.**
1. > TODO: show the re-projected right-hand side $R = (V^\top f(V\hat Q))^\top$ lies exactly in
   > $\operatorname{range}(D)$, because the reduced Galerkin model is exactly quadratic in $\hat q$.
2. > TODO: substitute $\theta = 0$ into Claim 1.
3. > TODO: one sentence on why this is the right test problem: the $\kappa^2$ term vanishes, so any
   > $\kappa^2$ behavior that remains must come from the *algorithm*.

### Claim 3: Householder QR is backward stable for least squares (TB Lecture 19)

**Statement.** The computed solution $\tilde x$ is the exact least-squares solution for a nearby
problem: $\tilde x = \arg\min_x \|(D+\delta D)x - b\|$ with $\|\delta D\|/\|D\| = O(\varepsilon)$.

**Proof outline.**
1. > TODO: backward stability of Householder QR (TB Lecture 16).
2. > TODO: computing $Q^\top b$ implicitly, by applying the reflectors.
3. > TODO: backward stability of back substitution (TB Lecture 17).
4. > TODO: combine the three. Check the exact theorem numbers in TB.

The SVD-based solve is also backward stable (TB Lecture 19).

> TODO: state that result, or cite it.

### Claim 4: forward error of QR and SVD

**Statement.** In the consistent case,
$\dfrac{\|\tilde x - x\|}{\|x\|} = O(\kappa\,\varepsilon)$.

**Proof outline.**
1. > TODO: forward error ≤ condition number × backward error (TB Lecture 15).
2. > TODO: plug in Claims 2 and 3.

**Numerical check** (`output/results.txt`, Q3; `output/q3_solver_accuracy.png`):

| r | κ(D) | QR error | SVD error | κε |
|---|---|---|---|---|
| 2 | 1.1e3 | 6.9e-15 | 8.3e-15 | 1.2e-13 |
| 4 | 8.4e6 | 1.9e-11 | 9.6e-12 | 9.3e-10 |
| 6 | 1.3e13 | 8.7e-6 | 7.8e-6 | 1.4e-3 |

> TODO: comment on the constant hidden in $O(\cdot)$. The errors sit 1–2 orders of magnitude
> below $\kappa\varepsilon$, so the bound holds but isn't tight.

### Claim 5: the normal equations square the condition number

**Statement.** (a) $\kappa(D^\top D) = \kappa(D)^2$.
(b) Solving $D^\top D x = D^\top b$ by Cholesky gives forward error $O(\kappa^2\varepsilon)$,
even in the consistent case, so it is **not** backward stable for the least-squares problem.
(c) Cholesky can break down once $\kappa^2 \varepsilon \gtrsim 1$, i.e. $\kappa \gtrsim 10^8$.

**Proof outline.**
1. > TODO (a): one line from the SVD $D = U\Sigma V^\top$.
2. > TODO (b): Cholesky is backward stable for $D^\top D$ (TB Lecture 23), and that system has
   > condition number $\kappa^2$. Explain why this is only as good as QR when $\tan\theta$ is large, since then Claim 1's $\kappa^2\tan\theta$ term dominates anyway.
3. > TODO (c): rounding when *forming* $D^\top D$ is of size $\varepsilon\|D\|^2$, comparable to
   > $\sigma_{\min}^2$ once $\kappa^2\varepsilon \sim 1$. Then the computed matrix need not be
   > positive definite.

**Numerical check:** error 1.4e-6 at κ = 8.4e6 (r = 4), versus κ²ε ≈ 8e-3; error 2.4 at κ = 1.1e10
(r = 5); Cholesky failed for every r ≥ 6 (κ ≥ 1.3e13).

> TODO: note that (c) says breakdown *can* happen past 1e8, not that it must. It survived at 1e10.

### Claim 6: Tikhonov regularization bounds the conditioning

**Statement.** $\min_x \|Dx-b\|^2 + \lambda\|x\|^2$ is ordinary least squares with
$D_\lambda = \begin{bmatrix} D \\ \sqrt\lambda\, I\end{bmatrix}$, whose singular values are
$\sqrt{\sigma_i^2 + \lambda}$, so
$\kappa(D_\lambda) = \sqrt{\dfrac{\sigma_{\max}^2+\lambda}{\sigma_{\min}^2+\lambda}} \le \sqrt{1 + \sigma_{\max}^2/\lambda}$.

**Proof outline.**
1. > TODO: expand $\|D_\lambda x - [b;0]\|^2$.
2. > TODO: $D_\lambda^\top D_\lambda = D^\top D + \lambda I$; read off the singular values.
3. > TODO: the bias. Using the SVD filter factors $\sigma_i^2/(\sigma_i^2+\lambda)$, show the
   > regularized solution differs from the true $x$ in exactly the directions with small $\sigma_i$.
   > That's the trade-off in Q4: bounded conditioning in exchange for biased operators.

### Claim 7 (discussion, not a theorem): why one trajectory makes κ(D) huge

> TODO: argue that along one trajectory every column of $D$ is a smooth function of $t$ sampled at
> $K$ points, so the $p$ columns are close to linearly dependent, as in a Vandermonde matrix.
> Contrast with Q6, where 4 trajectories cut $\kappa(D)$ by $10^3$–$10^{11}$. Conclusion to defend:
> past $\kappa \approx 1/\varepsilon$, Claims 3–4 say *no* backward-stable solver can recover the
> operators, so the fix has to come from the data, not the algorithm.

## Caveats

- Noise results are the median of 3 random seeds, so treat them as rough.
- The stability criterion is a simplified version of McQuarrie, Huang & Willcox (2021). It picks
  the λ with the lowest training error among those whose ROM stays within 1.5·max|Q̂_train| up to
  t = 4, and uses only training data.
- A single λ penalizes the linear and quadratic blocks equally. Separate λ₁, λ₂ (as McQuarrie et
  al. use) is a natural next experiment.
- At r = 18 with 4 trajectories, the prediction error jumps back to 4e-2. I haven't investigated
  why.
