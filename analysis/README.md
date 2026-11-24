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

| #  | Question             | Finding                                                                                                                                                                                                                                                                                                                | Figures  |
| -- | -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- |
| Q1 | Recovery             | With re-projected data, OpInf recovers the Galerkin operators to ~1e-14 at r = 2–3, degrading to 1e-5 by r = 6 as κ(D) grows; beyond that, rounding amplified by κ(D) takes over. Projected FOM derivatives give operators that differ from Galerkin by O(1) or more, yet the ROMs still track the data at small r. | `q1_*` |
| Q2 | Conditioning         | κ(D) reaches 1e13 at r = 6 and 2e29 at r = 18. That's far more than (σ₁/σᵣ)², and column scaling helps only for r ≲ 5. The cause is that one trajectory makes all columns smooth functions of t, so they are nearly dependent.                                                                                  | `q2_*` |
| Q3 | Solvers              | QR and SVD errors follow ε·κ(D). The normal equations lose about twice the digits and Cholesky fails for κ(D) ≳ 1e13 (r ≥ 6). All three are fast at this size.                                                                                                                                                   | `q3_*` |
| Q4 | Choosing λ (r = 15) | Stability-based λ = 1.8e-2 gives 7.7e-4 training / 4.1e-2 prediction error. The L-curve corner (1.8e-3) is close. GCV picks the smallest λ on the grid and predicts badly (4.7). The learned Â has eigenvalues ≈ 0, against −0.20 for Galerkin.                                                                   | `q4_*` |
| Q5 | Accuracy, noise      | With one trajectory, OpInf levels off at ~1e-3 training / 4e-2 prediction for r ≥ 6, while Galerkin keeps improving. Noise up to 1e-4 doesn't matter; at 1e-3 and above, prediction error rises to 0.1–0.2.                                                                                                          | `q5_*` |
| Q6 | Fix: 4 trajectories  | κ(D) drops by a factor of 1e3–1e11. Re-projected recovery holds to 5e-10 at r = 10. Prediction error at r = 15 falls from 4e-2 to 1e-5.                                                                                                                                                                              | `q6_*` |

Main takeaway: the bottleneck is the **information content of the training data**, not the solver.
A single trajectory makes D numerically rank-deficient no matter how it is solved or regularized.
Several initial conditions fix that.

## Theory: conditioning, stability and accuracy of OpInf (skeleton, to be filled in)

Goal: explain how OpInf's error depends on the condition numbers in the problem, which solvers
are stable, and where the final ROM error comes from. Standard least-squares results are
**quoted from Trefethen & Bau (TB) without proof** in the toolbox below. The TODOs are the parts that
are specific to OpInf.

### Setup and notation

- $D \in \mathbb{R}^{K \times p}$ is the data matrix, with $K$ snapshots and $p = r + r(r+1)/2$
  columns. Assume $K \ge p$ and full column rank.
- The OpInf problem $\min_O \|D O^\top - R\|_F$ splits into $r$ independent problems
  $\min_x \|Dx - b\|_2$, one per row of $O$ (one per column of $R$). It's enough to analyze one.
- $\kappa = \kappa(D) = \sigma_{\max}(D)/\sigma_{\min}(D)$, and $\varepsilon \approx 1.1\times10^{-16}$ is the unit roundoff.
- $\theta$ is the angle between $b$ and $\operatorname{range}(D)$, and $\eta = \|D\|\,\|x\| / \|Dx\|$.
- With the reduced SVD $D = U\Sigma V^\top$:
  $x = D^+ b = \sum_{i=1}^p \frac{u_i^\top b}{\sigma_i}\, v_i$, and the fit $Dx = UU^\top b$ is the
  orthogonal projection of $b$ onto $\operatorname{range}(D)$.

**Full-rank assumption.** We assume $\operatorname{rank}(D) = p$, or equivalently $\sigma_{\min}(D) = \sigma_p > 0$,
or equivalently $D^\top D$ is symmetric positive definite. This requires $K \ge p$. Under it, the
least-squares problem has a unique minimizer and $\kappa < \infty$.
In the experiments $K = 201$ and $p$ runs from 2 to 189, so $K \ge p$ always holds (only just at $r = 18$).

### Toolbox (quoted from TB)

|    | Result                                                                                                                                               | TB                 |
| -- | ---------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------ |
| T1 | Conditioning of least squares: $\kappa_{b\to x} = \dfrac{\kappa}{\eta\cos\theta}$, $\kappa_{D\to x} = \kappa + \dfrac{\kappa^2\tan\theta}{\eta}$ | Thm 18.1           |
| T2 | Householder QR and the SVD solve are backward stable for least squares                                                                               | Thm 19.1 and 19.4 |
| T3 | Forward error ≲ condition number × backward error                                                                                                  | Thm 15.1           |
| T4 | $\kappa(D^\top D) = \kappa^2$; the normal equations (via Cholesky) are not backward stable for least squares                                       | Thm 19.3           |
| T5 | The distance from $D$ to the nearest rank-deficient matrix is $\sigma_p$                                                                         | Thm 5.8            |

Combining T1–T4, for a backward-stable solver

$$
\frac{\|\tilde x - x\|}{\|x\|} \lesssim \Big(\kappa + \frac{\kappa^2\tan\theta}{\eta}\Big)\varepsilon,
\qquad\text{and for the normal equations}\qquad \frac{\|\tilde x - x\|}{\|x\|} \lesssim \kappa^2\varepsilon.
$$

Everything below is about what $\kappa$, $\theta$ and the solver choice turn out to be **for OpInf**.

---

### 1. Conditioning: what κ(D) depends on

**1a. Numerical rank.** In floating point, exact rank is the wrong test. By T5, $D$ is within
$\sigma_p$ of a rank-deficient matrix, and rounding the entries of $D$ already moves it by about
$\varepsilon\sigma_1$. So once $\kappa \gtrsim 1/\varepsilon \approx 10^{16}$, $D$ is
**numerically rank-deficient**: the operators are not determined by the data and the bound above
gives relative errors of order 1. What we need is $\kappa\varepsilon \ll 1$.

Q2 shows where that line falls: $\kappa(D) = 1.3\times10^{13}$ at $r = 6$ ($\kappa\varepsilon \approx 1.4\times10^{-3}$),
and $2.6\times10^{17} > 1/\varepsilon$ at $r = 7$. Q1 agrees: re-projected recovery error is
$8.7\times10^{-6}$ at $r = 6$ and jumps to $0.16$ at $r = 7$.

**1b. Dependence on r.** Q2 (`output/results.txt`):

| r  | p  | κ(D)  | σ₁/σᵣ of snapshots | (σ₁/σᵣ)² | κ(D), columns scaled |
| -- | -- | ------ | ---------------------- | ------------- | --------------------- |
| 2  | 5  | 1.1e3  | 5.5                    | 3.1e1         | 7.3e1                 |
| 4  | 14 | 8.4e6  | 1.0e2                  | 1.0e4         | 1.4e5                 |
| 6  | 27 | 1.3e13 | 1.7e3                  | 2.9e6         | 2.6e11                |
| 8  | 44 | 2.0e18 | 3.3e4                  | 1.1e9         | 2.7e15                |
| 10 | 65 | 1.9e20 | 6.5e5                  | 4.3e11        | 9.0e15                |

> TODO: explain why κ(D) grows faster than $(\sigma_1/\sigma_r)^2$. The quadratic columns
> $\hat q_i \hat q_j$ have scale $\sim\sigma_i\sigma_j$, which accounts for $(\sigma_1/\sigma_r)^2$.
> The rest must come from near-dependence between columns (1c).
>
> TODO: column scaling (last column) removes the scale disparity but not the near-dependence,
> which is why it only helps for $r \lesssim 5$. Say so with the numbers.

**1c. Dependence on the training data (main new claim).** Q6, one vs. four trajectories:

| r  | κ(D), 1 trajectory | κ(D), 4 trajectories |
| -- | ------------------- | --------------------- |
| 4  | 8.4e6               | 6.5e3                 |
| 6  | 1.3e13              | 1.3e5                 |
| 8  | 2.0e18              | 1.8e7                 |
| 10 | 1.9e20              | 2.4e9                 |

> TODO: argue that along one trajectory every column of $D$ is a smooth function of $t$ sampled
> at $K$ points, so the $p$ columns are nearly linearly dependent, as in a Vandermonde matrix.
> More initial conditions add directions in state space that one curve $\hat q(t)$ can't reach.
>
> Conclusion to defend: once $\kappa \gtrsim 1/\varepsilon$, T2–T3 say *no* backward-stable
> solver can recover the operators, so the fix has to come from the data, not the algorithm.

**1d. The angle θ: re-projected vs. projected derivatives.** T1 has a $\kappa^2\tan\theta$ term,
so the right-hand side matters as much as $D$.

> TODO: show the re-projected right-hand side $R = (V^\top f(V\hat Q))^\top$ lies exactly in
> $\operatorname{range}(D)$ (θ = 0), because the reduced Galerkin model is exactly quadratic in $\hat q$.
> Then only the $\kappa$ term survives, and any $\kappa^2$ behavior must come from the algorithm.
>
> TODO: with projected FOM derivatives $V^\top f(u)$, the residual is the closure error
> $V^\top f(u) - V^\top f(VV^\top u)$, so θ > 0 and the $\kappa^2\tan\theta$ term is active.
> This explains why projected data give operators O(1) away from Galerkin (Q1).
>
> TODO (number to compute): $\tan\theta$ for the projected data, as $\|b - Dx\|/\|Dx\|$, for a few r.

**1e. Regularization bounds the conditioning.** Tikhonov,
$\min_x \|Dx-b\|^2 + \lambda\|x\|^2$, is ordinary least squares with
$D_\lambda = \begin{bmatrix} D \\ \sqrt\lambda\, I\end{bmatrix}$. Since
$D_\lambda^\top D_\lambda = D^\top D + \lambda I$, its singular values are $\sqrt{\sigma_i^2+\lambda}$ and

$$
\kappa(D_\lambda) = \sqrt{\frac{\sigma_{\max}^2+\lambda}{\sigma_{\min}^2+\lambda}} \le \sqrt{1 + \sigma_{\max}^2/\lambda}.
$$

So λ caps the conditioning regardless of r or the data. The price is bias; see 3b.

---

### 2. Stability

There are two separate meanings here; keep them apart in the report.

**2a. Algorithmic stability of the least-squares solve.** By T2 and T4, QR and SVD are
backward stable and the normal equations aren't. Q3 tests this on the consistent (θ = 0) problem,
where the exact answer is the Galerkin operators:

| r | κ(D)  | normal eq.     | QR      | SVD     | κε    | κ²ε  |
| - | ------ | -------------- | ------- | ------- | ------- | ------- |
| 2 | 1.1e3  | 9.7e-13        | 6.9e-15 | 8.3e-15 | 1.2e-13 | 1.4e-10 |
| 3 | 5.0e4  | 2.3e-10        | 1.3e-13 | 1.4e-13 | 5.5e-12 | 2.7e-7  |
| 4 | 8.4e6  | 1.4e-6         | 1.9e-11 | 9.6e-12 | 9.3e-10 | 7.8e-3  |
| 5 | 1.1e10 | 2.4            | 1.8e-8  | 5.9e-9  | 1.2e-6  | 1.2e4   |
| 6 | 1.3e13 | Cholesky fails | 8.7e-6  | 7.8e-6  | 1.4e-3  | —      |

> TODO: QR/SVD errors sit 1–2 orders below κε and normal-equation errors below κ²ε, so both bounds
> hold but aren't tight. Comment on the constant.
>
> TODO: explain the Cholesky breakdown. Forming $D^\top D$ in floating point commits errors of size
> $\varepsilon\|D\|^2 = \varepsilon\sigma_1^2$, comparable to $\sigma_p^2$ once $\kappa^2\varepsilon \sim 1$,
> so the computed matrix need not be positive definite. That threshold is $\kappa \approx 10^8$;
> it survived to $10^{10}$ here (*can* break, not *must*).
>
> Consequence for the project: `../burgers/opInf.jl` uses the normal equations, and at r = 15 it is
> outside the range where they are reliable.

**2b. Stability of the learned ROM (dynamical).** A small regression residual does not make
$\dot{\hat q} = \hat A\hat q + \hat H(\hat q\otimes\hat q)$ stable. Q4 (r = 15):

| λ choice       | λ     | train err | predict err | max Re λ(Â) |
| --------------- | ------ | --------- | ----------- | ------------- |
| L-curve         | 1.8e-3 | 9.3e-4    | 4.5e-2      | −2.6e-10     |
| GCV             | 1e-12  | 4.4e-3    | 4.7         | −1.9e-2      |
| stability-based | 1.8e-2 | 7.7e-4    | 4.1e-2      | −5.8e-11     |
| Galerkin        | —     | —        | —          | −0.197       |

43 of the 57 λ values give a ROM that stays bounded to t = 4.

> TODO: explain why the learned $\hat A$ has eigenvalues at ≈ 0 while Galerkin's are at −0.2.
> Hypothesis: Tikhonov shrinks the small-σ directions of $x$ toward 0 (filter factors, 3b), which
> includes the dissipative part of $\hat A$. Check whether the ROM's boundedness then comes from
> $\hat H$ (energy-preserving quadratic term) rather than from $\hat A$.
>
> TODO: the GCV row is the cautionary example: the least-regularized model has the most negative
> learned eigenvalue yet blows up in prediction. Linear stability of Â alone does not decide it.

---

### 3. Accuracy

**3a. Error budget.** The ROM error compared with the full model splits into:

1. POD truncation (the "POD floor": best possible in the r-dim basis);
2. closure error, from θ > 0 with projected derivatives (1d);
3. finite-difference error in $\dot{\hat Q}$ (O(Δt²) for central differences);
4. regularization bias (3b);
5. rounding in the solve, ≲ κε for QR/SVD (2a).

> TODO: for each term, say which experiment isolates it and how big it is. Q5a has the data:

| r  | POD floor (train/pred) | Galerkin         | OpInf, exact deriv. | OpInf, FD deriv. |
| -- | ---------------------- | ---------------- | ------------------- | ---------------- |
| 4  | 2.1e-3 / 8.6e-3        | 4.0e-3 / 1.8e-2  | 2.1e-3 / 4.7e-2     | 2.1e-3 / 4.7e-2  |
| 6  | 1.2e-4 / 6.8e-4        | 2.1e-4 / 1.4e-3  | 8.7e-4 / 4.0e-2     | 9.7e-4 / 3.9e-2  |
| 10 | 3.8e-7 / 2.3e-6        | 7.0e-7 / 4.2e-6  | 7.7e-4 / 4.1e-2     | 1.1e-3 / 3.9e-2  |
| 15 | 3.8e-10 / 1.4e-9       | 6.5e-10 / 2.7e-9 | 7.7e-4 / 4.1e-2     | 1.1e-3 / 3.9e-2  |

> TODO: the OpInf columns are *identical* for r ≥ 8 while the POD floor keeps dropping. So the
> plateau isn't POD truncation, and exact vs. FD derivatives barely matter, so it isn't term 3
> either. Argue it is terms 2 and 4: regularization bias plus closure error.

**3b. Regularization bias.** With the SVD of $D$, the Tikhonov solution is

$$
x_\lambda = \sum_{i=1}^p \frac{\sigma_i^2}{\sigma_i^2+\lambda}\,\frac{u_i^\top b}{\sigma_i}\,v_i ,
$$

so it agrees with $x$ in directions with $\sigma_i^2 \gg \lambda$ and is shrunk to zero in directions
with $\sigma_i^2 \ll \lambda$.

> TODO: write $x - x_\lambda$ from this and bound it. Then connect: with one trajectory most
> $\sigma_i$ are tiny (1c), so a λ large enough to stabilize the ROM removes most of the operator.
> This is the trade-off: bounded κ(D_λ) (1e) in exchange for biased operators.

**3c. Why GCV fails here.** GCV assumes the error in $b$ is random white noise. Here it is the
closure error (1d), which is deterministic and lies along the trajectory.

> TODO: one paragraph. GCV has no interior minimum (it picks λ = 1e-12, the grid edge), and that
> model predicts with error 4.7.

**3d. Data fix.** With 4 trajectories (Q6), r = 15 prediction error drops from 4e-2 to 1e-5,
and re-projected recovery holds to 5e-10 at r = 10.

> TODO: tie back to 1c: smaller κ(D) means a smaller λ suffices, so less bias (3b), so the
> plateau in 3a moves down.

## Caveats

- Noise results are the median of 3 random seeds, so treat them as rough.
- The stability criterion is a simplified version of McQuarrie, Huang & Willcox (2021). It picks
  the λ with the lowest training error among those whose ROM stays within 1.5·max|Q̂_train| up to
  t = 4, and uses only training data.
- A single λ penalizes the linear and quadratic blocks equally. Separate λ₁, λ₂ (as McQuarrie et
  al. use) is a natural next experiment.
- At r = 18 with 4 trajectories, the prediction error jumps back to 4e-2. I haven't investigated
  why.
