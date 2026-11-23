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

## Caveats

- Noise results are the median of 3 random seeds, so treat them as rough.
- The stability criterion is a simplified version of McQuarrie, Huang & Willcox (2021). It picks
  the λ with the lowest training error among those whose ROM stays within 1.5·max|Q̂_train| up to
  t = 4, and uses only training data.
- A single λ penalizes the linear and quadratic blocks equally. Separate λ₁, λ₂ (as McQuarrie et
  al. use) is a natural next experiment.
- At r = 18 with 4 trajectories, the prediction error jumps back to 4e-2. I haven't investigated
  why.
