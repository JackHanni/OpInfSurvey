# OpInf on the Einstein field equations: Gowdy T³ cosmologies

This is an exploratory side study for the Burgers' OpInf project. I tried Operator Inference on
solutions of the vacuum Einstein equations G_μν = 0 in the
**T³ Gowdy** symmetry class. Part A uses exact solutions. Part B uses data from my own solver,
checked against an exact nonlinear solution.

## Why Gowdy T³ (and not SXS or Choptuik)

OpInf/POD needs a *field* on a spatial grid that evolves in time. Gowdy T³ spacetimes fit this
well:

* They are genuinely inhomogeneous, dynamical vacuum Einstein solutions. The metric depends
  on (τ, θ), and the field equations reduce *exactly* to two coupled 1+1 wave equations
  for P(τ, θ) and Q(τ, θ) on a periodic circle. Nothing is approximated physically.
* **Polarized case (Q ≡ 0):** the equation is linear and the general solution is known in
  closed form (Bessel functions). This gives maximally reliable data.
* **Unpolarized case:** the coupling term e^{2P} is non-polynomial, which is exactly the
  "needs lifting" situation in `../docs/project_plan.html`. There is also an exact nonlinear
  ("pseudo-unpolarized") solution that a solver can be checked against.

I rejected the alternatives for these reasons:
* **SXS waveforms** are a handful of mode time series h_ℓm(t), not a field, so POD has
  little to act on.
* **Choptuik collapse** has no external ground truth. It would rely entirely on self-validation,
  and its constraint ODEs in r add machinery without changing the NLA question.

## Sources (DOIs checked to resolve, metadata from Crossref)

* R. H. Gowdy, *Vacuum spacetimes with two-parameter spacelike isometry groups and compact
  invariant hypersurfaces*, Ann. Phys. 83, 203–241 (1974). doi:[10.1016/0003-4916(74)90384-4](https://doi.org/10.1016/0003-4916(74)90384-4)
* B. K. Berger & V. Moncrief, *Numerical investigation of cosmological singularities*,
  Phys. Rev. D 48, 4676–4687 (1993). doi:[10.1103/PhysRevD.48.4676](https://doi.org/10.1103/PhysRevD.48.4676),
  arXiv:gr-qc/9307032. **All equations below come from its Sec. III (eqs. 3.1–3.5, 3.18–3.20) and Sec. IV (eq. 4.2).**
* B. K. Berger, D. Garfinkle & V. Moncrief, *Numerical study of cosmological singularities*,
  arXiv:gr-qc/9709073 (1997). Same field equations (their eqs. 15–19); used as a cross-check.
* B. K. Berger, *Numerical approaches to spacetime singularities*, Living Rev. Relativ. 5, 1
  (2002). doi:[10.12942/lrr-2002-1](https://doi.org/10.12942/lrr-2002-1). Review; cited for context only (I did not re-derive anything from it).
* B. Peherstorfer & K. Willcox, *Data-driven operator inference for nonintrusive projection-based
  model reduction*, CMAME 306, 196–215 (2016). doi:[10.1016/j.cma.2016.03.025](https://doi.org/10.1016/j.cma.2016.03.025)
* E. Qian, B. Kramer, B. Peherstorfer & K. Willcox, *Lift & Learn*, Physica D 406, 132401 (2020).
  doi:[10.1016/j.physd.2020.132401](https://doi.org/10.1016/j.physd.2020.132401)

## Equations

Metric (B&M eq. 3.1): ds² = e^{λ/2}e^{τ/2}(−e^{−2τ}dτ² + dθ²) + e^{−τ}[e^P dσ² + 2e^P Q dσdδ + (e^P Q² + e^{−P})dδ²].
Here θ ∈ [0, 2π) is periodic and the singularity is at τ → ∞. The evolution equations (eqs. 3.2–3.3) are

    P_ττ − e^{−2τ} P_θθ − e^{2P}(Q_τ² − e^{−2τ} Q_θ²) = 0
    Q_ττ − e^{−2τ} Q_θθ + 2(P_τ Q_τ − e^{−2τ} P_θ Q_θ) = 0

λ is then fixed by quadrature from the constraints (eqs. 3.4–3.5). On T³ the momentum constraint
requires ∮(P_θ P_τ + e^{2P} Q_θ Q_τ) dθ = 0, and I use this as a check.

With the areal time t = e^{−τ} these become
P_tt + P_t/t − P_θθ = e^{2P}(Q_t² − Q_θ²), and similarly for Q.

**Non-autonomy.** Every form has explicitly time-dependent coefficients: 1/t in t, or e^{−2τ} in τ.
I did *not* hide this with a time change. Instead the coefficient is treated as a known scalar
**input** u(t), and the ROM has the bilinear form

    dx̂/dt = Â x̂ + u(t) N̂ x̂ + Ĥ (x̂ ⊗ x̂)        (Ĥ in compact form, as in ../burgers/opInf.jl)

### Part A: polarized, exact (`gowdy_polarized.jl`)
B&M eq. 3.18 gives P = Σ Z₀(n e^{−τ})(a_n cos nθ + b_n sin nθ). Here Z₀ is any order-0 Bessel solution, and
for n = 0 it reduces to α + β ln t. I use t ∈ [1, 16] (the expanding direction, where the solution
oscillates), modes n ≤ 10 with random coefficients ∝ 0.75ⁿ, and both J₀ and Y₀. The state is x = [P; Π], Π = P_t:

    dP/dt = Π,   dΠ/dt = P_θθ − (1/t) Π      →  A x + (1/t) N x,  exactly.

Training uses t ≤ 11 and prediction uses 11 < t ≤ 16. The grid has N = 128 points, and the time derivatives are analytic.

### Part B: unpolarized, own solver (`gowdy_unpolarized.jl`)
The solver uses state [P; P_τ; Q; Q_τ], 4th-order periodic finite differences, and RK4 in τ toward the singularity.
The initial data are the Berger–Moncrief test family P = 0, P_τ = v₀ cos θ, Q = cos θ, Q_τ = 0.

**Lifting.** I derived this myself and checked it numerically below. The variables are

    G = e^{−τ} P_θ,   R = e^{P} Q_τ,   S = e^{P−τ} Q_θ,   with Π = P_τ, v(τ) = e^{−τ}.

With them the equations close as an exactly quadratic system with a bilinear input:

    Π_τ = v G_θ + R² − S²
    G_τ = −G + v Π_θ
    R_τ = −Π R + v S_θ + G S
    S_τ = −S + Π S + v R_θ − G R

In this system P appears only through Π, so P is recovered as P(0) + ∫Π dτ. Q is not needed.
The derivation substitutes e^{P}Q_θθ = S_θ e^{τ} − P_θ S e^{τ} and uses the Q equation; the
e^{2P} factor is absorbed into R², S².

The lifted variables are the orthonormal-frame components of the Hamiltonian density in B&M eq. 3.8:
H-density = Π² + G² + R² + S². So they stay O(1), while Q and Q_τ grow to O(10–40).
The momentum constraint becomes ∮(GΠ + SR) dθ = 0.

I compare two ROMs, both with the model form above:
* **lifted** OpInf on [Π, G, R, S]. Its structure matches the lifted equations exactly.
* **naive** OpInf on [P, Π, Q, Q_τ]. A quadratic model cannot represent e^{2P}, so this form is wrong.

The intrusive baseline is POD-Galerkin projection of the lifted FD system, including the exact
quadratic tensor.

**ROM setup (both parts).** I use separate POD bases with k modes for each variable (block POD), so
r = (#variables)·k. In Part B each variable is scaled by its max |value| on the training window.
The least-squares problem is solved as the augmented problem [D; √λ I] via QR rather than the
normal equations, and the ROMs are integrated with RK4 (`rk4_step!` from `../burgers/burgers_fd.jl`).
λ is chosen per k by **training** error only.

## How to run

```bash
julia einstein/gowdy_polarized.jl     # ~1.5 min: checks, sweep, operator-recovery table
julia einstein/gowdy_unpolarized.jl   # ~15 min: validation, both v0 cases
```

Figures and data go to `output/`. The `.jld2` data files (≈ 26 MB) are not tracked in git; rerun
the scripts to regenerate them.

Both scripts are `include`-able. Parameters are in the `pol_params` / `unp_params` /
`unp_hard_params` blocks at the top. The packages used are LinearAlgebra, SparseArrays, Printf, Random
(stdlib) plus JLD2, Plots and **SpecialFunctions** (already installed; needed for J₀, Y₀, J₁, Y₁).

## Results

Full tables are in `output/polarized_log.txt` and `output/unpolarized_log.txt`. Errors are
relative Frobenius errors. "POD floor" is the best any r-dimensional model in this basis can do.

### Part A: polarized (linear, exact data)

| k per variable | POD floor (train / predict) | Galerkin | OpInf, constant-coefficient | OpInf with 1/t input (λ = 0) |
|---|---|---|---|---|
| 10 | 8.5e-2 / 5.4e-2 | 1.4e-1 / 1.5e-1 | 9.3e-2 / 9.4e-2 | unstable |
| 16 | 1.5e-2 / 1.2e-2 | 3.7e-2 / 3.4e-2 | unstable | 1.5e-2 / 1.7e-2 |
| 21 | 1e-15 / 9e-16 | 6.8e-4 / 1.0e-3 | 7.8e-7 / 3.1e-2 | 1.1e-7 / 3.2e-3 |

- The data checks pass: the PDE residual converges at 2nd order in h, the closed-form Π matches
  finite differences to 3e-8, and the FD Laplacian is accurate to 1.6e-4.
- The model needs the time-dependent term. The constant-coefficient model fits the training window
  but predicts 10× worse, or blows up.
- k = 21 spans the data exactly: there are 21 Fourier modes for n ≤ 10. OpInf then beats Galerkin,
  because Galerkin uses the FD operators while the data are exact.
- **Operator recovery at k = 21:**
  - 1 trajectory: cond(D) = 3e17 and the operators are wrong by O(1), even though the fit is perfect.
  - 2 trajectories: cond(D) = 2e8. N̂ is recovered to 4e-9, Â matches the spectral Laplacian to
    1e-11, and the prediction error is 1.7e-7.
- Regularization hurts here: λ = 1e-6 raises the prediction error at k = 21 from 3e-3 to 2e-2.

### Part B: unpolarized (nonlinear, own solver)

- **Solver:** against the exact pseudo-unpolarized solution it shows clean 4th-order convergence
  (2.6e-9 at N = 512). The momentum constraint holds to 1e-14 (v₀ = 2) and 4e-13 (v₀ = 3).
  Self-convergence for v₀ = 3 is only about 3rd order at the resolutions used, because the solution
  develops steep features as τ grows.
- **Lifting:** the lifted quadratic system reproduces the solver's derivatives to 5e-6 (v₀ = 2) and
  7e-5 (v₀ = 3). What remains is the discrete product-rule error, not a modelling error. (The four
  lifted equations were also checked by hand.)

Error in P at k = 10 per variable (train / predict):

| Case | Galerkin (lifted) | OpInf lifted | OpInf naive |
|---|---|---|---|
| v₀ = 2, train τ ≤ 2, predict τ ≤ 4 | 1.6e-7 / 1.8e-3 | 8.0e-8 / 3.5e-3 | 3.0e-8 / 6.7e-3 |
| v₀ = 3, train τ ≤ 2.5, predict τ ≤ 5 | 9.5e-6 / 2.7e-2 | 1.3e-6 / 2.5e-2 | 1.2e-6 / 4.3e-2 |

- **Training:** both OpInf models reach the POD floor.
- **Prediction:** the lifted model is about 2× better than the naive one, and on par with intrusive
  Galerkin. The naive form can't represent e^{2P}, and that shows up once the model leaves the
  training window.
- **v₀ = 3:** prediction is limited by the basis, not the regression. The POD floor itself is
  0.13–0.4 in the prediction window, because the solution moves into states the training data
  never saw.
- **Operators:** cond(D) is 1e18–1e26, and the learned operators differ from Galerkin by about
  100% even when the trajectories agree. This is the same single-trajectory non-uniqueness seen in
  Part A and in `../analysis` (Q1, Q6).

### Takeaways

1. OpInf works on exact solutions of Einstein's equations if the model form is right. That means
   time-dependent coefficients as inputs and, in the nonlinear case, a lifting to quadratic form.
2. Lifting pays off in prediction, not in training. Both forms fit the training data.
3. With one trajectory the operators are not identifiable. Two trajectories are enough in the
   linear case.

### Caveats

- The Gowdy equations are taken from Berger & Moncrief (1993) as transcribed by the agent that wrote
  this study. They were checked against the exact solutions numerically, not re-checked line by line
  against the paper.
- λ is picked per k by training error on a coarse grid.
- One run per case; no seeds or parameter sweeps beyond v₀ ∈ {2, 3}.

