# OpInf on the Einstein field equations: Gowdy T³ cosmologies

This is an exploratory side study for the Burgers' OpInf project. I tried Operator Inference on
solutions of the vacuum Einstein equations G_μν = 0 in the
**T³ Gowdy** symmetry class. Part A uses exact solutions. Part B uses data from my own solver,
checked against an exact nonlinear solution.

## Crash course: modeling Einstein's equations

This section is for readers who know numerical PDEs and linear algebra but not general relativity.
It covers just enough to follow the rest of this README. Units are geometric (G = c = 1).

### 1. What the equations say

The unknown is the **metric** g_μν(x): a symmetric 4×4 matrix field (10 functions of space and time)
that defines distances and times, ds² = g_μν dx^μ dx^ν. Einstein's equations are

    G_μν = 8π T_μν,        G_μν = R_μν − ½ R g_μν

- T_μν is the matter content (energy density, momentum, stress). In **vacuum** T_μν = 0, and
  spacetime can still be curved and dynamic: gravitational waves are vacuum solutions.
- G_μν (the Einstein tensor) is built from first and second derivatives of g_μν. The equations are
  a **nonlinear second-order PDE system** for g. They are linear in the second derivatives
  (quasilinear), but the coefficients depend on g and its inverse.
- Analogy: in Burgers', u_t + u u_x = ν u_xx, the unknown field also sets the coefficients of its
  own equation. Einstein's equations do the same, with the inverse metric g^μν playing the role
  of u in u·u_x. The inverse metric is where the non-polynomial terms come from.

### 2. Why it is hard numerically

1. **Coordinate (gauge) freedom.** Any change of coordinates gives another valid g_μν for the same
   physical spacetime. Four of the ten functions are pure coordinate choices, so the equations alone
   don't determine a unique solution. You must pick a gauge, and a bad one can make a smooth
   physical solution look singular on the grid.
2. **Constraints and evolution.** Split spacetime into slices of constant time (the **3+1** or ADM
   split):

       ds² = −α² dt² + γ_ij (dx^i + β^i dt)(dx^j + β^j dt)

   Here γ_ij is the metric within a slice, α (lapse) sets how far apart in proper time the slices
   are, and β^i (shift) sets how spatial coordinates move between slices. Einstein's equations then
   divide into:
   - **evolution equations** (hyperbolic-like) for γ_ij and its time derivative, the extrinsic
     curvature K_ij; and
   - **constraint equations** (elliptic) that every slice must satisfy, for example the Hamiltonian
     constraint R + K² − K_ij K^ij = 16πρ.

   The exact evolution preserves the constraints, but discretization error does not. Constraint
   violation can grow and wreck a simulation, so monitoring it is a standard accuracy check.
3. **Well-posedness.** The original ADM evolution equations are only weakly hyperbolic and tend to
   blow up numerically. Production codes use reformulations (BSSN, generalized harmonic) that are
   strongly hyperbolic. This is part of why full 3D numerical relativity took decades.
4. **Singularities.** Black holes and big-bang/big-crunch singularities are places where curvature
   blows up. Codes must avoid them (excision, special gauges) or approach them carefully.

### 3. Symmetry reductions: how a course project can touch this

Imposing symmetries collapses the 3+1 problem to something solvable on a laptop:

| Symmetry | Resulting problem | Example |
|---|---|---|
| Homogeneous and isotropic | ODEs in time | FLRW cosmology (Friedmann equations) |
| Homogeneous, anisotropic | ODEs in time | Bianchi cosmologies (the spatially homogeneous Yang–Mills reduction is similar) |
| Spherical | 1+1 PDEs in (t, r) | Scalar-field collapse (Choptuik 1993) |
| Two commuting spatial symmetries on T³ | 1+1 PDEs in (t, θ), periodic | **Gowdy T³**, used here |

1+1 PDEs are the sweet spot for OpInf: a genuine field on a grid, but small enough for dense SVDs.

### 4. Gowdy T³ in particular

Gowdy spacetimes are closed universes, spatially a 3-torus T³, that are unchanged when you move
along two of the three torus directions (σ, δ). Everything depends only on time τ and the third
angle θ. The metric (written out in "Equations" below) contains three functions:

- **P(τ, θ) and Q(τ, θ):** the two polarizations of gravitational waves travelling around the θ
  circle. These are the dynamical fields.
- **λ(τ, θ):** a background factor that is *not* evolved independently. It follows from P and Q by
  integrating the constraint equations. This is the 3+1 constraint/evolution split in miniature.

Key features, and how they show up in the code:

- **Time runs toward a singularity.** The areal time is t = e^{−τ}, and the universe has a big-bang
  singularity at t → 0, i.e. τ → ∞. The equations contain factors e^{−2τ} on every spatial-
  derivative term. As τ grows they switch off, and each θ evolves almost on its own. This is called
  **AVTD** (asymptotically velocity term dominated) behaviour. In the OpInf model, that e^{−2τ} (or
  1/t) factor is exactly the time-dependent input u(t).
- **Polarized vs unpolarized.** With Q ≡ 0 the P equation is *linear* and has exact solutions in
  Bessel functions (Part A). With Q ≠ 0 the equations couple through e^{2P}, which is non-polynomial
  (Part B). Near the singularity, Berger & Moncrief reported narrow "spikes" forming in the
  unpolarized case; such steep features are a known challenge for both solvers and POD.
- **Constraints as checks.** On T³ the momentum constraint requires ∮(P_θ P_τ + e^{2P} Q_θ Q_τ) dθ = 0.
  The solver monitors it; it holds to about 1e-14.
- **Gauge is already fixed.** The Gowdy form of the metric fixes the coordinates (areal time), so
  none of the gauge problems from Section 2 arise. This is a big part of why the model fits in a
  few hundred lines of Julia.

### 5. How a numerical model is built and trusted here

1. **Discretize:** periodic θ grid, 4th-order centered finite differences in θ, RK4 in τ. This is
   the same method-of-lines approach as `../burgers/burgers_fd.jl`.
2. **Verify against an exact solution:** the error should fall at the scheme's order (4th) as the
   grid is refined. The solver shows 4.00 at N = 512.
3. **Self-convergence** when no exact solution exists: compare runs at N and 2N.
4. **Monitor the constraints:** they must stay at rounding level or converge to zero with
   resolution.
5. **Only then** use the snapshots as OpInf training data.

### 6. Further reading

- S. Carroll, *Spacetime and Geometry* (Addison-Wesley, 2004): general relativity from scratch.
- M. Alcubierre, *Introduction to 3+1 Numerical Relativity* (Oxford University Press, 2008).
- T. Baumgarte & S. Shapiro, *Numerical Relativity: Solving Einstein's Equations on the Computer*
  (Cambridge University Press, 2010).
- E. Gourgoulhon, *3+1 Formalism in General Relativity* (Springer Lecture Notes in Physics 846,
  2012); free version at arXiv:gr-qc/0703035.
- B. Berger, *Numerical approaches to spacetime singularities*, Living Rev. Relativ. 5, 1 (2002),
  which covers Gowdy models specifically (see Sources below).

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

