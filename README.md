# OpInfSurvey

A survey of **Operator Inference (OpInf)** reduced-order models on PDEs of increasing difficulty,
done as a numerical linear algebra course project at BYU.

Every problem uses the same pipeline: simulate the full-order model, collect snapshots, compute a
POD basis with the SVD, learn the reduced operators by regularized least squares, then integrate
the reduced model and compare. The focus is the linear algebra: how the least-squares problem
behind OpInf is conditioned, solved and regularized, and when it recovers the operators that
intrusive projection would give.

| Problem | Status | Model form | Folder |
|---|---|---|---|
| Burgers' equation | Main study; analyses of all five project questions done | Quadratic | `burgers/`, `analysis/` |
| Einstein's field equations (Gowdy T³ cosmologies) | Exploratory study done | Quadratic after lifting, plus a time-dependent input | `einstein/` |
| KdV, Navier–Stokes, NLS, Yang–Mills | Planned (see Roadmap) | | |

## The problems

### Burgers' equation

The 1D viscous Burgers' equation

$$
u_t + u\,u_x = \nu\,u_{xx}, \qquad x \in [0, 1],\; t \in [0, T],
$$

with $u(x, 0) = \sin(\pi x)$, $u(0, t) = u(1, t) = 0$ and $\nu = 0.02$ by default. It is discretized
with centered finite differences on 201 points and RK4 in time.

The nonlinearity is exactly quadratic, so the OpInf model form
$\dot{\hat q} = \hat A \hat q + \hat H (\hat q \otimes \hat q)$ matches the structure of the true
POD-Galerkin reduced model.

**Main finding** (`analysis/README.md`): the regression is limited by the training data, not the
solver. With one trajectory, the data matrix reaches $\kappa(D) \approx 10^{13}$ at $r = 6$, and OpInf's
accuracy levels off. Training on four initial conditions lowers $\kappa(D)$ by $10^{3}$–$10^{11}$ and
the prediction error at $r = 15$ from $4\times10^{-2}$ to $10^{-5}$.

### Einstein's field equations: Gowdy T³ cosmologies

The vacuum Einstein equations $G_{\mu\nu} = 0$, restricted to **Gowdy T³** spacetimes. These are closed
universes, spatially a 3-torus, that are symmetric along two of the torus directions. With that
symmetry the equations reduce *exactly* to two coupled wave equations for the gravitational-wave
amplitudes $P(\tau, \theta)$ and $Q(\tau, \theta)$ on a periodic circle $\theta \in [0, 2\pi)$
(Berger & Moncrief 1993):

$$
\begin{aligned}
P_{\tau\tau} - e^{-2\tau} P_{\theta\theta} - e^{2P}\left(Q_\tau^2 - e^{-2\tau} Q_\theta^2\right) &= 0, \\
Q_{\tau\tau} - e^{-2\tau} Q_{\theta\theta} + 2\left(P_\tau Q_\tau - e^{-2\tau} P_\theta Q_\theta\right) &= 0 .
\end{aligned}
$$

Time runs toward a big-bang singularity at $\tau \to \infty$. Two features make this a harder test
for OpInf than Burgers':

- **Time-dependent coefficients.** The equations depend explicitly on time through $e^{-2\tau}$.
  The time-dependent factor ($1/t$ with $t = e^{-\tau}$ in the polarized case, $e^{-\tau}$ after lifting)
  is treated as a known input $u$, giving the model
  $\dot{\hat x} = \hat A \hat x + u\,\hat N \hat x + \hat H (\hat x \otimes \hat x)$.
- **A non-polynomial nonlinearity.** $e^{2P}$ can't be written as a quadratic. A change of variables
  (lifting) to $G = e^{-\tau}P_\theta$, $R = e^{P}Q_\tau$, $S = e^{P-\tau}Q_\theta$ makes the system
  exactly quadratic.

Two cases are studied:
- **Polarized** ($Q \equiv 0$): linear, with exact Bessel-function solutions.
- **Unpolarized:** nonlinear, solved with our own 4th-order solver and validated against an exact
  solution.

**Main findings** (`einstein/README.md`):
- The polarized model needs the time-dependent input term.
- The lifted model predicts about 2× better than OpInf on the original variables, and matches
  intrusive Galerkin.
- As with Burgers', one trajectory is not enough to identify the true operators; two are.

`einstein/README.md` starts with a crash course on modeling Einstein's equations, for readers who
know numerical PDEs but not general relativity.

## Layout

Each folder keeps its code at the top level and everything it generates in `output/`.

```
OpInfSurvey/
├── README.md
├── docs/
│   └── project_plan.html        project plan: questions, timeline, figures, risks, results so far
├── burgers/                     Burgers' full-order model and reduced model
│   ├── burgers_fd.jl            full-order solver (finite differences + RK4)
│   ├── opInf.jl                 OpInf reduced model for one choice of r and λ
│   ├── burgers_mol.jl           same PDE via MethodOfLines.jl (slow; kept for comparison)
│   └── output/                  burgers_data.jld2, burgers.gif, burgers_snapshots.png,
│                                burgers_rom.gif, burgers_mol_snapshots.png
├── analysis/                    conditioning, solver, regularization and robustness studies (Burgers')
│   ├── common.jl                shared helpers (data matrix, Galerkin operators, solvers, ROM)
│   ├── run_all.jl               runs plan questions Q1–Q5 plus Q6 (more trajectories)
│   ├── README.md                results table and a proof skeleton for the accuracy theory
│   └── output/                  figures (q1_… to q6_…) and results.txt
└── einstein/                    OpInf on Gowdy T³ solutions of Einstein's equations
    ├── gowdy_common.jl
    ├── gowdy_polarized.jl       linear case, exact Bessel-function solutions
    ├── gowdy_unpolarized.jl     nonlinear case, own solver + lifting to quadratic form
    ├── README.md                crash course, sources, equations, results
    └── output/                  figures and run logs (data files are regenerated, not tracked)
```

## Requirements

Julia ≥ 1.10 with:

```julia
using Pkg; Pkg.add(["JLD2", "Plots"])
```

`einstein/` also needs `SpecialFunctions`. `burgers/burgers_mol.jl` needs `MethodOfLines`,
`ModelingToolkit`, `OrdinaryDiffEq` and `OrdinaryDiffEqBDF`.

## Usage

```bash
julia burgers/opInf.jl                   # Burgers' reduced model; prints its error, writes burgers_rom.gif
julia burgers/burgers_fd.jl              # regenerate the Burgers' snapshot data (a few seconds)
julia analysis/run_all.jl                # all Burgers' analyses, ≈ 2 min
julia einstein/gowdy_polarized.jl        # Einstein, polarized case, ≈ 1.5 min
julia einstein/gowdy_unpolarized.jl      # Einstein, unpolarized case, ≈ 15 min
```

The Burgers' snapshot data and figures are in the repo, so `opInf.jl` runs straight after cloning.
`burgers_fd.jl` writes to `burgers/output/`:

- `burgers_data.jld2`: `xs`, `ts`, `U` (N × n_snapshots), `dUdt` (exact semi-discrete time derivatives), `ν`, `N`, `dt`, `save_dt`, `uL`, `uR`
- `burgers_snapshots.png`: the solution at 11 times
- `burgers.gif`: animation

### Changing the Burgers' problem

All settings live in `default_params` at the top of `burgers/burgers_fd.jl`: ν, domain, grid size `N`,
final time, snapshot spacing, initial condition, boundary values, output name, plot/GIF switches.
The time step is chosen automatically from the diffusion and advection stability limits, and the
solver warns if the grid is too coarse for ν (cell Péclet number > 2).

To run a variant without editing the file:

```julia
include("burgers/burgers_fd.jl")
out = run_burgers(merge(default_params, (ν = 0.01, N = 401, name = "nu0p01")))
```

The `name` prefix keeps outputs from different runs separate. The Einstein scripts work the same
way, with `pol_params`, `unp_params` and `unp_hard_params` at the top of each file.

## Roadmap

The full plan is in [`docs/project_plan.html`](docs/project_plan.html). The equations are ordered
by how much they change the linear algebra:

| Equation | Status | Model form | Main new challenge |
|---|---|---|---|
| Burgers' | Main study; analyses in `analysis/` | Quadratic | Baseline |
| Einstein (Gowdy T³ cosmologies) | Exploratory study in `einstein/` | Quadratic after lifting, plus input | Non-polynomial terms; time-dependent coefficients |
| Korteweg–de Vries | Planned | Quadratic | Stiff $u_{xxx}$ term; slowly decaying singular values (solitons) |
| Navier–Stokes (2D, vorticity form) | Planned | Quadratic | Large state; cheap SVD of a tall snapshot matrix |
| Nonlinear Schrödinger | If time permits | Cubic | $D$ grows like $r^3$; conserved quantities |
| Yang–Mills (classical SU(2)) | If time permits | Quadratic + cubic | Gauge freedom |

## Notes

- Keep `N ≳ 1/ν` for Burgers'; smaller ν gives steeper fronts and needs a finer grid.
- Use exact time derivatives (the saved `dUdt`) rather than finite differences of the snapshots when
  fitting OpInf operators. They make the fit more accurate; they don't change the conditioning of
  the data matrix, which depends only on the snapshots (see `analysis/README.md`).
- `burgers/output/` is tracked in git. If you rerun `burgers_fd.jl` with new parameters, commit the
  updated data and figures so they stay in sync with the code.
