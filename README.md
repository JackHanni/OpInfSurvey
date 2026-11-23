# OpInfBurger

Operator Inference (OpInf) reduced-order models for the 1D viscous Burgers' equation

$$
u_t + u\,u_x = \nu\,u_{xx}, \qquad x \in [0, 1],\; t \in [0, T],
$$

with $u(x, 0) = \sin(\pi x)$ and $u(0, t) = u(1, t) = 0$ by default.

Burgers' nonlinearity is exactly quadratic, so the OpInf model form
$\dot{\hat q} = \hat A \hat q + \hat H (\hat q \otimes \hat q)$ matches the structure of the
true POD-Galerkin reduced model.

## Layout

Each folder keeps its code at the top level and everything it generates in `output/`.

```
burgerProj/
├── README.md
├── docs/
│   └── project_plan.html        project plan: questions, timeline, figures, risks
├── burgers/                     the main study
│   ├── burgers_fd.jl            full-order solver (finite differences + RK4)
│   ├── opInf.jl                 OpInf reduced model for one choice of r and λ
│   ├── burgers_mol.jl           same PDE via MethodOfLines.jl (slow; kept for comparison)
│   └── output/                  burgers_data.jld2, burgers.gif, burgers_snapshots.png,
│                                burgers_rom.gif, burgers_mol_snapshots.png
├── analysis/                    conditioning, solver, regularization and robustness studies
│   ├── common.jl                shared helpers (data matrix, Galerkin operators, solvers, ROM)
│   ├── run_all.jl               runs plan questions Q1–Q5 plus Q6 (more trajectories)
│   ├── README.md                results table
│   └── output/                  figures (q1_… to q6_…) and results.txt
└── einstein/                    side study: OpInf on Gowdy T³ solutions of Einstein's equations
    ├── gowdy_common.jl
    ├── gowdy_polarized.jl       linear case, exact Bessel-function solutions
    ├── gowdy_unpolarized.jl     nonlinear case, own solver + lifting to quadratic form
    ├── README.md                sources, equations, results
    └── output/                  figures (data files are regenerated, not tracked)
```

## Requirements

Julia ≥ 1.10 with:

```julia
using Pkg; Pkg.add(["JLD2", "Plots"])
```

`einstein/` also needs `SpecialFunctions`. `burgers/burgers_mol.jl` needs `MethodOfLines`,
`ModelingToolkit`, `OrdinaryDiffEq` and `OrdinaryDiffEqBDF`.

## Usage

The snapshot data and figures are in the repo, so the reduced model runs straight after cloning:

```bash
julia burgers/opInf.jl          # prints the ROM's relative error, writes burgers/output/burgers_rom.gif
```

To regenerate the snapshot data (a few seconds, mostly plotting):

```bash
julia burgers/burgers_fd.jl
```

This writes to `burgers/output/`:

- `burgers_data.jld2`: `xs`, `ts`, `U` (N × n_snapshots), `dUdt` (exact semi-discrete time derivatives), `ν`, `N`, `dt`, `save_dt`, `uL`, `uR`
- `burgers_snapshots.png`: the solution at 11 times
- `burgers.gif`: animation

The analyses and the Einstein side study each have their own README:

```bash
julia analysis/run_all.jl                # ≈ 2 min
julia einstein/gowdy_polarized.jl        # ≈ 1.5 min
julia einstein/gowdy_unpolarized.jl      # ≈ 15 min
```

### Changing the problem

All settings live in `default_params` at the top of `burgers/burgers_fd.jl`: ν, domain, grid size `N`,
final time, snapshot spacing, initial condition, boundary values, output name, plot/GIF switches.
The time step is chosen automatically from the diffusion and advection stability limits, and the
solver warns if the grid is too coarse for ν (cell Péclet number > 2).

To run a variant without editing the file:

```julia
include("burgers/burgers_fd.jl")
out = run_burgers(merge(default_params, (ν = 0.01, N = 401, name = "nu0p01")))
```

The `name` prefix keeps outputs from different runs separate.

## Roadmap

The full plan is in [`docs/project_plan.html`](docs/project_plan.html). After the Burgers' study,
the same pipeline (snapshots → SVD → least squares → ROM) will be applied to harder equations,
easiest first:

| Equation | Status | Model form | Main new challenge |
|---|---|---|---|
| Burgers' | In progress; analyses in `analysis/` | Quadratic | Baseline |
| Korteweg–de Vries | Planned | Quadratic | Stiff $u_{xxx}$ term; slowly decaying singular values (solitons) |
| Navier–Stokes (2D, vorticity form) | Planned | Quadratic | Large state; cheap SVD of a tall snapshot matrix |
| Nonlinear Schrödinger | If time permits | Cubic | $D$ grows like $r^3$; conserved quantities |
| Yang–Mills (classical SU(2)) | If time permits | Quadratic + cubic | Gauge freedom |
| Einstein (Gowdy T³ cosmologies) | Exploratory study in `einstein/` | Quadratic after lifting | Non-polynomial terms; time-dependent coefficients |

## Notes

- Keep `N ≳ 1/ν`; smaller ν gives steeper fronts and needs a finer grid.
- Use the saved `dUdt` rather than finite differences of `U` when fitting OpInf operators. Exact
  derivatives make the fit more accurate; they don't change the conditioning of the data matrix,
  which depends only on the snapshots (see `analysis/README.md`).
- `burgers/output/` is tracked in git. If you rerun `burgers_fd.jl` with new parameters, commit
  the updated data and figures so they stay in sync with the code.
