# OpInfSurvey

A survey of **Operator Inference (OpInf)** reduced-order models on PDEs of increasing difficulty,
done as a numerical linear algebra course project at BYU.

Every problem uses the same pipeline: get snapshots of the full solution (from our own solver, or
published data for Einstein's equations), compute a POD basis with the SVD, learn the reduced operators by regularized least squares, then integrate
the reduced model and compare. The focus is the linear algebra: how the least-squares problem
behind OpInf is conditioned, solved and regularized, and when it recovers the operators that
intrusive projection would give.

| Problem | Status | Model form | Folder |
|---|---|---|---|
| Burgers' equation | Main study; analyses of all five project questions done | Quadratic | `burgers/`, `analysis/` |
| Einstein's field equations (binary black hole, published SXS data) | Exploratory study done | Linear / quadratic | `einstein/` |
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
accuracy levels off. Training on four initial conditions lowers $\kappa(D)$ by $10^{3}$ – $10^{11}$ and
the prediction error at $r = 15$ from $4\times10^{-2}$ to $10^{-5}$.

### Einstein's field equations: a binary black hole merger (published data)

The vacuum Einstein equations $G_{\mu\nu} = 0$ for two black holes that spiral in and merge. We
don't solve them: the solution is **SXS:BBH:0305** from the SXS catalog
([doi:10.26138/SXS:BBH:0305](https://doi.org/10.26138/SXS:BBH:0305)), a published simulation of the
GW150914-like system. The published solution is the gravitational-wave strain on the sphere,

$$
h(t, \theta, \phi) = \sum_{\ell \ge 2} \sum_{|m| \le \ell} h_{\ell m}(t)\, {}_{-2}Y_{\ell m}(\theta, \phi),
$$

with 77 modes ($\ell = 2, \dots, 8$). The OpInf state is $x = [\operatorname{Re} h_{\ell m}; \operatorname{Im} h_{\ell m}]$
for the 70 modes with $m \ne 0$ ($x \in \mathbb{R}^{140}$). We train on one time window, predict
forward, and compare with the published solution.

**Main findings** (`einstein/README.md`):
- **Ringdown** (after merger) is linear dynamics, and a linear OpInf model fits it. The learned
  $\hat A$ contains the remnant black hole's quasinormal-mode frequency, matching the Kerr value to
  0.8–2 %, and the match improves as training starts later after merger.
- **Inspiral** (before merger) chirps: frequency and amplitude grow together. A linear model can't
  represent this. A quadratic model at $r = 6$ fits training to 0.8 % but predicts only ~40 M ahead
  within 5 %.
- As with Burgers', the quadratic data matrix with one trajectory is badly conditioned:
  $\kappa(D) \ge 4\times10^{10}$ at $r = 8$.

`einstein/README.md` starts with a crash course on modeling Einstein's equations and on what the
catalog publishes, for readers who know numerical PDEs but not general relativity.

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
└── einstein/                    OpInf on a published binary-black-hole solution (SXS:BBH:0305)
    ├── fetch_sxs.py             downloads the SXS data and exports plain HDF5 (needs `pip install sxs h5py`)
    ├── opinf_sxs.jl             OpInf: ringdown and inspiral experiments, QNM check
    ├── README.md                crash course, data, method, results
    ├── data/                    downloaded data (gitignored; recreate with fetch_sxs.py)
    └── output/                  figures and sxs_log.txt
```

## Requirements

Julia ≥ 1.10 with:

```julia
using Pkg; Pkg.add(["JLD2", "Plots"])
```

`einstein/fetch_sxs.py` needs Python with `sxs` and `h5py` (download step only). `burgers/burgers_mol.jl` needs `MethodOfLines`,
`ModelingToolkit`, `OrdinaryDiffEq` and `OrdinaryDiffEqBDF`.

## Usage

```bash
julia burgers/opInf.jl                   # Burgers' reduced model; prints its error, writes burgers_rom.gif
julia burgers/burgers_fd.jl              # regenerate the Burgers' snapshot data (a few seconds)
julia analysis/run_all.jl                # all Burgers' analyses, ≈ 2 min
python einstein/fetch_sxs.py             # Einstein: download the SXS solution (once)
julia einstein/opinf_sxs.jl              # Einstein: OpInf on it, ≈ 1 min
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

The `name` prefix keeps outputs from different runs separate.

## Roadmap

The full plan is in [`docs/project_plan.html`](docs/project_plan.html). The equations are ordered
by how much they change the linear algebra:

| Equation | Status | Model form | Main new challenge |
|---|---|---|---|
| Burgers' | Main study; analyses in `analysis/` | Quadratic | Baseline |
| Einstein (binary black hole, SXS data) | Exploratory study in `einstein/` | Linear (ringdown), quadratic (inspiral) | Data-driven only; chirp isn't polynomial in h |
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
