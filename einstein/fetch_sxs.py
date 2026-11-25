"""Download a binary-black-hole solution of the Einstein equations from the SXS catalog
and export it as plain arrays for the Julia OpInf script.

Source: SXS Collaboration, simulation SXS:BBH:0305 (GW150914-like), highest resolution Lev6.
Record: https://doi.org/10.26138/SXS:BBH:0305  (hosted on CaltechDATA)

The published file uses SXS's compressed format, so the `sxs` package is needed to decode it:
    pip install sxs h5py
Run:
    python einstein/fetch_sxs.py
Writes einstein/data/sxs_bbh_0305.h5 with
    t      (n_t,)          uniform times, in units of the total mass M
    h_re, h_im   (n_t, n_modes)   strain modes r h_lm / M
    hdot_re, hdot_im             their exact time derivatives (spline derivative of the data)
    ell, m (n_modes,)      mode labels
and einstein/data/sxs_bbh_0305_horizons.h5 with the apparent horizons (for animate_merger_3d.jl)
    {A,B,C}_t, {A,B,C}_center, {A,B,C}_areal_mass   time, coordinate centre (x, y, z), areal mass
    of the two initial black holes (A, B) and of the common horizon (C), which appears at merger.
Only the horizon export is needed for the animation:  python einstein/fetch_sxs.py --horizons-only
"""
import json
import sys
import urllib.request
from pathlib import Path

import h5py
import numpy as np

RECORD = "zggft-s0306"  # CaltechDATA record behind doi:10.26138/SXS:BBH:0305
RESOLUTION = "Lev6"
BASE = f"https://data.caltech.edu/api/records/{RECORD}/files"
HERE = Path(__file__).resolve().parent
RAW_DIR = HERE / "data" / "raw"
STRAIN_OUT = HERE / "data" / "sxs_bbh_0305.h5"
HORIZONS_OUT = HERE / "data" / "sxs_bbh_0305_horizons.h5"

T_START = 300.0  # drop the initial "junk radiation" (non-physical transient from the initial data)
DT = 0.5         # uniform resampling step, in M

def download(name):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_DIR / name
    if not dest.exists():
        url = f"{BASE}/{RESOLUTION}:{name}/content"
        print("downloading", url)
        urllib.request.urlretrieve(url, dest)
    return dest

def export_horizons():
    """Copy the horizon trajectories into an uncompressed file that JLD2 can read."""
    horizons_file = download("Horizons.h5")
    with h5py.File(horizons_file, "r") as horizons, h5py.File(HORIZONS_OUT, "w") as out:
        for name in ("A", "B", "C"):
            center = horizons[f"Ah{name}.dir/CoordCenterInertial.dat"][:]   # columns: t, x, y, z
            mass = horizons[f"Ah{name}.dir/ArealMass.dat"][:]               # columns: t, mass
            out[f"{name}_t"] = center[:, 0]
            out[f"{name}_center"] = center[:, 1:4]
            out[f"{name}_areal_mass"] = np.interp(center[:, 0], mass[:, 0], mass[:, 1])
    print(f"wrote {HORIZONS_OUT}")

def export_strain():
    import sxs  # only needed to decode the strain file

    strain_file = download("Strain_N2.h5")
    download("Strain_N2.json")
    metadata = json.load(open(download("metadata.json")))

    strain = sxs.load(str(strain_file))  # extrapolated (N=2) strain modes, inertial frame
    t = np.arange(T_START, strain.t[-1], DT)
    strain_uniform = strain.interpolate(t)
    strain_dot = strain.dot.interpolate(t)  # derivative of the spline through the published data

    STRAIN_OUT.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(STRAIN_OUT, "w") as out:
        out["t"] = t
        out["h_re"], out["h_im"] = np.real(strain_uniform.data), np.imag(strain_uniform.data)
        out["hdot_re"], out["hdot_im"] = np.real(strain_dot.data), np.imag(strain_dot.data)
        out["ell"], out["m"] = strain_uniform.LM[:, 0], strain_uniform.LM[:, 1]
        out.attrs["source"] = "SXS:BBH:0305 " + RESOLUTION + ", doi:10.26138/SXS:BBH:0305"
        out.attrs["common_horizon_time"] = metadata["common_horizon_time"]
        out.attrs["remnant_mass"] = metadata["remnant_mass"]
        out.attrs["remnant_spin"] = metadata["remnant_dimensionless_spin"][2]
    print(f"wrote {STRAIN_OUT}: {len(t)} times, {strain_uniform.shape[1]} modes, t in [{t[0]}, {t[-1]}]")

if __name__ == "__main__":
    if "--horizons-only" not in sys.argv:
        export_strain()
    export_horizons()
