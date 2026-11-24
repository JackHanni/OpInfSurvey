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
"""
import json
import urllib.request
from pathlib import Path

import h5py
import numpy as np
import sxs

RECORD = "zggft-s0306"  # CaltechDATA record behind doi:10.26138/SXS:BBH:0305
LEV = "Lev6"
BASE = f"https://data.caltech.edu/api/records/{RECORD}/files"
HERE = Path(__file__).resolve().parent
RAW = HERE / "data" / "raw"
OUT = HERE / "data" / "sxs_bbh_0305.h5"

T_START = 300.0  # drop the initial "junk radiation" (non-physical transient from the initial data)
DT = 0.5         # uniform resampling step, in M

def download(name):
    RAW.mkdir(parents=True, exist_ok=True)
    dest = RAW / name
    if not dest.exists():
        url = f"{BASE}/{LEV}:{name}/content"
        print("downloading", url)
        urllib.request.urlretrieve(url, dest)
    return dest

def main():
    h5 = download("Strain_N2.h5")
    download("Strain_N2.json")
    meta = json.load(open(download("metadata.json")))

    h = sxs.load(str(h5))  # extrapolated (N=2) strain modes, inertial frame
    t = np.arange(T_START, h.t[-1], DT)
    hu = h.interpolate(t)
    hdot = h.dot.interpolate(t)  # derivative of the spline through the published data

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(OUT, "w") as f:
        f["t"] = t
        f["h_re"], f["h_im"] = np.real(hu.data), np.imag(hu.data)
        f["hdot_re"], f["hdot_im"] = np.real(hdot.data), np.imag(hdot.data)
        f["ell"], f["m"] = hu.LM[:, 0], hu.LM[:, 1]
        f.attrs["source"] = "SXS:BBH:0305 " + LEV + ", doi:10.26138/SXS:BBH:0305"
        f.attrs["common_horizon_time"] = meta["common_horizon_time"]
        f.attrs["remnant_mass"] = meta["remnant_mass"]
        f.attrs["remnant_spin"] = meta["remnant_dimensionless_spin"][2]
    print(f"wrote {OUT}: {len(t)} times, {hu.shape[1]} modes, t in [{t[0]}, {t[-1]}]")

if __name__ == "__main__":
    main()
