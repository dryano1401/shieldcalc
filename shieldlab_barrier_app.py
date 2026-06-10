"""
ShieldLab Barrier Calculator
============================
Streamlit app for computing required barrier thickness from a target broad-beam
transmission using Archer-equation parameter fits (Archer et al. 1983 form used
in Oumano et al. 2025, JACMP 26:e70084):

    T(x) = [ (1 + beta/alpha) * exp(alpha * gamma * x) - beta/alpha ]^(-1/gamma)

Closed-form inverse (thickness for a required transmission T):

    x = 1/(alpha*gamma) * ln[ (T^(-gamma) + beta/alpha) / (1 + beta/alpha) ]

In the limit gamma -> 0 the model reduces to T = exp(-(alpha+beta)*x).

Built-in parameters come from the most recent ShieldLab GATE 10 simulation
sweeps (broad beam, 90-degree cone, G4_WATER phantom). New radionuclide /
barrier combinations can be added in the Parameter Library tab or imported
from CSV without touching the code.

Run with:
    streamlit run shieldlab_barrier_app.py
"""

import io
import math

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Built-in Archer parameters (alpha, beta in mm^-1; gamma dimensionless)
# Source: ShieldLab simulation sweeps (most recent fits)
# ----------------------------------------------------------------------------
BUILTIN_PARAMS = [
    # radionuclide, barrier,       alpha,    beta,       gamma,  R2,     maxdpct
    ("Zr-89", "Lead",        0.07212,  0.007212,   1.143,  1.0,    3.6),
    ("Zr-89", "NW Concrete", 0.01148, -0.005740,   5.000,  0.9999, 4.3),
    ("Zr-89", "LW Concrete", 0.007808, -0.0007808, 0.5000, 0.9999, 2.9),
    ("Zr-89", "Steel",       0.03791, -0.003791,   0.2750, 0.9999, 5.6),
    ("Cu-64", "Lead",        0.1367,  -0.01367,   30.00,   0.9999, 1.6),
    ("Cu-64", "NW Concrete", 0.01408, -0.007040,   2.429,  1.0,    4.6),
    ("Cu-64", "LW Concrete", 0.009422, -0.004711,  3.071,  0.9994, 4.7),
    ("Cu-64", "Steel",       0.04571, -0.02286,    3.714,  0.9999, 2.6),
    ("Cu-64", "Glass",       0.01464, -0.007322,   3.071,  0.9999, 4.5),
    ("Ga-68", "Lead",        0.1216,   0.01216,    0.0500, 0.9986, 6.9),
    ("Ga-68", "NW Concrete", 0.01379, -0.006897,   3.071,  0.9992, 4.7),
    ("Ga-68", "LW Concrete", 0.009405, -0.004703,  3.714,  0.9995, 4.3),
    ("Ga-68", "Steel",       0.04663, -0.02331,    3.071,  0.9994, 4.4),
    ("Ga-68", "Glass",       0.01521, -0.007603,   2.429,  0.9999, 4.4),
    ("I-124", "Lead",        0.05686,  0.02843,    1.143,  0.9996, 5.4),
    ("I-124", "NW Concrete", 0.01039, -0.005194,   5.000,  0.9974, 5.6),
    ("I-124", "LW Concrete", 0.007160, -0.0007160, 1.143,  0.9978, 3.8),
    ("I-124", "Steel",       0.03472, -0.003472,   0.500,  0.9962, 5.5),
    ("Rb-82", "Lead",        0.1289,   0.01289,    1.143,  1.0,    3.5),
    ("Rb-82", "NW Concrete", 0.01439, -0.007196,   1.786,  0.9996, 6.2),
    ("Rb-82", "LW Concrete", 0.009960, -0.004980,  1.786,  0.9993, 7.3),
    ("Rb-82", "Steel",       0.04857, -0.02428,    2.429,  1.0,    5.7),
    ("Rb-82", "Glass",       0.01506, -0.007530,   2.429,  0.9998, 4.8),
]

PARAM_COLUMNS = ["Radionuclide", "Barrier", "alpha_mm", "beta_mm", "gamma",
                 "R2", "MaxDeltaPct", "Source"]

GAMMA_EPS = 1e-6  # below this |gamma|, use the exponential limit


# ----------------------------------------------------------------------------
# Archer model
# ----------------------------------------------------------------------------
def archer_transmission(x_mm, alpha, beta, gamma):
    """Broad-beam transmission at thickness x (mm)."""
    x = np.asarray(x_mm, dtype=float)
    if abs(gamma) < GAMMA_EPS:
        return np.exp(-(alpha + beta) * x)
    r = beta / alpha
    inner = (1.0 + r) * np.exp(alpha * gamma * x) - r
    with np.errstate(invalid="ignore"):
        T = np.where(inner > 0, inner ** (-1.0 / gamma), np.nan)
    return T


def archer_thickness(T, alpha, beta, gamma):
    """Closed-form barrier thickness (mm) for required transmission T.

    Returns (thickness_mm, message). thickness is None if T is unreachable
    with the given parameters.
    """
    if not (0.0 < T <= 1.0):
        return None, "Transmission must be in (0, 1]."
    if T == 1.0:
        return 0.0, None
    if abs(gamma) < GAMMA_EPS:
        eff = alpha + beta
        if eff <= 0:
            return None, "alpha + beta must be positive in the gamma -> 0 limit."
        return -math.log(T) / eff, None
    r = beta / alpha
    numerator = T ** (-gamma) + r
    denominator = 1.0 + r
    if numerator <= 0 or denominator <= 0 or numerator / denominator <= 0:
        return None, ("Requested transmission is not reachable with these "
                      "Archer parameters (log argument non-positive).")
    x = math.log(numerator / denominator) / (alpha * gamma)
    if x < 0:
        return None, "Computed thickness is negative — check the parameters."
    return x, None


def equivalent_value_layers(alpha, beta, gamma):
    """First HVL/TVL (from T=1) and equilibrium TVL (asymptotic slope alpha)."""
    hvl1, _ = archer_thickness(0.5, alpha, beta, gamma)
    tvl1, _ = archer_thickness(0.1, alpha, beta, gamma)
    tvl_e = math.log(10.0) / alpha if alpha > 0 else None
    return hvl1, tvl1, tvl_e


# ----------------------------------------------------------------------------
# Parameter library handling
# ----------------------------------------------------------------------------
def builtin_df():
    df = pd.DataFrame(BUILTIN_PARAMS,
                      columns=PARAM_COLUMNS[:-1])
    df["Source"] = "Built-in"
    return df


def get_library():
    """Built-in + user parameters as one DataFrame."""
    if "custom_params" not in st.session_state:
        st.session_state.custom_params = pd.DataFrame(columns=PARAM_COLUMNS)
    custom = st.session_state.custom_params
    lib = pd.concat([builtin_df(), custom], ignore_index=True)
    # User entries override built-ins for the same (radionuclide, barrier) pair
    lib = lib.drop_duplicates(subset=["Radionuclide", "Barrier"], keep="last")
    return lib.reset_index(drop=True)


def add_custom_param(radionuclide, barrier, alpha, beta, gamma, r2, maxd):
    row = pd.DataFrame([[radionuclide.strip(), barrier.strip(),
                         alpha, beta, gamma, r2, maxd, "User"]],
                       columns=PARAM_COLUMNS)
    st.session_state.custom_params = pd.concat(
        [st.session_state.custom_params, row], ignore_index=True)


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.set_page_config(page_title="ShieldLab Barrier Calculator",
                   page_icon="🛡️", layout="wide")

st.title("ShieldLab Barrier Calculator")
st.caption("Barrier thickness from Archer-equation broad-beam transmission fits "
           "(GATE 10 Monte Carlo, ShieldLab parameter sweeps)")

library = get_library()

tab_calc, tab_inverse, tab_library = st.tabs(
    ["Thickness from transmission", "Transmission from thickness",
     "Parameter library"])

# ----------------------------------------------------------------------------
# Tab 1: required thickness
# ----------------------------------------------------------------------------
with tab_calc:
    col_in, col_out = st.columns([1, 2], gap="large")

    with col_in:
        nuclides = sorted(library["Radionuclide"].unique())
        nuc = st.selectbox("Radionuclide / source", nuclides, key="calc_nuc")
        barriers = sorted(
            library.loc[library["Radionuclide"] == nuc, "Barrier"].unique())
        bar = st.selectbox("Barrier material", barriers, key="calc_bar")

        row = library[(library["Radionuclide"] == nuc)
                      & (library["Barrier"] == bar)].iloc[0]
        alpha, beta, gamma = float(row.alpha_mm), float(row.beta_mm), float(row.gamma)

        mode = st.radio("Specify required transmission as",
                        ["Direct value", "From design goal"], horizontal=True)

        if mode == "Direct value":
            T_req = st.number_input(
                "Required transmission B", min_value=1e-9, max_value=1.0,
                value=0.01, format="%.6g",
                help="Fraction of unshielded air kerma / dose that may pass "
                     "the barrier, e.g. 0.01 for a factor-100 reduction.")
        else:
            st.markdown("**B = P / (D₀ · T · O)** — design goal over occupancy- "
                        "and use-weighted unshielded dose")
            P = st.number_input("Shielding design goal P (µSv per week)",
                                min_value=1e-6, value=20.0, format="%.6g",
                                help="e.g. 100 µSv/wk controlled, "
                                     "20 µSv/wk uncontrolled (NCRP 147)")
            D0 = st.number_input("Unshielded dose at point of interest "
                                 "(µSv per week)", min_value=1e-6,
                                 value=2000.0, format="%.6g")
            occ = st.number_input("Occupancy factor T", min_value=0.0001,
                                  max_value=1.0, value=1.0, format="%.4g")
            T_req = min(P / (D0 * occ), 1.0)
            st.metric("Resulting required transmission", f"{T_req:.4g}")

        st.divider()
        with st.expander("Fit parameters in use"):
            st.write(f"α = {alpha:.6g} mm⁻¹")
            st.write(f"β = {beta:.6g} mm⁻¹")
            st.write(f"γ = {gamma:.6g}")
            r2 = row.get("R2")
            maxd = row.get("MaxDeltaPct")
            if pd.notna(r2):
                st.write(f"R² = {r2}")
            if pd.notna(maxd):
                st.write(f"Max |Δ| = {maxd}%")
            st.write(f"Source: {row.get('Source', 'Built-in')}")

    with col_out:
        x_mm, err = archer_thickness(T_req, alpha, beta, gamma)
        if err:
            st.error(err)
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Required thickness", f"{x_mm:.3f} mm")
            m2.metric("", f"{x_mm / 10.0:.3f} cm")
            m3.metric("", f"{x_mm / 25.4:.4f} in")

            hvl1, tvl1, tvl_e = equivalent_value_layers(alpha, beta, gamma)
            v1, v2, v3 = st.columns(3)
            if hvl1 is not None:
                v1.metric("HVL₁", f"{hvl1:.3f} mm")
            if tvl1 is not None:
                v2.metric("TVL₁", f"{tvl1:.3f} mm")
            if tvl_e is not None:
                v3.metric("TVLₑ (ln10/α)", f"{tvl_e:.3f} mm")

            # Transmission curve with the design point marked
            x_max = max(x_mm * 1.5, (tvl1 or x_mm) * 1.2, 1.0)
            xs = np.linspace(0, x_max, 600)
            Ts = archer_transmission(xs, alpha, beta, gamma)

            fig, ax = plt.subplots(figsize=(7.5, 4.5))
            ax.semilogy(xs, Ts, lw=2, color="#1f77b4",
                        label=f"{nuc} / {bar}")
            ax.axhline(T_req, color="#d62728", ls="--", lw=1)
            ax.axvline(x_mm, color="#d62728", ls="--", lw=1)
            ax.plot([x_mm], [T_req], "o", color="#d62728", ms=8,
                    label=f"B = {T_req:.3g} → x = {x_mm:.2f} mm")
            ax.set_xlabel("Barrier thickness (mm)")
            ax.set_ylabel("Broad-beam transmission T")
            ax.set_xlim(0, x_max)
            ax.grid(True, which="both", alpha=0.3)
            ax.legend()
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

# ----------------------------------------------------------------------------
# Tab 2: forward calculation
# ----------------------------------------------------------------------------
with tab_inverse:
    col_in2, col_out2 = st.columns([1, 2], gap="large")
    with col_in2:
        nuc2 = st.selectbox("Radionuclide / source",
                            sorted(library["Radionuclide"].unique()),
                            key="fwd_nuc")
        bars2 = sorted(
            library.loc[library["Radionuclide"] == nuc2, "Barrier"].unique())
        bar2 = st.selectbox("Barrier material", bars2, key="fwd_bar")
        row2 = library[(library["Radionuclide"] == nuc2)
                       & (library["Barrier"] == bar2)].iloc[0]
        a2, b2, g2 = float(row2.alpha_mm), float(row2.beta_mm), float(row2.gamma)

        unit = st.radio("Thickness unit", ["mm", "cm", "in"], horizontal=True)
        x_in = st.number_input(f"Barrier thickness ({unit})",
                               min_value=0.0, value=10.0, format="%.6g")
        x2_mm = x_in * {"mm": 1.0, "cm": 10.0, "in": 25.4}[unit]

    with col_out2:
        T_fwd = float(archer_transmission(x2_mm, a2, b2, g2))
        if not np.isfinite(T_fwd):
            st.error("Transmission undefined at this thickness for these "
                     "parameters.")
        else:
            c1, c2 = st.columns(2)
            c1.metric("Transmission T", f"{T_fwd:.4g}")
            c2.metric("Attenuation factor 1/T",
                      f"{1.0 / T_fwd:.4g}" if T_fwd > 0 else "—")
            st.write(f"Equivalent thickness: {x2_mm:.3f} mm "
                     f"({x2_mm / 10:.3f} cm, {x2_mm / 25.4:.4f} in)")

# ----------------------------------------------------------------------------
# Tab 3: parameter library / add new fits
# ----------------------------------------------------------------------------
with tab_library:
    st.subheader("Current parameter library")
    st.dataframe(
        library.rename(columns={"alpha_mm": "α (mm⁻¹)", "beta_mm": "β (mm⁻¹)",
                                "gamma": "γ", "MaxDeltaPct": "Max |Δ| (%)"}),
        use_container_width=True, hide_index=True)

    csv_buf = io.StringIO()
    library.to_csv(csv_buf, index=False)
    st.download_button("Download library as CSV", csv_buf.getvalue(),
                       file_name="shieldlab_archer_params.csv",
                       mime="text/csv")

    st.divider()
    st.subheader("Add a new isotope / barrier fit")
    st.caption("New entries with the same radionuclide + barrier pair "
               "override the built-in values for this session.")

    with st.form("add_param", clear_on_submit=True):
        c1, c2 = st.columns(2)
        new_nuc = c1.text_input("Radionuclide / source (e.g. Lu-177)")
        new_bar = c2.text_input("Barrier material (e.g. Lead)")
        c3, c4, c5 = st.columns(3)
        new_a = c3.number_input("α (mm⁻¹)", value=0.0, format="%.6g")
        new_b = c4.number_input("β (mm⁻¹)", value=0.0, format="%.6g")
        new_g = c5.number_input("γ", value=1.0, format="%.6g")
        c6, c7 = st.columns(2)
        new_r2 = c6.number_input("R² (optional)", value=0.0, min_value=0.0,
                                 max_value=1.0, format="%.4g")
        new_maxd = c7.number_input("Max |Δ| % (optional)", value=0.0,
                                   format="%.4g")
        submitted = st.form_submit_button("Add to library")

    if submitted:
        if not new_nuc or not new_bar:
            st.error("Radionuclide and barrier names are required.")
        elif new_a <= 0:
            st.error("α must be positive (asymptotic attenuation coefficient).")
        else:
            add_custom_param(new_nuc, new_bar, new_a, new_b, new_g,
                             new_r2 if new_r2 > 0 else np.nan,
                             new_maxd if new_maxd > 0 else np.nan)
            st.success(f"Added {new_nuc} / {new_bar}. "
                       "It is now selectable in the calculator tabs.")
            st.rerun()

    st.divider()
    st.subheader("Import parameters from CSV")
    st.caption("Columns required: Radionuclide, Barrier, alpha_mm, beta_mm, "
               "gamma. Optional: R2, MaxDeltaPct. Same format as the CSV "
               "export above.")
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up is not None:
        try:
            imported = pd.read_csv(up)
            required = {"Radionuclide", "Barrier", "alpha_mm", "beta_mm",
                        "gamma"}
            missing = required - set(imported.columns)
            if missing:
                st.error(f"Missing columns: {', '.join(sorted(missing))}")
            else:
                for opt in ("R2", "MaxDeltaPct"):
                    if opt not in imported.columns:
                        imported[opt] = np.nan
                imported["Source"] = "Imported"
                imported = imported[PARAM_COLUMNS]
                st.session_state.custom_params = pd.concat(
                    [st.session_state.custom_params, imported],
                    ignore_index=True)
                st.success(f"Imported {len(imported)} entries.")
                st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not read CSV: {exc}")

    if not st.session_state.custom_params.empty:
        st.divider()
        if st.button("Clear all user-added parameters"):
            st.session_state.custom_params = pd.DataFrame(
                columns=PARAM_COLUMNS)
            st.rerun()
