# app.py
# Streamlit app
# v2.01: version label + lambda regime toggle + right plot shows Id_agg + pivotal country demand

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

import integration_model_EU27_xres_final as model

APP_VERSION = "v2.01"


# --------------------------
# Helpers
# --------------------------

def safe_model_version():
    return getattr(model, "MODEL_VERSION", "UNKNOWN")


def compute_pivotal_series_optionB(Id_c_t: np.ndarray) -> tuple[int, float]:
    """
    Sign-consensus pivotal logic (Option B) on current country demand vector Id_c(t):
      - if all Id >= 0: pivot is argmin(Id) (most reluctant among supporters)
      - if all Id <= 0: pivot is argmax(Id) (least negative among blockers)
      - else (mixed signs): status quo; show strongest blocker = argmin(Id) (most negative)
    Returns (pivot_index, pivot_id_value)
    """
    if np.all(Id_c_t >= 0):
        idx = int(np.argmin(Id_c_t))
        return idx, float(Id_c_t[idx])
    if np.all(Id_c_t <= 0):
        idx = int(np.argmax(Id_c_t))
        return idx, float(Id_c_t[idx])

    idx = int(np.argmin(Id_c_t))
    return idx, float(Id_c_t[idx])


def build_CP_matrix(
    scenario: str,
    C: int,
    T: int,
    shock_level: float,
    crisis_periods: list[int],
    crisis_countries: list[str],
    codes: list[str],
) -> np.ndarray:
    CP = np.zeros((C, T + 1), dtype=float)

    if scenario == "Symmetric crisis":
        for t in crisis_periods:
            if 0 <= t <= T:
                CP[:, t] = float(shock_level)

    elif scenario == "Asymmetric crisis":
        code_to_idx = {c: i for i, c in enumerate(codes)}
        idxs = [code_to_idx[c] for c in crisis_countries if c in code_to_idx]
        for t in crisis_periods:
            if 0 <= t <= T:
                CP[idxs, t] = float(shock_level)

    return CP


# --------------------------
# App
# --------------------------

st.set_page_config(page_title="EU Integration Dynamics Simulator", layout="wide")
st.title("EU Integration Dynamics Simulator")
st.caption(f"App version: {APP_VERSION} | Model version: {safe_model_version()}")

DF_ALL = pd.DataFrame(model.COUNTRY_DATA)

st.markdown(
    """
Interactive playground for the EU integration dynamics model.

Use the sidebar to:
1. Choose **horizon T** and which **countries** are in the simulation.
2. Specify whether the crisis is **symmetric** or **asymmetric**.
3. Tick which **countries** and **periods** are hit in an asymmetric crisis.
4. Tune the structural parameters using the canonical notation.
"""
)

# ===============================================================
# SIDEBAR: SIMULATION SCOPE (T and country set)
# ===============================================================

st.sidebar.header("Simulation scope")

sim_T = st.sidebar.slider(
    "Number of periods T",
    min_value=5,
    max_value=100,
    value=int(getattr(model, "T", 30)),
    step=1,
    key="sim_T",
)

model.T = sim_T
T = sim_T

country_options = DF_ALL["Code"].tolist()
selected_countries = st.sidebar.multiselect(
    "Countries included in simulation",
    options=country_options,
    default=country_options,
    key="country_selection",
)

if not selected_countries:
    st.sidebar.warning("Select at least one country; reverting to all 27.")
    selected_countries = country_options

df_countries = DF_ALL[DF_ALL["Code"].isin(selected_countries)].reset_index(drop=True)

model.COUNTRY_DATA = df_countries.to_dict("records")
model.C = len(df_countries)
C = model.C

if hasattr(model, "set_weights_from_df"):
    try:
        model.set_weights_from_df(df_countries)
    except Exception:
        pass

# ===============================================================
# SIDEBAR: CRISIS SETTINGS
# ===============================================================

st.sidebar.header("Crisis settings")

scenario = st.sidebar.selectbox(
    "Crisis pattern",
    [
        "No crisis",
        "Symmetric crisis",
        "Asymmetric crisis",
    ],
    key="scenario",
)

shock_level = st.sidebar.slider(
    "Shock intensity during crisis periods (CP level)",
    min_value=0.0,
    max_value=0.50,
    value=0.15,
    step=0.01,
    key="shock_level",
)

if scenario != "No crisis":
    period_options = list(range(1, T + 1))
    default_periods = [t for t in [4, 5, 6] if t <= T] or period_options[: min(3, len(period_options))]
    crisis_periods = st.sidebar.multiselect(
        "Crisis periods (select values of t with CP > 0)",
        options=period_options,
        default=default_periods,
        key="crisis_periods",
    )
else:
    crisis_periods = []

if scenario == "Asymmetric crisis":
    crisis_country_options = df_countries["Code"].tolist()
    default_crisis_countries = [c for c in ["IT", "ES", "EL"] if c in crisis_country_options] \
        or crisis_country_options[: min(3, len(crisis_country_options))]
    crisis_countries = st.sidebar.multiselect(
        "Countries hit by the asymmetric crisis",
        options=crisis_country_options,
        default=default_crisis_countries,
        key="crisis_countries",
    )
    if not crisis_countries:
        st.sidebar.warning("Select at least one country to be hit; reverting to default.")
        crisis_countries = default_crisis_countries
else:
    crisis_countries = []

# ===============================================================
# SIDEBAR: POLITICAL SETTINGS (θ regime + voting rule + EC + λ toggle)
# ===============================================================

st.sidebar.header("Political settings")

theta_mode_label = st.sidebar.selectbox(
    "θ regime (performance-gap coefficient)",
    ["fixed", "endogenous (1 - CP)"],
)

voting_rule = st.sidebar.selectbox(
    "Voting rule for aggregation of demands",
    ["majority", "unanimity"],
)

use_EC = st.sidebar.checkbox(
    "Allow endogenous integration capacity (EC)",
    value=True,
)

# ---- (change #2) lambda regime toggle ----
lambda_regime = st.sidebar.selectbox(
    "λ regime",
    ["Endogenous λ (default)", "Force λ = 1"],
    index=0,
)
force_lambda_one = (lambda_regime == "Force λ = 1")

theta_mode_internal = "fixed" if theta_mode_label == "fixed" else "endogenous"

st.sidebar.markdown("---")

# ===============================================================
# SIDEBAR: MODEL PARAMETERS (CANONICAL NOTATION)
# ===============================================================

with st.sidebar.expander("Model parameters (canonical notation)", expanded=False):

    p = st.slider("p  (overall policy sovereignty quotient)", 0.0, 1.0, float(getattr(model, "p", 0.5)), 0.01)
    pD = st.slider("p_d  (policy sovereignty of democratic policies)", 0.0, 1.0, float(getattr(model, "pD", 0.5)), 0.01)
    D_level = st.slider("D  (level of democratic control)", 0.0, 1.0, float(getattr(model, "D_level", 0.5)), 0.01)
    alpha_p0 = st.slider("α_p0  (baseline sovereignty coefficient)", 0.0, 2.0, float(getattr(model, "alpha_p0", 1.0)), 0.05)
    gamma_alpha = st.slider("γ_α  (mobilisation elasticity of sovereignty)", 0.0, 1.0, float(getattr(model, "gamma_alpha", 0.5)), 0.01)
    alpha_D = st.slider("α_D  (relevance of democratic control on sovereignty)", 0.0, 2.0, float(getattr(model, "alpha_D", 1.0)), 0.05)

    ie = st.slider("Ie  (ecological interdependence)", 0.0, 1.0, float(getattr(model, "ie", 0.1)), 0.01)
    is_ = st.slider("Is  (strategic interdependence)", 0.0, 1.0, float(getattr(model, "is_", 0.1)), 0.01)
    ip = st.slider("Ip  (policy interdependence)", 0.0, 1.0, float(getattr(model, "ip", 0.1)), 0.01)

    beta_A = st.slider("β_A  (output legitimacy coefficient)", 0.0, 2.0, float(getattr(model, "beta_A", 1.0)), 0.05)
    betaE0 = st.slider("β_e  (baseline mobilisation coefficient)", 0.0, 2.0, float(getattr(model, "betaE0", 1.0)), 0.05)
    eta_A = st.slider("η_A  (output legitimacy coefficient of identity)", 0.0, 1.0, float(getattr(model, "eta_A", 0.5)), 0.01)
    eta_S = st.slider("η_S  (shared experience coefficient of identity)", 0.0, 1.0, float(getattr(model, "eta_S", 0.5)), 0.01)

    kappa_long = st.slider("k_long  (long-term mobilisation coefficient)", 0.0, 2.0, float(getattr(model, "kappa_long", 1.0)), 0.05)
    kappa_short = st.slider("k_short  (short-term mobilisation coefficient)", 0.0, 2.0, float(getattr(model, "kappa_short", 1.0)), 0.05)

    alpha_cp = st.slider("α_IP  (crisis → integration pressure weight)", 0.0, 2.0, float(getattr(model, "alpha_cp", 1.0)), 0.05)
    alpha_pf = st.slider("ω  (propagation / spillover weight)", 0.0, 2.0, float(getattr(model, "alpha_pf", 1.0)), 0.05)

    gamma_crc = st.slider("γ_CRC  (counter-crisis elasticity)", 0.0, 2.0, float(getattr(model, "gamma_crc", 1.0)), 0.05)
    x_res = st.slider("x  (EU-level crisis resolution capacity)", 0.0, 1.0, float(getattr(model, "x_res", 0.5)), 0.01)

    theta_fixed = st.slider("θ  (performance-gap coefficient)", 0.0, 1.0, float(getattr(model, "theta_fixed", 0.5)), 0.01)

# Update model globals
model.p = p
model.pD = pD if pD <= p else p
model.D_level = D_level
model.alpha_p0 = alpha_p0
model.gamma_alpha = gamma_alpha
model.alpha_D = alpha_D
model.ie = ie
model.is_ = is_
model.ip = ip
model.beta_A = beta_A
model.betaE0 = betaE0
model.eta_A = eta_A
model.eta_S = eta_S
model.kappa_long = kappa_long
model.kappa_short = kappa_short
model.alpha_cp = alpha_cp
model.alpha_pf = alpha_pf
model.gamma_crc = gamma_crc
model.x_res = x_res
model.theta_fixed = theta_fixed
model.K = (model.ie + model.is_ + model.ip) / 3.0

st.sidebar.markdown("---")

# ===============================================================
# BUILD CP
# ===============================================================

codes = df_countries["Code"].tolist()
CP = build_CP_matrix(
    scenario=scenario,
    C=C,
    T=T,
    shock_level=shock_level,
    crisis_periods=[int(x) for x in crisis_periods],
    crisis_countries=crisis_countries,
    codes=codes,
)

# ===============================================================
# RUN SIMULATION (unchanged, except optional force_lambda_one)
# ===============================================================

try:
    results = model.run_simulation(
        CP=CP,
        use_EC=use_EC,
        theta_mode=theta_mode_internal,
        voting_rule=voting_rule,
        force_lambda_one=force_lambda_one,
    )
except TypeError:
    results = model.run_simulation(
        CP=CP,
        use_EC=use_EC,
        theta_mode=theta_mode_internal,
        voting_rule=voting_rule,
    )
    # best-effort display override if model doesn't support it internally
    if force_lambda_one and ("lambda" in results):
        results["lambda"] = np.ones_like(np.asarray(results["lambda"]), dtype=float)

I = np.asarray(results["I"])
I_star = np.asarray(results["I_star"])
Id_agg = np.asarray(results["Id_agg"])
lambda_t = np.asarray(results["lambda"])

CRC = results.get("CRC", None)
IP_c = results.get("IP", None)
SP_c = results.get("SP", None)
PF_c = results.get("PF", None)
CP_out = results.get("CP", CP)

# ===============================================================
# HORIZON SLIDER (DISPLAYED T)
# ===============================================================

max_h = len(I) - 1
H = st.sidebar.slider(
    "Horizon to display (last period t)",
    min_value=1,
    max_value=max_h,
    value=max_h,
    key="horizon_display",
)
idx = np.arange(H + 1)
time = idx

I_plot = I[idx]
I_star_plot = I_star[idx]
Id_agg_plot = Id_agg[idx]
lambda_plot = lambda_t[idx]

# ===============================================================
# MAIN PLOTS
# ===============================================================

col1, col2 = st.columns(2)

with col1:
    st.subheader("Integration level I(t) and functional target I*(t)")
    fig, ax = plt.subplots()
    ax.plot(time, I_plot, label="I(t) – actual integration")
    ax.plot(time, I_star_plot, linestyle="--", label="I*(t) – functional target")
    ax.set_xlabel("Time (t)")
    ax.set_ylabel("Integration level")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle=":", linewidth=0.5)
    ax.legend()
    st.pyplot(fig)

with col2:
    st.subheader("Aggregate demand Id_agg(t) and pivotal/constraint country demand")

    IP = np.asarray(IP_c) if IP_c is not None else None
    SP = np.asarray(SP_c) if SP_c is not None else None
    if IP is None or SP is None or IP.ndim != 2 or SP.ndim != 2:
        st.error("Model output must include 2D arrays 'IP' and 'SP' to plot pivotal demand.")
    else:
        Id_c = IP - SP

        Tplot = len(time)
        if Id_c.shape[1] < Tplot:
            Tplot = Id_c.shape[1]

        piv_codes = []
        Id_piv = np.zeros(Tplot, dtype=float)

        for t in range(Tplot):
            idx_p, idv = compute_pivotal_series_optionB(Id_c[:, t])
            piv_codes.append(codes[idx_p])
            Id_piv[t] = idv

        change = np.zeros(Tplot, dtype=bool)
        change[0] = True
        for t in range(1, Tplot):
            change[t] = (piv_codes[t] != piv_codes[t - 1])

        fig2, ax2 = plt.subplots()
        ax2.axhline(0.0, linestyle=":", linewidth=0.8)
        ax2.plot(time[:Tplot], Id_agg_plot[:Tplot], label="Id_agg(t)")
        ax2.plot(time[:Tplot], Id_piv, linestyle="--", label="Id_piv(t) – pivotal/constraint country")

        for t in range(Tplot):
            if change[t]:
                ax2.axvline(time[t], alpha=0.25)
                ax2.text(time[t], Id_piv[t], f" {piv_codes[t]}", fontsize=9, va="bottom")

        ax2.set_xlabel("Time (t)")
        ax2.set_ylabel("Demand")
        ax2.grid(True, linestyle=":", linewidth=0.5)
        ax2.legend(loc="upper right")
        st.pyplot(fig2)

# ===============================================================
# OPTIONAL: CRISIS PATTERN VIEW
# ===============================================================

with st.expander("Show crisis pressure CP(t) for a few countries"):
    n_show = min(5, C)
    sample_df = df_countries.head(n_show).copy()
    idxs_c = sample_df.index.to_list()
    cp_subset = np.asarray(CP_out)[idxs_c, :][:, idx]

    fig3, ax4 = plt.subplots()
    for i_c, idx_c in enumerate(idxs_c):
        ax4.plot(time, cp_subset[i_c, :], label=sample_df.loc[idx_c, "Code"])
    ax4.set_xlabel("Time (t)")
    ax4.set_ylabel("Crisis pressure CP")
    ax4.grid(True, linestyle=":", linewidth=0.5)
    ax4.legend()
    st.pyplot(fig3)

# ===============================================================
# PER-COUNTRY COMPONENTS
# ===============================================================

with st.expander("Per-country components (IP, SP, PF, CRC)"):
    component = st.selectbox(
        "Component to show",
        [
            "IP (integration pressure)",
            "SP (status-quo pressure)",
            "PF (spillover pressure)",
            "CRC (counter-crisis capacity)",
        ],
        key="component_select",
    )

    codes_all = df_countries["Code"].tolist()
    default_selection = [c for c in ["DE", "FR", "IT"] if c in codes_all]
    selected_codes = st.multiselect(
        "Countries to display",
        options=codes_all,
        default=default_selection if default_selection else codes_all[:3],
        key="component_countries",
    )

    comp_map = {
        "IP (integration pressure)": IP_c,
        "SP (status-quo pressure)": SP_c,
        "PF (spillover pressure)": PF_c,
        "CRC (counter-crisis capacity)": CRC,
    }
    data = comp_map.get(component, None)

    if data is None:
        st.info("Selected component is not available in the results.")
    else:
        data = np.asarray(data)
        figc, axc = plt.subplots()
        for code in selected_codes:
            if code in codes_all:
                idx_c = codes_all.index(code)
                series_c = data[idx_c, :][idx]
                axc.plot(time, series_c, label=code)
        axc.set_xlabel("Time (t)")
        axc.set_ylabel(component)
        axc.grid(True, linestyle=":", linewidth=0.5)
        axc.legend()
        st.pyplot(figc)

# ===============================================================
# DATA TABLE & CSV EXPORT
# ===============================================================

st.subheader("Simulation data (first 15 periods of selected horizon)")

summary_df = pd.DataFrame(
    {
        "t": time,
        "I": I_plot,
        "I_star": I_star_plot,
        "Id_agg": Id_agg_plot,
        "lambda": lambda_plot,
    }
)

st.dataframe(summary_df.head(15))

st.markdown("### Download results as CSV")

scenario_label = st.text_input(
    "Scenario label (for filename)",
    value="scenario",
    key="scenario_label",
)

full_df = pd.DataFrame(
    {
        "t": time,
        "I": I_plot,
        "I_star": I_star_plot,
        "Id_agg": Id_agg_plot,
        "lambda": lambda_plot,
    }
)

csv_bytes = full_df.to_csv(index=False).encode("utf-8")
csv_name = f"results_{(scenario_label or 'scenario').replace(' ', '_')}_T{H}.csv"

st.download_button(
    label="Download current horizon results as CSV",
    data=csv_bytes,
    file_name=csv_name,
    mime="text/csv",
)
