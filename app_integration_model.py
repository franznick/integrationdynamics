import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

import integration_model_EU27_xres_final as model

# --------------------------
# BASIC CONSTANT REFERENCES (will be overwritten by user choices)
# --------------------------

C_default = model.C
T_default = model.T
DF_ALL = pd.DataFrame(model.COUNTRY_DATA)

st.set_page_config(page_title="EU Integration Dynamics Simulator", layout="wide")

st.title("EU Integration Dynamics Simulator")

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

# 1) Horizon T (periods)
sim_T = st.sidebar.slider(
    "Number of periods T",
    min_value=5,
    max_value=100,
    value=T_default,
    step=1,
    key="sim_T",
)

# Update model horizon
model.T = sim_T
T = sim_T

# 2) Country selection
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

# Update model country set
model.COUNTRY_DATA = df_countries.to_dict("records")
model.C = len(df_countries)
C = model.C

# Recompute weights with the active set
omega_econ, omega_pol, omega_f, w = model.set_weights_from_df(df_countries)

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

# 3) Crisis periods: tick individual periods
if scenario != "No crisis":
    period_options = list(range(1, T + 1))
    default_periods = [t for t in [4, 5, 6] if t <= T] or period_options[:min(3, len(period_options))]
    crisis_periods = st.sidebar.multiselect(
        "Crisis periods (select values of t with CP > 0)",
        options=period_options,
        default=default_periods,
        key="crisis_periods",
    )
else:
    crisis_periods = []

# 4) Crisis countries (for asymmetric scenario)
if scenario == "Asymmetric crisis":
    crisis_country_options = df_countries["Code"].tolist()
    default_crisis_countries = [c for c in ["IT", "ES", "EL"] if c in crisis_country_options] \
                               or crisis_country_options[:min(3, len(crisis_country_options))]
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
# SIDEBAR: POLITICAL SETTINGS (θ regime)
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

theta_mode_internal = "fixed" if theta_mode_label == "fixed" else "endogenous"

st.sidebar.markdown("---")

# ===============================================================
# SIDEBAR: MODEL PARAMETERS (CANONICAL NOTATION)
# ===============================================================

with st.sidebar.expander("Model parameters (canonical notation)", expanded=False):

    # --- Sovereignty / Political ---

    p = st.slider("p  (overall policy sovereignty quotient)",
                  0.0, 1.0, float(model.p), 0.01)

    pD = st.slider("p_d  (policy sovereignty of democratic policies)",
                   0.0, 1.0, float(model.pD), 0.01)

    D_level = st.slider("D  (level of democratic control)",
                        0.0, 1.0, float(model.D_level), 0.01)

    alpha_p0 = st.slider("α_p0  (baseline sovereignty coefficient)",
                         0.0, 2.0, float(model.alpha_p0), 0.05)

    gamma_alpha = st.slider("γ_α  (mobilisation elasticity of sovereignty)",
                            0.0, 1.0, float(model.gamma_alpha), 0.01)

    alpha_D = st.slider("α_D  (relevance of democratic control on sovereignty)",
                        0.0, 2.0, float(model.alpha_D), 0.05)


    # --- Interdependence ---

    ie = st.slider("Ie  (ecological interdependence)",
                   0.0, 1.0, float(model.ie), 0.01)

    is_ = st.slider("Is  (strategic interdependence)",
                    0.0, 1.0, float(model.is_), 0.01)

    ip = st.slider("Ip  (policy interdependence)",
                   0.0, 1.0, float(model.ip), 0.01)


    # --- Identity / Legitimacy ---

    beta_A = st.slider("β_A  (output legitimacy coefficient)",
                       0.0, 2.0, float(model.beta_A), 0.05)

    betaE0 = st.slider("β_e  (baseline mobilisation coefficient)",
                       0.0, 2.0, float(model.betaE0), 0.05)

    eta_A = st.slider("η_A  (output legitimacy coefficient of identity)",
                      0.0, 1.0, float(model.eta_A), 0.01)

    eta_S = st.slider("η_S  (shared experience coefficient of identity)",
                      0.0, 1.0, float(model.eta_S), 0.01)


    # --- Endogenous mobilisation ---

    kappa_long = st.slider("k_long  (long-term mobilisation coefficient)",
                           0.0, 2.0, float(model.kappa_long), 0.05)

    kappa_short = st.slider("k_short  (short-term mobilisation coefficient)",
                            0.0, 2.0, float(model.kappa_short), 0.05)


    # --- Crisis & integration pressure weights ---

    alpha_cp = st.slider("α_IP  (crisis → integration pressure weight)",
                         0.0, 2.0, float(model.alpha_cp), 0.05)

    alpha_pf = st.slider("ω  (propagation / spillover weight)",
                         0.0, 2.0, float(model.alpha_pf), 0.05)


    # --- Crisis response capacity ---

    gamma_crc = st.slider("γ_CRC  (counter-crisis elasticity)",
                          0.0, 2.0, float(model.gamma_crc), 0.05)

    x_res = st.slider("x  (EU-level crisis resolution capacity)",
                      0.0, 1.0, float(model.x_res), 0.01)


    # --- Cohesion / performance gap ---

    theta_fixed = st.slider("θ  (performance-gap coefficient)",
                            0.0, 1.0, float(model.theta_fixed), 0.01)


# Update model globals with canonical parameters
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

# recompute K as in your original script (Ie, Is, Ip)
model.K = (model.ie + model.is_ + model.ip) / 3.0

st.sidebar.markdown("---")

# ===============================================================
# BUILD CRISIS PRESSURE MATRIX CP
# ===============================================================

CP = np.zeros((C, T + 1))

if scenario == "Symmetric crisis":
    for t in crisis_periods:
        if 0 <= t <= T:
            CP[:, t] = shock_level

elif scenario == "Asymmetric crisis":
    code_to_idx = {code: i for i, code in enumerate(df_countries["Code"].tolist())}
    crisis_idx = [code_to_idx[c] for c in crisis_countries if c in code_to_idx]
    for t in crisis_periods:
        if 0 <= t <= T:
            CP[crisis_idx, t] = shock_level

# (No crisis scenario leaves CP as zeros)

# ===============================================================
# RUN SIMULATION
# ===============================================================

results = model.run_simulation(
    CP=CP,
    use_EC=use_EC,
    theta_mode=theta_mode_internal,
    voting_rule=voting_rule,
)

I = results["I"]
I_star = results["I_star"]
Id_agg = results["Id_agg"]
lambda_t = results["lambda"]

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
    st.subheader("Aggregate demand Id_agg(t) and political capacity λ(t)")
    fig2, ax2 = plt.subplots()
    ax2.plot(time, Id_agg_plot, label="Id_agg(t)")
    ax2.axhline(0.0, linestyle=":", linewidth=0.8)
    ax2.set_xlabel("Time (t)")
    ax2.set_ylabel("Aggregate demand Id_agg")
    ax2.grid(True, linestyle=":", linewidth=0.5)

    ax3 = ax2.twinx()
    ax3.plot(time, lambda_plot, linestyle="--", label="λ(t)")
    ax3.set_ylabel("Political capacity λ(t)")

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax3.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    st.pyplot(fig2)

# ===============================================================
# OPTIONAL: CRISIS PATTERN VIEW
# ===============================================================

with st.expander("Show crisis pressure CP(t) for a few countries"):
    n_show = min(5, C)
    sample_df = df_countries.head(n_show).copy()
    idxs_c = sample_df.index.to_list()
    cp_subset = CP_out[idxs_c, :][:, idx]

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
        figc, axc = plt.subplots()
        for code in selected_codes:
            if code in codes_all:
                idx_c = codes_all.index(code)
                series = data[idx_c, :][idx]
                axc.plot(time, series, label=code)
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

st.markdown(
    """
To explore more:
- Adjust the **T slider** and crisis periods to shape the scenario.
- Restrict the simulation to a **subset of countries** in the scope panel.
- Use the **per-country components** view to see how IP, SP, PF, and CRC evolve.
"""
)
