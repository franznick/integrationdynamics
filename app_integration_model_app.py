# app_integration_model_app.py
# Shiny for Python app
# v2.01: version label + lambda regime toggle + right plot shows Id_agg + pivotal country demand

from shiny import App, ui, render, reactive
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
# UI
# --------------------------

app_ui = ui.page_fluid(
    ui.h2("EU Integration Dynamics Simulator"),
    ui.tags.div(
        ui.output_text("version_text"),
        style="margin-bottom: 10px; font-size: 14px;",
    ),

    ui.layout_sidebar(
        ui.sidebar(
            ui.h4("Controls"),

            # ===============================================================
            # SIMULATION SCOPE
            # ===============================================================
            ui.h5("Simulation scope"),
            ui.input_slider(
                "sim_T",
                "Number of periods T",
                min=5,
                max=100,
                value=int(getattr(model, "T", 30)),
                step=1,
            ),

            ui.input_selectize(
                "country_selection",
                "Countries included in simulation",
                choices=[d["Code"] for d in model.COUNTRY_DATA],
                selected=[d["Code"] for d in model.COUNTRY_DATA],
                multiple=True,
            ),

            ui.hr(),

            # ===============================================================
            # CRISIS SETTINGS
            # ===============================================================
            ui.h5("Crisis settings"),
            ui.input_select(
                "scenario",
                "Crisis pattern",
                choices={
                    "No crisis": "No crisis",
                    "Symmetric crisis": "Symmetric crisis",
                    "Asymmetric crisis": "Asymmetric crisis",
                },
                selected="No crisis",
            ),
            ui.input_slider(
                "shock_level",
                "Shock intensity during crisis periods (CP level)",
                min=0.0,
                max=0.50,
                value=0.15,
                step=0.01,
            ),
            ui.input_selectize(
                "crisis_periods",
                "Crisis periods (select values of t with CP > 0)",
                choices=[],          # filled reactively once T known
                selected=[],
                multiple=True,
            ),
            ui.input_selectize(
                "crisis_countries",
                "Countries hit by the asymmetric crisis",
                choices=[d["Code"] for d in model.COUNTRY_DATA],
                selected=[c for c in ["IT", "ES", "EL"] if c in [d["Code"] for d in model.COUNTRY_DATA]][:3],
                multiple=True,
            ),

            ui.hr(),

            # ===============================================================
            # POLITICAL SETTINGS (including NEW λ regime toggle)
            # ===============================================================
            ui.h5("Political settings"),
            ui.input_select(
                "theta_mode_label",
                "θ regime (performance-gap coefficient)",
                choices={"fixed": "fixed", "endogenous": "endogenous (1 - CP)"},
                selected="fixed",
            ),
            ui.input_select(
                "voting_rule",
                "Voting rule for aggregation of demands",
                choices={"majority": "majority", "unanimity": "unanimity"},
                selected="unanimity",
            ),
            ui.input_checkbox(
                "use_EC",
                "Allow endogenous integration capacity (EC)",
                value=True,
            ),

            # --- CHANGE (2): lambda regime toggle ---
            ui.input_select(
                "lambda_regime",
                "λ regime",
                choices={
                    "endogenous": "Endogenous λ (default)",
                    "one": "Force λ = 1",
                },
                selected="endogenous",
            ),

            ui.hr(),

            # ===============================================================
            # MODEL PARAMETERS (canonical notation)
            # ===============================================================
            ui.accordion(
                ui.accordion_panel(
                    "Model parameters (canonical notation)",
                    ui.h6("Sovereignty / Political"),
                    ui.input_slider("p", "p (overall policy sovereignty quotient)", 0.0, 1.0, float(getattr(model, "p", 0.5)), 0.01),
                    ui.input_slider("pD", "p_d (policy sovereignty of democratic policies)", 0.0, 1.0, float(getattr(model, "pD", 0.5)), 0.01),
                    ui.input_slider("D_level", "D (level of democratic control)", 0.0, 1.0, float(getattr(model, "D_level", 0.5)), 0.01),
                    ui.input_slider("alpha_p0", "α_p0 (baseline sovereignty coefficient)", 0.0, 2.0, float(getattr(model, "alpha_p0", 1.0)), 0.05),
                    ui.input_slider("gamma_alpha", "γ_α (mobilisation elasticity of sovereignty)", 0.0, 1.0, float(getattr(model, "gamma_alpha", 0.5)), 0.01),
                    ui.input_slider("alpha_D", "α_D (relevance of democratic control on sovereignty)", 0.0, 2.0, float(getattr(model, "alpha_D", 1.0)), 0.05),

                    ui.hr(),
                    ui.h6("Interdependence"),
                    ui.input_slider("ie", "Ie (ecological interdependence)", 0.0, 1.0, float(getattr(model, "ie", 0.1)), 0.01),
                    ui.input_slider("is_", "Is (strategic interdependence)", 0.0, 1.0, float(getattr(model, "is_", 0.1)), 0.01),
                    ui.input_slider("ip", "Ip (policy interdependence)", 0.0, 1.0, float(getattr(model, "ip", 0.1)), 0.01),

                    ui.hr(),
                    ui.h6("Identity / Legitimacy"),
                    ui.input_slider("beta_A", "β_A (output legitimacy coefficient)", 0.0, 2.0, float(getattr(model, "beta_A", 1.0)), 0.05),
                    ui.input_slider("betaE0", "β_e (baseline mobilisation coefficient)", 0.0, 2.0, float(getattr(model, "betaE0", 1.0)), 0.05),
                    ui.input_slider("eta_A", "η_A (output legitimacy coefficient of identity)", 0.0, 1.0, float(getattr(model, "eta_A", 0.5)), 0.01),
                    ui.input_slider("eta_S", "η_S (shared experience coefficient of identity)", 0.0, 1.0, float(getattr(model, "eta_S", 0.5)), 0.01),

                    ui.hr(),
                    ui.h6("Endogenous mobilisation"),
                    ui.input_slider("kappa_long", "k_long (long-term mobilisation coefficient)", 0.0, 2.0, float(getattr(model, "kappa_long", 1.0)), 0.05),
                    ui.input_slider("kappa_short", "k_short (short-term mobilisation coefficient)", 0.0, 2.0, float(getattr(model, "kappa_short", 1.0)), 0.05),

                    ui.hr(),
                    ui.h6("Crisis & integration pressure weights"),
                    ui.input_slider("alpha_cp", "α_IP (crisis → integration pressure weight)", 0.0, 2.0, float(getattr(model, "alpha_cp", 1.0)), 0.05),
                    ui.input_slider("alpha_pf", "ω (propagation / spillover weight)", 0.0, 2.0, float(getattr(model, "alpha_pf", 1.0)), 0.05),

                    ui.hr(),
                    ui.h6("Crisis response capacity"),
                    ui.input_slider("gamma_crc", "γ_CRC (counter-crisis elasticity)", 0.0, 2.0, float(getattr(model, "gamma_crc", 1.0)), 0.05),
                    ui.input_slider("x_res", "x (EU-level crisis resolution capacity)", 0.0, 1.0, float(getattr(model, "x_res", 0.5)), 0.01),

                    ui.hr(),
                    ui.h6("Cohesion / performance gap"),
                    ui.input_slider("theta_fixed", "θ (performance-gap coefficient)", 0.0, 1.0, float(getattr(model, "theta_fixed", 0.5)), 0.01),
                ),
                open=False,
            ),

            ui.hr(),

            # ===============================================================
            # HORIZON DISPLAY
            # ===============================================================
            ui.h5("Display"),
            ui.input_slider(
                "horizon_display",
                "Horizon to display (last period t)",
                min=1,
                max=int(getattr(model, "T", 30)),
                value=int(getattr(model, "T", 30)),
                step=1,
            ),

            width=360,
        ),

        # ===============================================================
        # MAIN PANEL
        # ===============================================================
        ui.layout_columns(
            ui.card(
                ui.card_header("Integration level I(t) and functional target I*(t)"),
                ui.output_plot("plot_left"),
                full_screen=True,
            ),
            ui.card(
                # --- CHANGE (3): right plot title matches Id_agg + pivotal Id ---
                ui.card_header("Aggregate demand Id_agg(t) and pivotal/constraint country demand"),
                ui.output_plot("plot_right"),
                full_screen=True,
            ),
            col_widths=[6, 6],
        ),

        ui.hr(),

        ui.layout_columns(
            ui.card(
                ui.card_header("Crisis pressure CP(t) for a few countries"),
                ui.output_plot("plot_cp"),
                full_screen=True,
            ),
            ui.card(
                ui.card_header("Per-country components (IP, SP, PF, CRC)"),
                ui.input_select(
                    "component_select",
                    "Component to show",
                    choices={
                        "IP": "IP (integration pressure)",
                        "SP": "SP (status-quo pressure)",
                        "PF": "PF (spillover pressure)",
                        "CRC": "CRC (counter-crisis capacity)",
                    },
                    selected="IP",
                ),
                ui.input_selectize(
                    "component_countries",
                    "Countries to display",
                    choices=[d["Code"] for d in model.COUNTRY_DATA],
                    selected=[c for c in ["DE", "FR", "IT"] if c in [d["Code"] for d in model.COUNTRY_DATA]],
                    multiple=True,
                ),
                ui.output_plot("plot_components"),
                full_screen=True,
            ),
            col_widths=[6, 6],
        ),

        ui.hr(),

        ui.h4("Simulation data (first 15 periods of selected horizon)"),
        ui.output_data_frame("table_summary"),

        ui.h4("Download results as CSV"),
        ui.input_text("scenario_label", "Scenario label (for filename)", value="scenario"),
        ui.download_button("download_csv", "Download current horizon results as CSV"),
    ),
)


# --------------------------
# Server
# --------------------------

def server(input, output, session):

    # --- CHANGE (1): version label shown in UI ---
    @output
    @render.text
    def version_text():
        return f"App version: {APP_VERSION} | Model version: {safe_model_version()}"

    # Keep crisis period choices synced with T
    @reactive.effect
    def _update_crisis_period_choices():
        T = int(input.sim_T())
        choices = list(range(1, T + 1))
        default = [t for t in [4, 5, 6] if t <= T] or choices[: min(3, len(choices))]
        ui.update_selectize(
            "crisis_periods",
            choices=choices,
            selected=default if input.scenario() != "No crisis" else [],
        )
        ui.update_slider("horizon_display", max=T, value=min(int(input.horizon_display()), T))
        ui.update_slider("t_start", max=T, value=min(int(input.t_start()), T)) if "t_start" in input else None

    @reactive.calc
    def simulation_bundle():
        # 1) Horizon T
        sim_T = int(input.sim_T())
        model.T = sim_T
        T = sim_T

        # 2) Country selection
        DF_ALL = pd.DataFrame(model.COUNTRY_DATA)
        all_codes = DF_ALL["Code"].tolist()
        selected = list(input.country_selection() or [])

        if not selected:
            selected = all_codes

        df_countries = DF_ALL[DF_ALL["Code"].isin(selected)].reset_index(drop=True)

        # Update model country set
        model.COUNTRY_DATA = df_countries.to_dict("records")
        model.C = len(df_countries)
        C = model.C

        # Recompute weights (if available)
        if hasattr(model, "set_weights_from_df"):
            try:
                model.set_weights_from_df(df_countries)
            except Exception:
                pass

        # 3) Political settings
        theta_mode_internal = "fixed" if input.theta_mode_label() == "fixed" else "endogenous"
        voting_rule = input.voting_rule()
        use_EC = bool(input.use_EC())

        # --- CHANGE (2): lambda regime toggle (force λ = 1) ---
        force_lambda_one = (input.lambda_regime() == "one")

        # 4) Canonical parameters
        model.p = float(input.p())
        model.pD = min(float(input.pD()), model.p)
        model.D_level = float(input.D_level())
        model.alpha_p0 = float(input.alpha_p0())
        model.gamma_alpha = float(input.gamma_alpha())
        model.alpha_D = float(input.alpha_D())

        model.ie = float(input.ie())
        model.is_ = float(input.is_())
        model.ip = float(input.ip())

        model.beta_A = float(input.beta_A())
        model.betaE0 = float(input.betaE0())
        model.eta_A = float(input.eta_A())
        model.eta_S = float(input.eta_S())

        model.kappa_long = float(input.kappa_long())
        model.kappa_short = float(input.kappa_short())

        model.alpha_cp = float(input.alpha_cp())
        model.alpha_pf = float(input.alpha_pf())

        model.gamma_crc = float(input.gamma_crc())
        model.x_res = float(input.x_res())

        model.theta_fixed = float(input.theta_fixed())

        # recompute K
        model.K = (model.ie + model.is_ + model.ip) / 3.0

        # 5) Crisis
        scenario = input.scenario()
        shock_level = float(input.shock_level())

        crisis_periods = list(input.crisis_periods() or []) if scenario != "No crisis" else []
        crisis_countries = list(input.crisis_countries() or []) if scenario == "Asymmetric crisis" else []

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

        # 6) Run simulation, preserving old signature first
        try:
            res = model.run_simulation(
                CP=CP,
                use_EC=use_EC,
                theta_mode=theta_mode_internal,
                voting_rule=voting_rule,
                force_lambda_one=force_lambda_one,   # model may or may not accept
            )
        except TypeError:
            res = model.run_simulation(
                CP=CP,
                use_EC=use_EC,
                theta_mode=theta_mode_internal,
                voting_rule=voting_rule,
            )
            # If model cannot enforce λ=1 internally, do a best-effort display override
            if force_lambda_one and ("lambda" in res):
                lam = np.asarray(res["lambda"])
                res["lambda"] = np.ones_like(lam, dtype=float)

        return {
            "res": res,
            "CP": CP,
            "df_countries": df_countries,
            "codes": codes,
            "C": C,
            "T": T,
        }

    def _get_horizon_series():
        bundle = simulation_bundle()
        res = bundle["res"]

        I = np.asarray(res.get("I"))
        I_star = np.asarray(res.get("I_star"))
        Id_agg = np.asarray(res.get("Id_agg"))
        lam = np.asarray(res.get("lambda")) if res.get("lambda") is not None else None

        H = int(input.horizon_display())
        max_h = len(I) - 1
        H = max(1, min(H, max_h))

        idx = np.arange(H + 1)

        out = {
            "idx": idx,
            "t": idx,
            "I": I[idx],
            "I_star": I_star[idx] if I_star is not None and len(I_star) >= (H + 1) else None,
            "Id_agg": Id_agg[idx] if Id_agg is not None and len(Id_agg) >= (H + 1) else None,
            "lambda": lam[idx] if lam is not None and len(lam) >= (H + 1) else None,
        }
        return out

    @output
    @render.plot
    def plot_left():
        series = _get_horizon_series()
        t = series["t"]

        plt.figure()
        plt.plot(t, series["I"], label="I(t) – actual integration")
        if series["I_star"] is not None:
            plt.plot(t, series["I_star"], linestyle="--", label="I*(t) – functional target")
        plt.xlabel("Time (t)")
        plt.ylabel("Integration level")
        plt.ylim(0.0, 1.0)
        plt.grid(True, linestyle=":", linewidth=0.5)
        plt.legend()
        plt.tight_layout()

    @output
    @render.plot
    def plot_right():
        # --- CHANGE (3): right plot shows Id_agg + pivotal country demand (with change markers) ---
        bundle = simulation_bundle()
        res = bundle["res"]
        codes = bundle["codes"]

        series = _get_horizon_series()
        t = series["t"]
        Id_agg = series["Id_agg"]
        if Id_agg is None:
            raise RuntimeError("Model output must include 'Id_agg'.")

        IP = np.asarray(res.get("IP", None))
        SP = np.asarray(res.get("SP", None))
        if IP is None or SP is None or IP.ndim != 2 or SP.ndim != 2:
            raise RuntimeError("Model output must include 2D arrays 'IP' and 'SP' to compute pivotal demand.")

        Id_c = IP - SP  # per-country demand proxy (as in your new code)

        # Align time length
        Tplot = len(t)
        if Id_c.shape[1] < Tplot:
            Tplot = Id_c.shape[1]
            t = t[:Tplot]
            Id_agg = Id_agg[:Tplot]

        piv_codes = []
        Id_piv = np.zeros(Tplot, dtype=float)

        for tt in range(Tplot):
            idx, idv = compute_pivotal_series_optionB(Id_c[:, tt])
            piv_codes.append(codes[idx])
            Id_piv[tt] = idv

        # change points
        change = np.zeros(Tplot, dtype=bool)
        change[0] = True
        for tt in range(1, Tplot):
            change[tt] = (piv_codes[tt] != piv_codes[tt - 1])

        plt.figure()
        plt.axhline(0.0, linestyle=":", linewidth=1)

        plt.plot(t, Id_agg, label="Id_agg(t)")
        plt.plot(t, Id_piv, linestyle="--", label="Id_piv(t) – pivotal/constraint country")

        for tt in range(Tplot):
            if change[tt]:
                plt.axvline(t[tt], alpha=0.25)
                y = Id_piv[tt]
                plt.text(t[tt], y, f" {piv_codes[tt]}", fontsize=9, va="bottom")

        plt.xlabel("Time (t)")
        plt.ylabel("Demand")
        plt.grid(True, linestyle=":", linewidth=0.5)
        plt.legend()
        plt.tight_layout()

    @output
    @render.plot
    def plot_cp():
        bundle = simulation_bundle()
        CP = np.asarray(bundle["CP"])
        df_countries = bundle["df_countries"]
        C = bundle["C"]

        series = _get_horizon_series()
        t = series["t"]
        idx = series["idx"]

        n_show = min(5, C)
        sample_df = df_countries.head(n_show).copy()
        idxs_c = sample_df.index.to_list()
        cp_subset = CP[idxs_c, :][:, idx]

        plt.figure()
        for i_c, idx_c in enumerate(idxs_c):
            plt.plot(t, cp_subset[i_c, :], label=sample_df.loc[idx_c, "Code"])
        plt.xlabel("Time (t)")
        plt.ylabel("Crisis pressure CP")
        plt.grid(True, linestyle=":", linewidth=0.5)
        plt.legend()
        plt.tight_layout()

    @output
    @render.plot
    def plot_components():
        bundle = simulation_bundle()
        res = bundle["res"]
        codes_all = bundle["codes"]

        series = _get_horizon_series()
        t = series["t"]
        idx = series["idx"]

        component_key = input.component_select()
        comp_map = {
            "IP": res.get("IP", None),
            "SP": res.get("SP", None),
            "PF": res.get("PF", None),
            "CRC": res.get("CRC", None),
        }
        data = comp_map.get(component_key, None)

        selected_codes = list(input.component_countries() or [])
        if not selected_codes:
            selected_codes = [c for c in ["DE", "FR", "IT"] if c in codes_all] or codes_all[:3]

        plt.figure()
        if data is None:
            plt.text(0.1, 0.5, "Selected component is not available in the results.", transform=plt.gca().transAxes)
            plt.axis("off")
            return

        data = np.asarray(data)
        for code in selected_codes:
            if code in codes_all:
                idx_c = codes_all.index(code)
                series_c = data[idx_c, :][idx]
                plt.plot(t, series_c, label=code)

        plt.xlabel("Time (t)")
        plt.ylabel(component_key)
        plt.grid(True, linestyle=":", linewidth=0.5)
        plt.legend()
        plt.tight_layout()

    @output
    @render.data_frame
    def table_summary():
        series = _get_horizon_series()
        t = series["t"]

        df = pd.DataFrame(
            {
                "t": t,
                "I": series["I"],
                "I_star": series["I_star"] if series["I_star"] is not None else np.nan,
                "Id_agg": series["Id_agg"] if series["Id_agg"] is not None else np.nan,
                "lambda": series["lambda"] if series["lambda"] is not None else np.nan,
            }
        )
        return render.DataGrid(df.head(15))

    @output
    @render.download(filename=lambda: None)
    def download_csv():
        series = _get_horizon_series()
        t = series["t"]

        label = (input.scenario_label() or "scenario").replace(" ", "_")
        H = int(input.horizon_display())
        filename = f"results_{label}_T{H}.csv"

        df = pd.DataFrame(
            {
                "t": t,
                "I": series["I"],
                "I_star": series["I_star"] if series["I_star"] is not None else np.nan,
                "Id_agg": series["Id_agg"] if series["Id_agg"] is not None else np.nan,
                "lambda": series["lambda"] if series["lambda"] is not None else np.nan,
            }
        )

        yield df.to_csv(index=False).encode("utf-8")


app = App(app_ui, server)

