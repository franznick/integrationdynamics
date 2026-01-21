import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

C = 27
T = 30

COUNTRY_DATA = [
    {"Code": "DE", "Country": "Germany", "GDP_share_pct": 24.0, "Council_votes": 29, "Debt_to_GDP_pct": 62.3},
    {"Code": "FR", "Country": "France", "GDP_share_pct": 16.27, "Council_votes": 29, "Debt_to_GDP_pct": 114.7},
    {"Code": "IT", "Country": "Italy", "GDP_share_pct": 12.22, "Council_votes": 29, "Debt_to_GDP_pct": 137.9},
    {"Code": "ES", "Country": "Spain", "GDP_share_pct": 8.87, "Council_votes": 29, "Debt_to_GDP_pct": 102.8},
    {"Code": "NL", "Country": "Netherlands", "GDP_share_pct": 6.32, "Council_votes": 14, "Debt_to_GDP_pct": 43.2},
    {"Code": "PL", "Country": "Poland", "GDP_share_pct": 4.71, "Council_votes": 27, "Debt_to_GDP_pct": 57.4},
    {"Code": "BE", "Country": "Belgium", "GDP_share_pct": 3.42, "Council_votes": 12, "Debt_to_GDP_pct": 106.8},
    {"Code": "SE", "Country": "Sweden", "GDP_share_pct": 3.12, "Council_votes": 12, "Debt_to_GDP_pct": 33.9},
    {"Code": "IE", "Country": "Ireland", "GDP_share_pct": 2.97, "Council_votes": 7, "Debt_to_GDP_pct": 33.1},
    {"Code": "AT", "Country": "Austria", "GDP_share_pct": 2.7, "Council_votes": 10, "Debt_to_GDP_pct": 84.9},
    {"Code": "DK", "Country": "Denmark", "GDP_share_pct": 2.21, "Council_votes": 10, "Debt_to_GDP_pct": 29.9},
    {"Code": "RO", "Country": "Romania", "GDP_share_pct": 1.97, "Council_votes": 27, "Debt_to_GDP_pct": 48.8},
    {"Code": "CZ", "Country": "Czechia", "GDP_share_pct": 1.78, "Council_votes": 12, "Debt_to_GDP_pct": 38.9},
    {"Code": "PT", "Country": "Portugal", "GDP_share_pct": 1.59, "Council_votes": 12, "Debt_to_GDP_pct": 94.0},
    {"Code": "FI", "Country": "Finland", "GDP_share_pct": 1.54, "Council_votes": 7, "Debt_to_GDP_pct": 82.1},
    {"Code": "EL", "Country": "Greece", "GDP_share_pct": 1.32, "Council_votes": 13, "Debt_to_GDP_pct": 152.5},
    {"Code": "HU", "Country": "Hungary", "GDP_share_pct": 1.15, "Council_votes": 12, "Debt_to_GDP_pct": 75.3},
    {"Code": "SK", "Country": "Slovakia", "GDP_share_pct": 0.73, "Council_votes": 7, "Debt_to_GDP_pct": 53.9},
    {"Code": "BG", "Country": "Bulgaria", "GDP_share_pct": 0.58, "Council_votes": 10, "Debt_to_GDP_pct": 23.9},
    {"Code": "HR", "Country": "Croatia", "GDP_share_pct": 0.48, "Council_votes": 3, "Debt_to_GDP_pct": 54.0},
    {"Code": "LU", "Country": "Luxembourg", "GDP_share_pct": 0.48, "Council_votes": 4, "Debt_to_GDP_pct": 26.3},
    {"Code": "LT", "Country": "Lithuania", "GDP_share_pct": 0.44, "Council_votes": 7, "Debt_to_GDP_pct": 39.2},
    {"Code": "SI", "Country": "Slovenia", "GDP_share_pct": 0.37, "Council_votes": 4, "Debt_to_GDP_pct": 69.9},
    {"Code": "EE", "Country": "Estonia", "GDP_share_pct": 0.22, "Council_votes": 4, "Debt_to_GDP_pct": 20.7},
    {"Code": "LV", "Country": "Latvia", "GDP_share_pct": 0.22, "Council_votes": 7, "Debt_to_GDP_pct": 45.6},
    {"Code": "CY", "Country": "Cyprus", "GDP_share_pct": 0.19, "Council_votes": 4, "Debt_to_GDP_pct": 64.3},
    {"Code": "MT", "Country": "Malta", "GDP_share_pct": 0.13, "Council_votes": 4, "Debt_to_GDP_pct": 48.1},
]

p = 0.3
pD = 0.15
D_level = 0.3
if pD > p:
    pD = p
ie = 0.1
is_ = 0.1
ip = 0.1
K = (ie + is_ + ip) / 3.0
alpha_cp = 1.0
alpha_pf = 0.5
alpha_p0 = 0.2
gamma_alpha = 0.05
alpha_D = 0.6
beta_A = 0.5
betaE0 = 0.5
kappa_long = 1.0
kappa_short = 1.0
eta_A = 0.1
eta_S = 0.05
theta_fixed = 0.8
gamma_crc = 1.0
x_res = 0.1
omega_pol = None
omega_econ = None
omega_f = None
w = None

def set_weights_from_df(df):
    global omega_pol, omega_econ, omega_f, w
    gdp_shares = df["GDP_share_pct"].values.astype(float) / 100.0
    council_votes = df["Council_votes"].values.astype(float)
    debt_to_gdp = df["Debt_to_GDP_pct"].values.astype(float)
    omega_econ_local = gdp_shares / gdp_shares.sum()
    omega_pol_local = council_votes / council_votes.sum()
    inv_debt = 1.0 / debt_to_gdp
    omega_f_raw = inv_debt * omega_econ_local
    omega_f_local = omega_f_raw / omega_f_raw.sum()
    omega_econ = omega_econ_local
    omega_pol = omega_pol_local
    omega_f = omega_f_local
    w_local = np.zeros((C, C))
    for c in range(C):
        for k in range(C):
            if c != k:
                w_local[c, k] = 1.0 / (C - 1)
    w = w_local
    return omega_econ, omega_pol, omega_f, w

def run_simulation(CP_ex, use_EC=True, theta_mode="fixed", voting_rule="majority", force_lambda_one=False):
    global omega_pol, omega_econ, omega_f, w
    if omega_pol is None or omega_econ is None or omega_f is None or w is None:
        raise RuntimeError("Weights not set.")
    I = np.zeros(T+1)
    I_star = np.zeros(T+1)
    EC = np.zeros(T+1)
    IP = np.zeros((C, T+1))
    CP_tot = np.zeros((C, T+1))  # endogenous total crisis pressure (stock)
    PF = np.zeros((C, T+1))
    SP = np.zeros((C, T+1))
    Id_c = np.zeros((C, T+1))
    Id_agg = np.zeros(T+1)
    E = np.zeros((C, T+1))
    A = np.zeros((C, T+1))
    betaE_ct = np.zeros((C, T+1))
    lambda_t = np.zeros(T+1)
    IP_bar = np.zeros(T+1)
    CRC = np.zeros((C, T+1))
    alpha_p_ct = np.zeros((C, T+1))
    I[0] = 0.3
    I_star[0] = 0.3
    E[:, 0] = 0.5
    alpha_p_ct[:, 0] = alpha_p0
    R = 0.0 if voting_rule == "majority" else 1.0
    for t in range(T):
        if theta_mode == "fixed":
            theta_c = np.full(C, theta_fixed)
        else:
            theta_c = np.clip(1.0 - CP_ex[:, t], 0.0, 1.0)
        A[:, t] = theta_c * I[t]
        S_t = 1.0 - np.std(CP_ex[:, t])
        for c in range(C):
            E[c, t+1] = E[c, t] + eta_A * A[c, t] + eta_S * S_t
        E[:, t+1] = np.clip(E[:, t+1], 0.0, 1.0)
        EC[t] = max(0.0, I_star[t] - I[t]) if use_EC else 0.0
        factor = max(0.0, 1.0 - x_res * I[t])
        CP_eff_t = CP_ex[:, t] * factor
        for c in range(C):
            PF[c, t] = ie * np.sum(w[c, :] * CP_eff_t)
        # Crisis stock carryover is reduced by change in integration (ΔI), not by the level I.
        if t == 0:
            carry = np.zeros(C)
        else:
            delta_I_prev = 0.0 if t == 1 else (I[t-1] - I[t-2])
            carry = np.maximum(0.0, CP_tot[:, t-1] - delta_I_prev)
        # Total crisis pressure (stock) adds current attenuated flow plus propagated flow
        CP_tot[:, t] = (CP_eff_t + alpha_pf * PF[:, t]) + carry
        interdep_avg = (is_ + ip) / 2.0
        for c in range(C):
            CRC[c, t] = (1.0 - I[t]) * (1.0 - interdep_avg) * omega_f[c]
        for c in range(C):
            IP[c, t] = (alpha_cp * CP_tot[c, t] + ip * EC[t] - gamma_crc * CRC[c, t])
        IP_bar[t] = np.dot(omega_pol, IP[:, t])
        delta_I = 0.0 if t == 0 else I[t] - I[t-1]
        p_t = p * (1.0 + delta_I)
        for c in range(C):
            long_term = I[t] - I_star[t] if use_EC else 0.0
            short_term = abs(delta_I - IP[c, t])
            base = betaE0 + kappa_long * long_term + kappa_short * short_term
            betaE_ct[c, t] = base * p_t
        for c in range(C):
            val_SP = (alpha_p_ct[c, t] * p_t - alpha_D * pD * D_level + betaE_ct[c, t] * (1.0 - E[c, t]) - beta_A * A[c, t])
            SP[c, t] = max(0.0, val_SP)
            Id_c[c, t] = IP[c, t] - SP[c, t]
        if voting_rule == "majority":
            Id_agg[t] = np.dot(omega_pol, Id_c[:, t])
        else:
            # Option B: "two-sided unanimity" with sign-consensus.
            # - If all countries' net demand Id_c(t) is non-negative, integrate at the pace of the weakest supporter.
            # - If all countries' net demand Id_c(t) is non-positive, disintegrate at the pace of the weakest opponent (closest to zero).
            # - Otherwise, status quo (no change) under unanimity.
            Id_t = Id_c[:, t]
            if np.all(Id_t >= 0):
                c_star = int(np.argmin(Id_t))
                Id_agg[t] = float(Id_t[c_star])
            elif np.all(Id_t <= 0):
                c_star = int(np.argmax(Id_t))  # least negative (closest to zero)
                Id_agg[t] = float(Id_t[c_star])
            else:
                Id_agg[t] = 0.0
        sd_CP = np.std(CP_tot[:, t])
        lambda_raw = ((1.0 + (1.0 - R)) / C) - sd_CP
        lambda_t[t] = max(0.0, min(1.0, lambda_raw))
        if use_EC:
            I_star_next = I_star[t] + K * I[t] * (1.0 + IP_bar[t])
            I_star[t+1] = np.clip(I_star_next, 0.0, 1.0)
        else:
            I_star[t+1] = I_star[t]
        I_next = I[t] + (1.0 if force_lambda_one else lambda_t[t]) * Id_agg[t]
        I[t+1] = np.clip(I_next, 0.0, 1.0)
        for c in range(C):
            alpha_next = alpha_p_ct[c, t] + gamma_alpha * betaE_ct[c, t] * (1.0 - alpha_p_ct[c, t])
            alpha_p_ct[c, t+1] = np.clip(alpha_next, 0.0, 1.0)
    t = T
    EC[t] = max(0.0, I_star[t] - I[t]) if use_EC else 0.0
    return {"I": I, "I_star": I_star, "EC": EC, "Id_agg": Id_agg, "lambda": lambda_t, "CP_ex": CP_ex, "CP_tot": CP_tot, "alpha_p": alpha_p_ct, "betaE": betaE_ct, "CRC": CRC, "IP": IP, "SP": SP, "PF": PF}

def build_CP_ex_symmetric_three_windows(shock_level=0.15):
    CP_ex = np.zeros((C, T+1))
    windows = [(4,7),(14,17),(24,27)]
    for start, end in windows:
        CP_ex[:, start:end] = shock_level
    return CP_ex

def build_CP_asymmetric_three_windows(shock_level=0.15, crisis_codes=("IT","ES","EL")):
    CP = np.zeros((C, T+1))
    windows = [(4,7),(14,17),(24,27)]
    df = pd.DataFrame(COUNTRY_DATA)
    code_to_idx = {code: i for i, code in enumerate(df["Code"].tolist())}
    crisis_idx = [code_to_idx[c] for c in crisis_codes]
    for start, end in windows:
        CP[crisis_idx, start:end] = shock_level
    return CP_ex
