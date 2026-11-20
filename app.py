"""
IoT Based Predictive Maintenance and Wind Protection System for Wind Turbines
Software simulation built with Python + Streamlit (no physical hardware).

Run:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------- settings
CUT_IN_WIND = 3.0      # m/s: turbine starts producing power
RATED_WIND = 12.0      # m/s: rated power reached
WARNING_WIND = 20.0    # m/s: reduce load (pitch blades)
CUT_OUT_WIND = 25.0    # m/s: emergency stop (wind protection)
RATED_POWER_KW = 2000

st.set_page_config(page_title="Wind Turbine Monitor", layout="wide")
st.title("IoT Predictive Maintenance & Wind Protection System")
st.caption("Simulated sensor data for a wind turbine (software simulation)")

# ---------------------------------------------------------------- sidebar
st.sidebar.header("Simulation controls")
hours = st.sidebar.slider("Hours simulated", 24, 336, 96, step=24)
seed = st.sidebar.number_input("Random seed", value=42, step=1)
storm = st.sidebar.checkbox("Inject storm event", value=True)
fault = st.sidebar.checkbox("Inject gearbox fault (degradation)", value=True)
z_limit = st.sidebar.slider("Anomaly threshold (z-score)", 2.0, 5.0, 3.0, 0.5)


# ---------------------------------------------------------------- simulation
@st.cache_data
def simulate(hours, seed, storm, fault):
    rng = np.random.default_rng(seed)
    t = pd.date_range("2026-01-01", periods=hours * 6, freq="10min")
    n = len(t)

    # wind speed: daily cycle + noise (+ optional storm)
    day = np.sin(np.linspace(0, 2 * np.pi * hours / 24, n))
    wind = 9 + 3 * day + rng.normal(0, 1.5, n)
    if storm:
        s = int(n * 0.7)
        wind[s:s + 18] += np.linspace(8, 20, 18)
    wind = np.clip(wind, 0, 35)

    # rotor speed and power from wind
    rpm = np.clip(wind / RATED_WIND, 0, 1) * 16 + rng.normal(0, 0.3, n)
    power = np.where(
        wind < CUT_IN_WIND, 0,
        np.minimum(RATED_POWER_KW, RATED_POWER_KW * ((wind - CUT_IN_WIND) / (RATED_WIND - CUT_IN_WIND)) ** 3),
    )

    # degradation ramp for gearbox fault in last 30% of data
    ramp = np.zeros(n)
    if fault:
        start = int(n * 0.6)
        ramp[start:] = np.linspace(0, 1, n - start)

    vibration = 1.2 + 0.05 * rpm + 2.5 * ramp + rng.normal(0, 0.15, n)       # mm/s
    gearbox_temp = 45 + 0.6 * rpm + 25 * ramp + rng.normal(0, 1.2, n)         # deg C
    oil_pressure = 4.5 - 1.2 * ramp + rng.normal(0, 0.1, n)                   # bar

    df = pd.DataFrame({
        "time": t, "wind_speed": wind, "rpm": rpm, "power_kw": power,
        "vibration": vibration, "gearbox_temp": gearbox_temp, "oil_pressure": oil_pressure,
    }).set_index("time")
    return df


df = simulate(hours, seed, storm, fault)

# ---------------------------------------------------------------- analytics
# 1. Anomaly detection: rolling z-score on key health sensors
sensors = ["vibration", "gearbox_temp", "oil_pressure"]
window = 36  # 6 hours
for s in sensors:
    mean = df[s].rolling(window, min_periods=12).mean().shift(1)
    std = df[s].rolling(window, min_periods=12).std().shift(1)
    df[f"{s}_z"] = ((df[s] - mean) / std).abs()

df["anomaly"] = (df[[f"{s}_z" for s in sensors]] > z_limit).any(axis=1)

# 2. Health score (0-100): penalise high vibration, temperature, low oil pressure
vib_pen = np.clip((df["vibration"] - 2.0) / 3.0, 0, 1)
temp_pen = np.clip((df["gearbox_temp"] - 60) / 30, 0, 1)
oil_pen = np.clip((4.0 - df["oil_pressure"]) / 2.0, 0, 1)
df["health"] = 100 * (1 - (0.4 * vib_pen + 0.35 * temp_pen + 0.25 * oil_pen))

# 3. Wind protection logic
def protection_state(w):
    if w >= CUT_OUT_WIND:
        return "EMERGENCY STOP"
    if w >= WARNING_WIND:
        return "PITCH / DERATE"
    if w < CUT_IN_WIND:
        return "IDLE"
    return "NORMAL"


df["protection"] = df["wind_speed"].apply(protection_state)
df.loc[df["protection"] == "EMERGENCY STOP", ["power_kw", "rpm"]] = 0

# 4. Maintenance recommendation
latest = df.iloc[-1]
recent_anoms = int(df["anomaly"].tail(36).sum())
if latest["health"] < 50 or recent_anoms > 10:
    advice, level = "Schedule maintenance immediately: gearbox degradation likely.", "error"
elif latest["health"] < 75 or recent_anoms > 3:
    advice, level = "Inspect within the next maintenance window.", "warning"
else:
    advice, level = "Turbine healthy. No action needed.", "success"

# ---------------------------------------------------------------- dashboard
c1, c2, c3, c4 = st.columns(4)
c1.metric("Wind speed", f"{latest['wind_speed']:.1f} m/s")
c2.metric("Power output", f"{latest['power_kw']:.0f} kW")
c3.metric("Health score", f"{latest['health']:.0f} / 100")
c4.metric("Protection state", latest["protection"])

getattr(st, level)(f"Maintenance advice: {advice}")

tab1, tab2, tab3 = st.tabs(["Wind & power", "Condition monitoring", "Alerts"])

with tab1:
    st.subheader("Wind speed (m/s)")
    st.line_chart(df["wind_speed"])
    st.subheader("Power output (kW)")
    st.line_chart(df["power_kw"])
    st.write(
        f"Warning at {WARNING_WIND} m/s (derate), emergency stop at {CUT_OUT_WIND} m/s. "
        f"Time in protection mode: {(df['protection'].isin(['EMERGENCY STOP', 'PITCH / DERATE'])).mean() * 100:.1f}%"
    )

with tab2:
    st.subheader("Vibration (mm/s)")
    st.line_chart(df["vibration"])
    st.subheader("Gearbox temperature (deg C)")
    st.line_chart(df["gearbox_temp"])
    st.subheader("Oil pressure (bar)")
    st.line_chart(df["oil_pressure"])
    st.subheader("Health score")
    st.line_chart(df["health"])

with tab3:
    alerts = df[(df["anomaly"]) | (df["protection"].isin(["EMERGENCY STOP", "PITCH / DERATE"]))]
    st.write(f"{len(alerts)} alert records")
    st.dataframe(
        alerts[["wind_speed", "vibration", "gearbox_temp", "oil_pressure", "health", "protection"]].round(2),
        use_container_width=True,
    )
    st.download_button("Download full data (CSV)", df.to_csv().encode(), "turbine_data.csv", "text/csv")
