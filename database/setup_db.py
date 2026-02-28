"""
Generate synthetic mock data for local development.
Produces an SQLite database that mirrors the Databricks Marketplace
Predictive Maintenance & Asset Management schema (Dataknobs).

Run once before launching the app locally:
    python database/setup_db.py
"""
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DB_PATH = Path(__file__).parent / "warehouse.db"
RNG = np.random.default_rng(42)


# ── CNC Machine Data (AI4I 2020) ──────────────────────────────────────────────
def _make_cnc(n: int = 10_000) -> pd.DataFrame:
    type_choices = RNG.choice(["H", "M", "L"], size=n, p=[0.60, 0.30, 0.10])
    air_temp     = RNG.normal(300, 2, n).clip(295, 305)
    proc_temp    = air_temp + RNG.uniform(9.0, 11.0, n)

    # RPM depends on machine type
    rpm_map  = {"H": (1200, 2000), "M": (1200, 2500), "L": (1500, 2860)}
    rpm      = np.array([RNG.integers(*rpm_map[t]) for t in type_choices])

    # Torque inversely correlated with RPM
    torque   = (RNG.normal(0, 1, n) * 4 + 40 - (rpm - 1800) * 0.015).clip(3.8, 76.6)
    tool_wear = RNG.integers(0, 254, n)

    power    = torque * rpm * (2 * np.pi / 60)

    # Failure modes
    twf = ((tool_wear > 200) & (torque > 45)).astype(int)
    hdf = ((proc_temp - air_temp) < 8.6).astype(int)
    pwf = ((power < 3500) | (power > 9000)).astype(int)
    osf = ((tool_wear > 200) & (torque > 50)).astype(int)
    rnf = RNG.choice([0, 1], size=n, p=[0.999, 0.001])
    machine_failure = np.clip(twf + hdf + pwf + osf + rnf, 0, 1)

    product_ids = [f"{'HML'[['H','M','L'].index(t)]}{RNG.integers(10000, 99999)}"
                   for t in type_choices]

    df = pd.DataFrame({
        "UDI":                      range(1, n + 1),
        "Product ID":               product_ids,
        "Type":                     type_choices,
        "Air temperature [K]":      air_temp.round(1),
        "Process temperature [K]":  proc_temp.round(1),
        "Rotational speed [rpm]":   rpm,
        "Torque [Nm]":              torque.round(2),
        "Tool wear [min]":          tool_wear,
        "Machine failure":          machine_failure,
        "TWF":                      twf,
        "HDF":                      hdf,
        "PWF":                      pwf,
        "OSF":                      osf,
        "RNF":                      rnf,
    })
    return df


# ── NASA Turbofan Engine RUL ──────────────────────────────────────────────────
def _make_engine_rul(n_engines: int = 100) -> pd.DataFrame:
    rows = []
    for eng_id in range(1, n_engines + 1):
        max_cycles = RNG.integers(80, 350)
        for cycle in range(1, max_cycles + 1):
            rul = max_cycles - cycle
            degradation = cycle / max_cycles
            rows.append({
                "id":    eng_id,
                "Cycle": cycle,
                "OpSet1": RNG.uniform(-0.0009, 0.0009),
                "OpSet2": RNG.uniform(-0.0006, 0.0006),
                "OpSet3": int(RNG.choice([0, 20, 25])),
                "SensorMeasure1":  round(float(RNG.uniform(518, 519)), 2),
                "SensorMeasure2":  round(float(642 + degradation * 8 + RNG.normal(0, 0.5)), 2),
                "SensorMeasure3":  round(float(1580 + degradation * 20 + RNG.normal(0, 2)), 2),
                "SensorMeasure4":  round(float(1400 + degradation * 15 + RNG.normal(0, 3)), 2),
                "SensorMeasure5":  round(float(14.62 + RNG.normal(0, 0.02)), 2),
                "SensorMeasure6":  round(float(21.61 + RNG.normal(0, 0.1)), 2),
                "SensorMeasure7":  round(float(554 - degradation * 5 + RNG.normal(0, 1)), 2),
                "SensorMeasure8":  round(float(2388 + RNG.normal(0, 5)), 2),
                "SensorMeasure9":  round(float(9046 + RNG.normal(0, 20)), 2),
                "SensorMeasure10": round(float(1.3 + RNG.normal(0, 0.02)), 2),
                "SensorMeasure11": round(float(47.2 + degradation * 0.5 + RNG.normal(0, 0.5)), 2),
                "SensorMeasure12": round(float(521.4 + RNG.normal(0, 1)), 2),
                "SensorMeasure13": round(float(2388 + RNG.normal(0, 5)), 2),
                "SensorMeasure14": round(float(8138 + RNG.normal(0, 10)), 2),
                "SensorMeasure15": round(float(8.42 + RNG.normal(0, 0.05)), 2),
                "SensorMeasure16": round(float(0.03 + RNG.normal(0, 0.001)), 3),
                "SensorMeasure17": int(391 + RNG.integers(-2, 3)),
                "SensorMeasure18": int(2388),
                "SensorMeasure19": round(float(100.0 + RNG.normal(0, 0.2)), 2),
                "SensorMeasure20": round(float(38.86 + RNG.normal(0, 0.2)), 2),
                "SensorMeasure21": round(float(23.42 + RNG.normal(0, 0.1)), 2),
                "RemainingUsefulLife": rul,
            })
    return pd.DataFrame(rows)


# ── Electrical Fault ──────────────────────────────────────────────────────────
def _make_electrical(n: int = 2000) -> pd.DataFrame:
    # Fault flags: 0000=normal, else some fault
    fault_type = RNG.choice(
        ["normal", "LG", "LL", "LLG", "LLL"],
        size=n, p=[0.65, 0.15, 0.10, 0.07, 0.03]
    )
    g = np.zeros(n, int); c = np.zeros(n, int)
    b = np.zeros(n, int); a = np.zeros(n, int)

    for i, ft in enumerate(fault_type):
        if ft == "LG":   g[i] = 1; a[i] = 1
        elif ft == "LL":  b[i] = 1; c[i] = 1
        elif ft == "LLG": g[i] = 1; b[i] = 1; c[i] = 1
        elif ft == "LLL": a[i] = b[i] = c[i] = 1

    ia = RNG.normal(5.0, 0.2, n) + a * RNG.uniform(5, 20, n)
    ib = RNG.normal(5.0, 0.2, n) + b * RNG.uniform(5, 20, n)
    ic = RNG.normal(5.0, 0.2, n) + c * RNG.uniform(5, 20, n)
    va = RNG.normal(230, 2, n)   - a * RNG.uniform(10, 80, n)
    vb = RNG.normal(230, 2, n)   - b * RNG.uniform(10, 80, n)
    vc = RNG.normal(230, 2, n)   - c * RNG.uniform(10, 80, n)

    return pd.DataFrame({
        "G": g, "C": c, "B": b, "A": a,
        "Ia": ia.round(4), "Ib": ib.round(4), "Ic": ic.round(4),
        "Va": va.round(4), "Vb": vb.round(4), "Vc": vc.round(4),
    })


# ── Transformer Readings ──────────────────────────────────────────────────────
def _make_transformer(n: int = 500) -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=n, freq="1h")
    t = np.linspace(0, 2 * np.pi, n)

    # Introduce a gradual degradation trend + random spikes
    oti = 70 + 5 * np.sin(t) + RNG.normal(0, 1, n) + np.linspace(0, 10, n) * 0.3
    wti = oti + 5 + RNG.normal(0, 0.5, n)
    ati = 25 + RNG.normal(0, 2, n)
    oli = 95 - np.linspace(0, 5, n) + RNG.normal(0, 1, n)

    vl1 = 11_000 + RNG.normal(0, 50, n)
    vl2 = 11_000 + RNG.normal(0, 50, n)
    vl3 = 11_000 + RNG.normal(0, 50, n)
    il1 = 100 + 20 * np.sin(t) + RNG.normal(0, 5, n)
    il2 = 100 + 20 * np.sin(t + 2.09) + RNG.normal(0, 5, n)
    il3 = 100 + 20 * np.sin(t + 4.19) + RNG.normal(0, 5, n)
    inut = RNG.normal(0.5, 0.3, n).clip(0)
    # Add a few anomaly spikes
    spike_idx = RNG.choice(n, size=5, replace=False)
    inut[spike_idx] += RNG.uniform(5, 15, 5)

    return pd.DataFrame({
        "DeviceTimeStamp": timestamps.astype(str),
        "OTI":  oti.round(2),  "WTI": wti.round(2),
        "ATI":  ati.round(2),  "OLI": oli.round(2),
        "OTI_A": oti.round(2), "OTI_T": wti.round(2),
        "VL1":  vl1.round(1),  "VL2":  vl2.round(1),  "VL3":  vl3.round(1),
        "IL1":  il1.round(2),  "IL2":  il2.round(2),  "IL3":  il3.round(2),
        "VL12": (vl1 - vl2).round(1), "VL23": (vl2 - vl3).round(1),
        "VL31": (vl3 - vl1).round(1),
        "INUT": inut.round(4),
        "MOG_A": RNG.choice([0, 1], size=n, p=[0.97, 0.03]),
    })


# ── Heater / Battery Cycles ───────────────────────────────────────────────────
def _make_heater(n_cycles: int = 50) -> pd.DataFrame:
    rows = []
    for cycle_id in range(1, n_cycles + 1):
        n_steps = RNG.integers(100, 300)
        degradation = cycle_id / n_cycles
        phid = f"B{cycle_id:04d}"
        for step in range(n_steps):
            t = step * 0.1
            rows.append({
                "Voltage_measured":    round(4.2 - degradation * 0.3 - step * 0.002 + float(RNG.normal(0, 0.01)), 3),
                "Current_measured":    round(float(RNG.normal(1.5, 0.1)), 3),
                "Temperature_measured": round(25 + step * 0.05 + float(RNG.normal(0, 0.5)), 2),
                "Current_charge":      round(float(RNG.normal(1.5, 0.05)), 3),
                "Voltage_charge":      round(float(RNG.uniform(4.15, 4.25)), 3),
                "Time":                t,
                "Capacity":            round(1.86 - degradation * 0.3 + float(RNG.normal(0, 0.02)), 3),
                "id_cycle":            cycle_id,
                "type":                RNG.choice(["discharge", "charge"], p=[0.6, 0.4]),
                "ambient_temperature": 25,
                "time_year":           2024,
                "PhID":                phid,
            })
    return pd.DataFrame(rows)


# ── Database setup ────────────────────────────────────────────────────────────
def _create_and_insert(conn: sqlite3.Connection, table: str, df: pd.DataFrame) -> None:
    df.to_sql(table, conn, if_exists="replace", index=False)
    print(f"  ✓  {table:30s} {len(df):>7,} rows")


def setup(quiet: bool = False) -> None:
    DB_PATH.parent.mkdir(exist_ok=True)
    print(f"Building mock database at {DB_PATH} …")

    cnc_df   = _make_cnc(10_000)
    eng_df   = _make_engine_rul(100)
    elec_df  = _make_electrical(2_000)
    trans_df = _make_transformer(500)
    heat_df  = _make_heater(50)

    conn = sqlite3.connect(DB_PATH)
    try:
        _create_and_insert(conn, "cnc_machine",         cnc_df)
        _create_and_insert(conn, "engine_rul",          eng_df)
        _create_and_insert(conn, "electrical_fault",    elec_df)
        _create_and_insert(conn, "transformer_reading", trans_df)
        _create_and_insert(conn, "heater_cycle",        heat_df)
        conn.commit()
    finally:
        conn.close()

    print(f"\nDone! Run `python app.py` to start the app.")


if __name__ == "__main__":
    setup()
