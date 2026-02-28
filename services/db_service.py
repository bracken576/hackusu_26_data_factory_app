"""
Data access layer — all SQL execution lives here.
Returns pd.DataFrames with standardized column names regardless of backend.

Backend detection:
  - DATABRICKS_WAREHOUSE_ID set → Databricks SQL Warehouse (production)
  - Not set → SQLite local mock (run database/setup_db.py first)
"""
import os
import time
import logging
import sqlite3
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ── Backend detection ──────────────────────────────────────────────────────────
_IS_DB = bool(os.getenv("DATABRICKS_WAREHOUSE_ID"))
_DB_PATH = Path(__file__).parent.parent / "database" / "warehouse.db"

_CATALOG = os.getenv("DATABRICKS_CATALOG", "main")
_SCHEMA  = os.getenv("DATABRICKS_SCHEMA", "predictive_maintenance")

# Table references — names match Dataknobs Marketplace dataset (installed via Unity Catalog)
# If your catalog/schema differ, update DATABRICKS_CATALOG and DATABRICKS_SCHEMA in app.yaml
CNC_TABLE         = f"{_CATALOG}.{_SCHEMA}.cnc_data_ai_4_i_2020"
ENGINE_TABLE      = f"{_CATALOG}.{_SCHEMA}.nasa_data_train_test"
ELEC_TABLE        = f"{_CATALOG}.{_SCHEMA}.eletrical_fault_train_test_data"  # note: typo is in dataset
TRANSFORMER_TABLE = f"{_CATALOG}.{_SCHEMA}.transformer_train_test_data"
HEATER_TABLE      = f"{_CATALOG}.{_SCHEMA}.heater_train_test_data"
AUDIT_TABLE       = f"{_CATALOG}.governance.audit_log"

_CNC_TBL  = CNC_TABLE         if _IS_DB else "cnc_machine"
_ENG_TBL  = ENGINE_TABLE      if _IS_DB else "engine_rul"
_ELC_TBL  = ELEC_TABLE        if _IS_DB else "electrical_fault"
_TRN_TBL  = TRANSFORMER_TABLE if _IS_DB else "transformer_reading"
_HTR_TBL  = HEATER_TABLE      if _IS_DB else "heater_cycle"

# Quote character for identifiers with spaces/brackets
_Q = "`" if _IS_DB else '"'


def _q(col: str) -> str:
    """Wrap column name with the appropriate quote char when needed."""
    if " " in col or "[" in col or "]" in col:
        return f"{_Q}{col}{_Q}"
    return col


# ── Low-level executor ─────────────────────────────────────────────────────────
def _sql_query(query: str) -> pd.DataFrame:
    """Execute SQL and return a DataFrame. Never call from UI layer directly."""
    t0 = time.monotonic()
    if _IS_DB:
        from databricks import sql as dbsql
        from databricks.sdk.core import Config
        cfg = Config()
        with dbsql.connect(
            server_hostname=cfg.host,
            http_path=f"/sql/1.0/warehouses/{os.getenv('DATABRICKS_WAREHOUSE_ID')}",
            credentials_provider=lambda: cfg.authenticate,
        ) as conn:
            with conn.cursor() as cursor:
                cursor.execute(query)
                df = cursor.fetchall_arrow().to_pandas()
    else:
        if not _DB_PATH.exists():
            raise FileNotFoundError(
                f"Mock database not found at {_DB_PATH}. "
                "Run: python database/setup_db.py"
            )
        conn = sqlite3.connect(_DB_PATH)
        try:
            df = pd.read_sql_query(query, conn)
        finally:
            conn.close()

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    logger.debug("Query took %d ms, returned %d rows", elapsed_ms, len(df))
    return df


# ── KPI summary (called at startup) ───────────────────────────────────────────
def get_summary_kpis() -> dict:
    """Return dict of all dashboard KPIs. Runs 3 fast aggregate queries."""
    kpis = {}
    try:
        cnc = _sql_query(f"""
            SELECT
                ROUND((1.0 - AVG(CAST({_q("Machine failure")} AS FLOAT))) * 100, 1) AS health_score,
                COUNT(DISTINCT {_q("Product ID")})  AS total_machines,
                ROUND(AVG(CAST({_q("Tool wear [min]")} AS FLOAT)), 0) AS avg_tool_wear,
                SUM({_q("Machine failure")}) AS total_failures,
                ROUND(AVG(CAST({_q("Machine failure")} AS FLOAT)) * 100, 2) AS failure_rate_pct
            FROM {_CNC_TBL}
        """)
        if len(cnc) > 0:
            kpis.update({
                "health_score":    round(float(cnc["health_score"].iloc[0] or 0), 1),
                "total_machines":  int(cnc["total_machines"].iloc[0] or 0),
                "avg_tool_wear":   int(cnc["avg_tool_wear"].iloc[0] or 0),
                "total_failures":  int(cnc["total_failures"].iloc[0] or 0),
                "failure_rate_pct": float(cnc["failure_rate_pct"].iloc[0] or 0),
            })
    except Exception as exc:
        logger.warning("CNC KPI query failed: %s", exc)
        kpis.update({"health_score": 0, "total_machines": 0, "avg_tool_wear": 0,
                     "total_failures": 0, "failure_rate_pct": 0})

    try:
        eng = _sql_query(f"""
            SELECT
                ROUND(AVG(CAST(RemainingUsefulLife AS FLOAT)), 0) AS avg_rul,
                SUM(CASE WHEN RemainingUsefulLife < 50 THEN 1 ELSE 0 END) AS critical_engines,
                COUNT(DISTINCT id) AS total_engines
            FROM {_ENG_TBL}
            WHERE Cycle = (SELECT MAX(Cycle) FROM {_ENG_TBL} e2 WHERE e2.id = {_ENG_TBL}.id)
        """)
        if len(eng) > 0:
            kpis.update({
                "avg_rul":        int(eng["avg_rul"].iloc[0] or 0),
                "critical_engines": int(eng["critical_engines"].iloc[0] or 0),
                "total_engines":  int(eng["total_engines"].iloc[0] or 0),
            })
    except Exception as exc:
        logger.warning("Engine KPI query failed: %s", exc)
        kpis.update({"avg_rul": 0, "critical_engines": 0, "total_engines": 0})

    try:
        elec = _sql_query(f"""
            SELECT
                ROUND(AVG(CASE WHEN (G + C + B + A) > 0 THEN 1.0 ELSE 0.0 END) * 100, 1)
                    AS elec_fault_rate
            FROM {_ELC_TBL}
        """)
        if len(elec) > 0:
            kpis["elec_fault_rate"] = float(elec["elec_fault_rate"].iloc[0] or 0)
    except Exception as exc:
        logger.warning("Electrical KPI query failed: %s", exc)
        kpis["elec_fault_rate"] = 0

    return kpis


# ── CNC Machine functions ──────────────────────────────────────────────────────
def get_cnc_failure_modes() -> pd.DataFrame:
    """Return failure mode counts (TWF, HDF, PWF, OSF, RNF)."""
    df = _sql_query(f"""
        SELECT
            SUM({_q("TWF")}) AS twf,
            SUM({_q("HDF")}) AS hdf,
            SUM({_q("PWF")}) AS pwf,
            SUM({_q("OSF")}) AS osf,
            SUM({_q("RNF")}) AS rnf
        FROM {_CNC_TBL}
    """)
    # Melt to long format for plotting
    row = df.iloc[0]
    return pd.DataFrame({
        "failure_mode": ["Tool Wear (TWF)", "Heat Dissipation (HDF)", "Power Failure (PWF)",
                         "Overstrain (OSF)", "Random (RNF)"],
        "count": [int(row["twf"] or 0), int(row["hdf"] or 0), int(row["pwf"] or 0),
                  int(row["osf"] or 0), int(row["rnf"] or 0)],
    })


def get_cnc_scatter_data(machine_type: str | None = None, limit: int = 3000) -> pd.DataFrame:
    """Return CNC scatter data with standardized column names."""
    where = f"WHERE {_q('Type')} = '{machine_type}'" if machine_type else ""
    df = _sql_query(f"""
        SELECT
            {_q("Rotational speed [rpm]")}  AS rpm,
            {_q("Torque [Nm]")}              AS torque_nm,
            {_q("Tool wear [min]")}          AS tool_wear_min,
            {_q("Air temperature [K]")}      AS air_temp_k,
            {_q("Process temperature [K]")}  AS proc_temp_k,
            {_q("Type")}                     AS machine_type,
            {_q("Machine failure")}          AS failure
        FROM {_CNC_TBL}
        {where}
        LIMIT {limit}
    """)
    df["failure_label"] = df["failure"].apply(lambda x: "Failure" if x == 1 else "Normal")
    return df


def get_cnc_failure_by_type() -> pd.DataFrame:
    """Return failure rate and avg tool wear grouped by machine type."""
    return _sql_query(f"""
        SELECT
            {_q("Type")} AS machine_type,
            COUNT(*) AS total_records,
            SUM({_q("Machine failure")}) AS failures,
            ROUND(AVG(CAST({_q("Machine failure")} AS FLOAT)) * 100, 2) AS failure_rate_pct,
            ROUND(AVG(CAST({_q("Tool wear [min]")} AS FLOAT)), 1) AS avg_tool_wear_min,
            ROUND(AVG(CAST({_q("Torque [Nm]")} AS FLOAT)), 2) AS avg_torque_nm
        FROM {_CNC_TBL}
        GROUP BY {_q("Type")}
        ORDER BY failure_rate_pct DESC
    """)


def get_cnc_anomalies(limit: int = 50) -> pd.DataFrame:
    """Return records with anomalous readings (high tool wear + high torque)."""
    return _sql_query(f"""
        SELECT
            {_q("UDI")}                      AS udi,
            {_q("Product ID")}               AS product_id,
            {_q("Type")}                     AS machine_type,
            {_q("Tool wear [min]")}          AS tool_wear_min,
            {_q("Torque [Nm]")}              AS torque_nm,
            {_q("Rotational speed [rpm]")}   AS rpm,
            {_q("Air temperature [K]")}      AS air_temp_k,
            {_q("Machine failure")}          AS failure
        FROM {_CNC_TBL}
        WHERE {_q("Tool wear [min]")} > 200
           OR {_q("Torque [Nm]")} > 65
        ORDER BY {_q("Tool wear [min]")} DESC
        LIMIT {limit}
    """)


# ── Engine / NASA RUL functions ────────────────────────────────────────────────
def get_engine_rul_buckets() -> pd.DataFrame:
    """Return engine counts bucketed by health status (latest cycle per engine)."""
    df = _sql_query(f"""
        SELECT
            id AS engine_id,
            MAX(Cycle) AS max_cycle,
            MIN(RemainingUsefulLife) AS final_rul
        FROM {_ENG_TBL}
        GROUP BY id
    """)
    df["bucket"] = pd.cut(
        df["final_rul"],
        bins=[-1, 49, 99, float("inf")],
        labels=["Critical (<50)", "Warning (50-99)", "Healthy (≥100)"],
    )
    return df.groupby("bucket", observed=True).size().reset_index(name="count")


def get_engine_rul_trend(engine_ids: list | None = None, limit_engines: int = 10) -> pd.DataFrame:
    """Return RUL over cycles for selected engines (or top N by lowest final RUL)."""
    if engine_ids:
        id_list = ",".join(str(i) for i in engine_ids[:20])
        where = f"WHERE id IN ({id_list})"
    else:
        # Pick the engines with lowest final RUL (most critical)
        sub = _sql_query(f"""
            SELECT id FROM (
                SELECT id, MIN(RemainingUsefulLife) AS final_rul
                FROM {_ENG_TBL}
                GROUP BY id
                ORDER BY final_rul ASC
                LIMIT {limit_engines}
            ) t
        """)
        if len(sub) == 0:
            return pd.DataFrame(columns=["engine_id", "cycle", "rul"])
        id_list = ",".join(str(i) for i in sub["id"].tolist())
        where = f"WHERE id IN ({id_list})"

    return _sql_query(f"""
        SELECT id AS engine_id, Cycle AS cycle, RemainingUsefulLife AS rul
        FROM {_ENG_TBL}
        {where}
        ORDER BY id, cycle
    """)


def get_engine_latest_status(limit: int = 100) -> pd.DataFrame:
    """Return one row per engine showing latest cycle stats — for the summary table."""
    return _sql_query(f"""
        SELECT
            id AS engine_id,
            MAX(Cycle) AS total_cycles,
            MIN(RemainingUsefulLife) AS remaining_rul,
            ROUND(AVG(SensorMeasure2), 2) AS avg_temp,
            ROUND(AVG(SensorMeasure3), 2) AS avg_pressure,
            CASE
                WHEN MIN(RemainingUsefulLife) < 50  THEN 'Critical'
                WHEN MIN(RemainingUsefulLife) < 100 THEN 'Warning'
                ELSE 'Healthy'
            END AS status
        FROM {_ENG_TBL}
        GROUP BY id
        ORDER BY remaining_rul ASC
        LIMIT {limit}
    """)


# ── Electrical fault functions ─────────────────────────────────────────────────
def get_electrical_fault_summary() -> pd.DataFrame:
    """Return fault type distribution (G, C, B, A phase faults)."""
    df = _sql_query(f"SELECT SUM(G) AS g, SUM(C) AS c, SUM(B) AS b, SUM(A) AS a FROM {_ELC_TBL}")
    row = df.iloc[0]
    return pd.DataFrame({
        "fault_type": ["Ground (G)", "Phase-C (C)", "Phase-B (B)", "Phase-A (A)"],
        "count": [int(row["g"] or 0), int(row["c"] or 0), int(row["b"] or 0), int(row["a"] or 0)],
    })


def get_electrical_phase_data(limit: int = 500) -> pd.DataFrame:
    """Return phase current and voltage readings for scatter analysis."""
    return _sql_query(f"""
        SELECT
            Ia AS ia, Ib AS ib, Ic AS ic,
            Va AS va, Vb AS vb, Vc AS vc,
            (G + C + B + A) AS has_fault
        FROM {_ELC_TBL}
        LIMIT {limit}
    """)


# ── Transformer functions ──────────────────────────────────────────────────────
def get_transformer_trend(limit: int = 300) -> pd.DataFrame:
    """Return transformer readings for time-series charts."""
    return _sql_query(f"""
        SELECT
            DeviceTimeStamp AS ts,
            OTI AS oti, WTI AS wti, ATI AS ati,
            VL1 AS vl1, VL2 AS vl2, VL3 AS vl3,
            IL1 AS il1, IL2 AS il2, IL3 AS il3,
            INUT AS inut
        FROM {_TRN_TBL}
        ORDER BY DeviceTimeStamp
        LIMIT {limit}
    """)


def get_transformer_summary() -> pd.DataFrame:
    """Return aggregate transformer health statistics."""
    return _sql_query(f"""
        SELECT
            ROUND(AVG(OTI), 2) AS avg_oti,
            ROUND(MAX(OTI), 2) AS max_oti,
            ROUND(AVG(WTI), 2) AS avg_wti,
            ROUND(AVG((VL1 + VL2 + VL3) / 3.0), 1) AS avg_voltage,
            ROUND(AVG((IL1 + IL2 + IL3) / 3.0), 2) AS avg_current
        FROM {_TRN_TBL}
    """)


# ── Schema context for AI chat ────────────────────────────────────────────────
def get_schema_context() -> str:
    """Return schema description string for Bedrock Text-to-SQL prompt."""
    catalog = _CATALOG
    schema  = _SCHEMA
    return f"""
Tables in {catalog}.{schema}:

1. {catalog}.{schema}.cnc_machine_data
   Columns: UDI (int), "Product ID" (str), Type (str: H/M/L),
   "Air temperature [K]" (float), "Process temperature [K]" (float),
   "Rotational speed [rpm]" (int), "Torque [Nm]" (float), "Tool wear [min]" (int),
   "Machine failure" (0/1), TWF (0/1), HDF (0/1), PWF (0/1), OSF (0/1), RNF (0/1)
   Note: Type H=High quality, M=Medium, L=Low; failure modes are binary flags

2. {catalog}.{schema}.nasa_engine_rul
   Columns: id (int engine ID), Cycle (int), OpSet1/2/3 (float),
   SensorMeasure1..SensorMeasure21 (float), RemainingUsefulLife (int)
   Note: RUL = cycles remaining before maintenance needed; lower = more urgent

3. {catalog}.{schema}.electrical_fault
   Columns: G (0/1 ground fault), C (0/1 phase-C fault), B (0/1 phase-B fault),
   A (0/1 phase-A fault), Ia/Ib/Ic (float amps), Va/Vb/Vc (float volts)

4. {catalog}.{schema}.transformer_reading
   Columns: DeviceTimeStamp (timestamp), OTI (float oil temp), WTI (float winding temp),
   ATI (float ambient temp), OLI (float oil level), VL1/VL2/VL3 (float volts),
   IL1/IL2/IL3 (float amps), VL12/VL23/VL31 (float line volts), INUT (float neutral current)

Sample questions: "Which machine type fails most often?",
"Show engines with RUL under 50 cycles", "What is the average torque when failures occur?"
""".strip()
