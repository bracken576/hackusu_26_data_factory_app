"""Engine Health tab — NASA turbofan RUL tracking with maintenance schedule predictor."""
import logging
from datetime import date, timedelta

import gradio as gr
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from services import db_service, auth_service, audit_service

logger = logging.getLogger(__name__)

_STATUS_COLOR = {"Critical": "#E53935", "Warning": "#FB8C00", "Healthy": "#43A047"}


def build() -> None:
    """Build the Engine Health tab inside a gr.Blocks context."""

    gr.Markdown("## Engine Health & Remaining Useful Life")
    gr.Markdown(
        "NASA turbofan degradation dataset — track engine RUL across operating cycles.  \n"
        "_Source: `main.predictive_maintenance.nasa_engine_rul`_"
    )

    with gr.Row():
        load_btn = gr.Button("Load Engine Data", variant="primary")
        status   = gr.Markdown("")

    # ── Summary KPI row ────────────────────────────────────────────────────────
    with gr.Row():
        kpi_critical = gr.Markdown("### Critical Engines\n# **—**")
        kpi_warning  = gr.Markdown("### Warning Engines\n# **—**")
        kpi_healthy  = gr.Markdown("### Healthy Engines\n# **—**")
        kpi_avg_rul  = gr.Markdown("### Fleet Avg RUL\n# **—**")

    # ── Charts ─────────────────────────────────────────────────────────────────
    with gr.Row():
        rul_trend   = gr.Plot(label="RUL Over Time — Top 10 Most Critical Engines")
        rul_buckets = gr.Plot(label="Fleet Health Distribution")

    with gr.Row():
        sensor_plot = gr.Plot(label="Sensor Profile (Avg SensorMeasure2 & 7 by Cycle)")
        status_bar  = gr.Plot(label="Engine Status Breakdown")

    gr.Markdown("### Engine Status Table")
    engine_table = gr.DataFrame(label="All Engines — sorted by lowest RUL", height=350)

    # ── STRETCH GOAL: Maintenance Schedule ─────────────────────────────────────
    with gr.Accordion("Maintenance Schedule Predictor (Stretch Goal)", open=False):
        gr.Markdown(
            "Automatically generates a 30-day maintenance calendar based on each engine's "
            "RUL. Engines with RUL ≤ 30 cycles are flagged as **URGENT** (schedule within 7 days)."
        )
        sched_btn = gr.Button("Generate Schedule", variant="secondary")
        sched_tbl = gr.DataFrame(label="Recommended Maintenance Schedule", height=300)

    # ── Event handlers ─────────────────────────────────────────────────────────
    def load_data(request: gr.Request):
        user  = auth_service.get_user_from_request(request)
        email = user["email"] if user else "unknown"
        role  = user["role"]  if user else "viewer"

        try:
            # RUL trend for critical engines
            trend_df  = db_service.get_engine_rul_trend(limit_engines=10)
            trend_fig = go.Figure()
            if len(trend_df) > 0:
                for eng_id, grp in trend_df.groupby("engine_id"):
                    trend_fig.add_trace(go.Scatter(
                        x=grp["cycle"], y=grp["rul"],
                        mode="lines", name=f"Engine {eng_id}",
                        line=dict(width=1.5),
                    ))
                trend_fig.add_hline(y=50, line_dash="dash", line_color="#E53935",
                                    annotation_text="Critical threshold (50)")
                trend_fig.update_layout(
                    title="RUL Over Cycles — 10 Most Critical Engines",
                    xaxis_title="Cycle", yaxis_title="Remaining Useful Life",
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="top", y=-0.2),
                )

            # Health buckets
            bucket_df  = db_service.get_engine_rul_buckets()
            bucket_fig = px.pie(
                bucket_df, names="bucket", values="count",
                color="bucket",
                color_discrete_map={
                    "Critical (<50)":  "#E53935",
                    "Warning (50-99)": "#FB8C00",
                    "Healthy (≥100)":  "#43A047",
                },
                title="Fleet Health Distribution",
                hole=0.45,
            )

            # Sensor trend (averaged over all engines per cycle)
            sensor_raw = db_service._sql_query(f"""
                SELECT Cycle AS cycle,
                       ROUND(AVG(SensorMeasure2), 2) AS sensor2,
                       ROUND(AVG(SensorMeasure7), 2) AS sensor7
                FROM {db_service._ENG_TBL}
                GROUP BY Cycle
                ORDER BY Cycle
                LIMIT 500
            """)
            sensor_fig = go.Figure()
            sensor_fig.add_trace(go.Scatter(x=sensor_raw["cycle"], y=sensor_raw["sensor2"],
                                            name="Sensor 2 (Temp)", mode="lines"))
            sensor_fig.add_trace(go.Scatter(x=sensor_raw["cycle"], y=sensor_raw["sensor7"],
                                            name="Sensor 7 (Fan Speed)", mode="lines"))
            sensor_fig.update_layout(title="Fleet-Average Sensor Readings by Cycle",
                                     xaxis_title="Cycle", yaxis_title="Reading")

            # Status bar
            stat_fig = px.bar(
                bucket_df, x="bucket", y="count",
                color="bucket",
                color_discrete_map={
                    "Critical (<50)":  "#E53935",
                    "Warning (50-99)": "#FB8C00",
                    "Healthy (≥100)":  "#43A047",
                },
                title="Engine Count by Status",
                labels={"bucket": "Status", "count": "Engines"},
            )
            stat_fig.update_layout(showlegend=False)

            # Engine table
            eng_df = db_service.get_engine_latest_status(limit=200)

            # KPI cards
            critical = int(bucket_df[bucket_df["bucket"] == "Critical (<50)"]["count"].sum())
            warning  = int(bucket_df[bucket_df["bucket"] == "Warning (50-99)"]["count"].sum())
            healthy  = int(bucket_df[bucket_df["bucket"] == "Healthy (≥100)"]["count"].sum())
            avg_rul  = int(eng_df["remaining_rul"].mean()) if len(eng_df) > 0 else 0

            audit_service.log_event(
                action_type="QUERY",
                user_email=email, user_role=role,
                source_tables=db_service.ENGINE_TABLE,
                query_text="Engine health tab load",
                row_count=len(eng_df),
            )

            return (
                trend_fig, bucket_fig, sensor_fig, stat_fig, eng_df,
                f"### Critical Engines\n# **{critical}**",
                f"### Warning Engines\n# **{warning}**",
                f"### Healthy Engines\n# **{healthy}**",
                f"### Fleet Avg RUL\n# **{avg_rul} cycles**",
                "",
            )

        except Exception as exc:
            logger.error("Engine tab load error: %s", exc)
            empty = go.Figure()
            return (empty, empty, empty, empty, gr.update(),
                    "### Critical Engines\n# **—**",
                    "### Warning Engines\n# **—**",
                    "### Healthy Engines\n# **—**",
                    "### Fleet Avg RUL\n# **—**",
                    f"⚠️ {str(exc)[:120]}")

    def generate_schedule(request: gr.Request):
        """STRETCH: Build a maintenance schedule from current RUL data."""
        try:
            eng_df = db_service.get_engine_latest_status(limit=500)
            today  = date.today()
            rows   = []
            for _, row in eng_df.iterrows():
                rul = row["remaining_rul"]
                if rul < 0:
                    rul = 0
                # Assume 1 cycle ≈ 1 day of operation
                if rul <= 30:
                    sched_date = today + timedelta(days=max(1, int(rul * 0.5)))
                    priority   = "URGENT" if rul <= 15 else "HIGH"
                elif rul <= 100:
                    sched_date = today + timedelta(days=int(rul * 0.6))
                    priority   = "MEDIUM"
                else:
                    continue  # Healthy — no near-term action needed

                rows.append({
                    "Engine ID":         int(row["engine_id"]),
                    "Current Status":    row["status"],
                    "RUL (cycles)":      int(rul),
                    "Scheduled Date":    sched_date.isoformat(),
                    "Priority":          priority,
                    "Action":            "Preventive Maintenance Inspection",
                })
            if not rows:
                rows = [{"Engine ID": "—", "note": "All engines healthy — no urgent maintenance"}]
            return pd.DataFrame(rows)
        except Exception as exc:
            return pd.DataFrame({"error": [str(exc)]})

    load_btn.click(
        fn=load_data,
        outputs=[rul_trend, rul_buckets, sensor_plot, status_bar, engine_table,
                 kpi_critical, kpi_warning, kpi_healthy, kpi_avg_rul, status],
    )
    sched_btn.click(fn=generate_schedule, outputs=[sched_tbl])
