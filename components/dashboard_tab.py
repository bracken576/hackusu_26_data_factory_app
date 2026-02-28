"""Overview Dashboard tab — KPI cards + summary charts across all asset types."""
import logging

import gradio as gr
import plotly.express as px
import plotly.graph_objects as go

from services import db_service, auth_service, audit_service

logger = logging.getLogger(__name__)

_STATUS_COLORS = {
    "Critical (<50)":  "#E53935",
    "Warning (50-99)": "#FB8C00",
    "Healthy (≥100)":  "#43A047",
}


def build(summary: dict) -> list:
    """
    Build the Overview tab inside a gr.Blocks context.
    Returns list of chart components that app.py wires to demo.load().
    """
    # ── KPI Row ────────────────────────────────────────────────────────────────
    gr.Markdown("## Asset Health Overview")
    gr.Markdown(
        "_Source: `main.predictive_maintenance.*` · Unity Catalog · "
        "Predictive Maintenance & Asset Management (Dataknobs)_"
    )

    with gr.Row():
        with gr.Column(scale=1, min_width=150):
            gr.Markdown("### Equipment Health")
            health_kpi = gr.Markdown(
                f"# **{summary.get('health_score', '--')}%**",
                elem_id="kpi-health",
            )
        with gr.Column(scale=1, min_width=150):
            gr.Markdown("### Avg Tool Wear")
            wear_kpi = gr.Markdown(
                f"# **{summary.get('avg_tool_wear', '--')} min**",
                elem_id="kpi-wear",
            )
        with gr.Column(scale=1, min_width=150):
            gr.Markdown("### Avg Engine RUL")
            rul_kpi = gr.Markdown(
                f"# **{summary.get('avg_rul', '--')} cycles**",
                elem_id="kpi-rul",
            )
        with gr.Column(scale=1, min_width=150):
            gr.Markdown("### Critical Engines")
            crit_kpi = gr.Markdown(
                f"# **{summary.get('critical_engines', '--')}**",
                elem_id="kpi-crit",
            )
        with gr.Column(scale=1, min_width=150):
            gr.Markdown("### Total CNC Failures")
            fail_kpi = gr.Markdown(
                f"# **{summary.get('total_failures', '--')}**",
                elem_id="kpi-fail",
            )
        with gr.Column(scale=1, min_width=150):
            gr.Markdown("### Electrical Fault Rate")
            elec_kpi = gr.Markdown(
                f"# **{summary.get('elec_fault_rate', '--')}%**",
                elem_id="kpi-elec",
            )

    with gr.Row():
        refresh_btn = gr.Button("Refresh Charts", variant="secondary", size="sm")
        status_msg  = gr.Markdown("")

    # ── Charts ─────────────────────────────────────────────────────────────────
    with gr.Row():
        failure_modes_chart = gr.Plot(label="CNC Failure Mode Breakdown")
        rul_buckets_chart   = gr.Plot(label="Engine Health Distribution")

    with gr.Row():
        torque_chart  = gr.Plot(label="Torque vs RPM (coloured by failure)")
        type_chart    = gr.Plot(label="Failure Rate & Tool Wear by Machine Type")

    # Anomaly alert banner
    gr.Markdown("### ⚠️ Anomaly Alerts (tool wear > 200 min or torque > 65 Nm)")
    anomaly_table = gr.DataFrame(label="At-Risk Assets", height=250)

    # ── Event handlers ─────────────────────────────────────────────────────────
    def load_charts(request: gr.Request):
        user  = auth_service.get_user_from_request(request)
        email = user["email"] if user else "unknown"
        role  = user["role"]  if user else "viewer"

        try:
            # --- Failure modes bar chart ---
            fm_df  = db_service.get_cnc_failure_modes()
            fm_fig = px.bar(
                fm_df, x="failure_mode", y="count",
                color="count", color_continuous_scale="Reds",
                title="CNC Failure Mode Breakdown",
                labels={"failure_mode": "Failure Mode", "count": "Incidents"},
            )
            fm_fig.update_layout(coloraxis_showscale=False, showlegend=False)

            # --- RUL health buckets pie ---
            rul_df  = db_service.get_engine_rul_buckets()
            rul_fig = px.pie(
                rul_df, names="bucket", values="count",
                color="bucket",
                color_discrete_map=_STATUS_COLORS,
                title="Engine Health Distribution",
                hole=0.4,
            )

            # --- Torque vs RPM scatter ---
            sc_df    = db_service.get_cnc_scatter_data(limit=2000)
            tor_fig  = px.scatter(
                sc_df, x="rpm", y="torque_nm",
                color="failure_label",
                color_discrete_map={"Normal": "#43A047", "Failure": "#E53935"},
                opacity=0.6,
                title="Torque vs RPM",
                labels={"rpm": "Rotational Speed (RPM)", "torque_nm": "Torque (Nm)",
                        "failure_label": "Status"},
            )

            # --- Type comparison bar ---
            typ_df  = db_service.get_cnc_failure_by_type()
            typ_fig = go.Figure()
            typ_fig.add_trace(go.Bar(
                name="Avg Tool Wear (min)", x=typ_df["machine_type"],
                y=typ_df["avg_tool_wear_min"], marker_color="#1976D2",
            ))
            typ_fig.add_trace(go.Bar(
                name="Failure Rate (%)", x=typ_df["machine_type"],
                y=typ_df["failure_rate_pct"], marker_color="#E53935",
                yaxis="y2",
            ))
            typ_fig.update_layout(
                title="Tool Wear & Failure Rate by Machine Type",
                yaxis=dict(title="Avg Tool Wear (min)"),
                yaxis2=dict(title="Failure Rate (%)", overlaying="y", side="right"),
                barmode="group",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )

            # --- Anomaly table ---
            anom_df = db_service.get_cnc_anomalies(limit=30)

            audit_service.log_event(
                action_type="QUERY",
                user_email=email, user_role=role,
                source_tables=f"{db_service.CNC_TABLE}, {db_service.ENGINE_TABLE}",
                query_text="Dashboard overview load",
                row_count=len(sc_df),
            )

            return fm_fig, rul_fig, tor_fig, typ_fig, anom_df, ""

        except Exception as exc:
            logger.error("Dashboard load_charts error: %s", exc)
            empty = go.Figure()
            empty.update_layout(title="Data unavailable")
            return empty, empty, empty, empty, gr.update(), f"⚠️ {str(exc)[:120]}"

    outputs = [failure_modes_chart, rul_buckets_chart, torque_chart,
               type_chart, anomaly_table, status_msg]
    refresh_btn.click(fn=load_charts, outputs=outputs)

    # Return (outputs, load_fn) so app.py can wire demo.load()
    return outputs, load_charts
