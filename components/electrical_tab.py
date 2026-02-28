"""Electrical Monitor tab — fault detection for electrical systems and transformers."""
import logging

import gradio as gr
import plotly.express as px
import plotly.graph_objects as go

from services import db_service, auth_service, audit_service

logger = logging.getLogger(__name__)


def build() -> None:
    """Build the Electrical Monitor tab inside a gr.Blocks context."""

    gr.Markdown("## Electrical Fault & Transformer Monitor")
    gr.Markdown(
        "Phase fault detection and transformer health monitoring.  \n"
        "_Sources: `electrical_fault`, `transformer_reading` — main.predictive_maintenance_"
    )

    with gr.Row():
        load_btn = gr.Button("Load Electrical Data", variant="primary")
        status   = gr.Markdown("")

    with gr.Tabs():
        # ── Electrical Fault sub-tab ───────────────────────────────────────────
        with gr.Tab("Phase Fault Analysis"):
            with gr.Row():
                fault_pie    = gr.Plot(label="Fault Type Distribution (G / C / B / A)")
                current_scat = gr.Plot(label="Phase Currents — Ia vs Ib (coloured by fault)")

            with gr.Row():
                voltage_scat = gr.Plot(label="Phase Voltages — Va vs Vb")
                abc_scatter  = gr.Plot(label="3-Phase Current Balance (Ia / Ib / Ic)")

        # ── Transformer sub-tab ────────────────────────────────────────────────
        with gr.Tab("Transformer Status"):
            with gr.Row():
                trans_kpi_oti = gr.Markdown("### Avg Oil Temp\n# **—°C**")
                trans_kpi_wti = gr.Markdown("### Avg Winding Temp\n# **—°C**")
                trans_kpi_v   = gr.Markdown("### Avg Voltage\n# **—V**")
                trans_kpi_i   = gr.Markdown("### Avg Current\n# **—A**")

            with gr.Row():
                temp_trend  = gr.Plot(label="OTI & WTI Temperature Trend")
                voltage_bar = gr.Plot(label="Line Voltage Comparison (VL1 / VL2 / VL3)")

            with gr.Row():
                current_bar = gr.Plot(label="Phase Currents (IL1 / IL2 / IL3)")
                inut_plot   = gr.Plot(label="Neutral Current (INUT) — Anomaly Indicator")

    # ── Event handlers ─────────────────────────────────────────────────────────
    def load_data(request: gr.Request):
        user  = auth_service.get_user_from_request(request)
        email = user["email"] if user else "unknown"
        role  = user["role"]  if user else "viewer"

        try:
            # Fault type pie
            fault_df  = db_service.get_electrical_fault_summary()
            fault_fig = px.pie(
                fault_df, names="fault_type", values="count",
                title="Fault Type Distribution",
                color_discrete_sequence=px.colors.qualitative.Set2,
                hole=0.35,
            )

            # Phase scatter data
            phase_df = db_service.get_electrical_phase_data(limit=500)
            phase_df["fault_label"] = phase_df["has_fault"].apply(
                lambda x: "Fault" if x > 0 else "Normal"
            )

            curr_fig = px.scatter(
                phase_df, x="ia", y="ib",
                color="fault_label",
                color_discrete_map={"Normal": "#43A047", "Fault": "#E53935"},
                opacity=0.6,
                title="Ia vs Ib Phase Currents",
                labels={"ia": "Phase A Current (A)", "ib": "Phase B Current (A)",
                        "fault_label": "Status"},
            )

            volt_fig = px.scatter(
                phase_df, x="va", y="vb",
                color="fault_label",
                color_discrete_map={"Normal": "#1976D2", "Fault": "#E53935"},
                opacity=0.6,
                title="Va vs Vb Phase Voltages",
                labels={"va": "Phase A Voltage (V)", "vb": "Phase B Voltage (V)",
                        "fault_label": "Status"},
            )

            abc_fig = go.Figure()
            abc_fig.add_trace(go.Scatter(
                y=phase_df["ia"].head(200), name="Ia", mode="lines", line=dict(color="#E53935")))
            abc_fig.add_trace(go.Scatter(
                y=phase_df["ib"].head(200), name="Ib", mode="lines", line=dict(color="#1976D2")))
            abc_fig.add_trace(go.Scatter(
                y=phase_df["ic"].head(200), name="Ic", mode="lines", line=dict(color="#43A047")))
            abc_fig.update_layout(
                title="3-Phase Current Balance",
                xaxis_title="Sample", yaxis_title="Current (A)",
            )

            # Transformer data
            trans_df   = db_service.get_transformer_trend(limit=300)
            trans_summ = db_service.get_transformer_summary()

            if len(trans_summ) > 0:
                r = trans_summ.iloc[0]
                oti_md = f"### Avg Oil Temp\n# **{r.get('avg_oti', '—')} °C**"
                wti_md = f"### Avg Winding Temp\n# **{r.get('avg_wti', '—')} °C**"
                v_md   = f"### Avg Voltage\n# **{r.get('avg_voltage', '—')} V**"
                i_md   = f"### Avg Current\n# **{r.get('avg_current', '—')} A**"
            else:
                oti_md = wti_md = v_md = i_md = "# **—**"

            temp_fig = go.Figure()
            if len(trans_df) > 0:
                temp_fig.add_trace(go.Scatter(
                    x=trans_df["ts"], y=trans_df["oti"], name="OTI (Oil Temp)", mode="lines"))
                temp_fig.add_trace(go.Scatter(
                    x=trans_df["ts"], y=trans_df["wti"], name="WTI (Winding Temp)", mode="lines"))
            temp_fig.update_layout(
                title="Transformer Temperature Trend",
                xaxis_title="Timestamp", yaxis_title="Temperature Index",
            )

            v_bar_fig = go.Figure()
            if len(trans_df) > 0:
                v_bar_fig.add_trace(go.Box(y=trans_df["vl1"], name="VL1"))
                v_bar_fig.add_trace(go.Box(y=trans_df["vl2"], name="VL2"))
                v_bar_fig.add_trace(go.Box(y=trans_df["vl3"], name="VL3"))
            v_bar_fig.update_layout(
                title="Line Voltage Distribution (VL1 / VL2 / VL3)",
                yaxis_title="Voltage (V)",
            )

            i_bar_fig = go.Figure()
            if len(trans_df) > 0:
                i_bar_fig.add_trace(go.Box(y=trans_df["il1"], name="IL1"))
                i_bar_fig.add_trace(go.Box(y=trans_df["il2"], name="IL2"))
                i_bar_fig.add_trace(go.Box(y=trans_df["il3"], name="IL3"))
            i_bar_fig.update_layout(
                title="Phase Current Distribution (IL1 / IL2 / IL3)",
                yaxis_title="Current (A)",
            )

            inut_fig = go.Figure()
            if len(trans_df) > 0:
                inut_fig.add_trace(go.Scatter(
                    x=trans_df["ts"], y=trans_df["inut"],
                    mode="lines", line=dict(color="#E53935"),
                    name="Neutral Current (INUT)",
                ))
                # Anomaly threshold (3-sigma)
                mean_inut = trans_df["inut"].mean()
                std_inut  = trans_df["inut"].std()
                inut_fig.add_hline(y=mean_inut + 3 * std_inut,
                                   line_dash="dash", line_color="orange",
                                   annotation_text="3σ anomaly threshold")
            inut_fig.update_layout(
                title="Neutral Current — Imbalance Indicator",
                xaxis_title="Timestamp", yaxis_title="Current (A)",
            )

            audit_service.log_event(
                action_type="QUERY",
                user_email=email, user_role=role,
                source_tables=f"{db_service.ELEC_TABLE}, {db_service.TRANSFORMER_TABLE}",
                query_text="Electrical monitor tab load",
                row_count=len(phase_df),
            )

            return (fault_fig, curr_fig, volt_fig, abc_fig,
                    oti_md, wti_md, v_md, i_md,
                    temp_fig, v_bar_fig, i_bar_fig, inut_fig, "")

        except Exception as exc:
            logger.error("Electrical tab load error: %s", exc)
            empty = go.Figure()
            return (empty, empty, empty, empty,
                    "# **—**", "# **—**", "# **—**", "# **—**",
                    empty, empty, empty, empty, f"⚠️ {str(exc)[:120]}")

    load_btn.click(
        fn=load_data,
        outputs=[fault_pie, current_scat, voltage_scat, abc_scatter,
                 trans_kpi_oti, trans_kpi_wti, trans_kpi_v, trans_kpi_i,
                 temp_trend, voltage_bar, current_bar, inut_plot, status],
    )
