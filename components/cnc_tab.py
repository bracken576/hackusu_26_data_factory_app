"""CNC Machine Analysis tab — failure mode deep-dive with interactive filters."""
import logging

import gradio as gr
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from services import db_service, auth_service, audit_service

logger = logging.getLogger(__name__)


def build() -> None:
    """Build the CNC Machine Analysis tab inside a gr.Blocks context."""

    gr.Markdown("## CNC Machine Analysis")
    gr.Markdown(
        "Deep-dive into CNC machine operating conditions and failure patterns.  \n"
        "_Source: `main.predictive_maintenance.cnc_machine_data`_"
    )

    # ── Filters ────────────────────────────────────────────────────────────────
    with gr.Row():
        type_filter = gr.Dropdown(
            choices=["All", "H (High Quality)", "M (Medium Quality)", "L (Low Quality)"],
            value="All",
            label="Machine Type Filter",
        )
        load_btn = gr.Button("Load / Refresh", variant="primary")
        status   = gr.Markdown("")

    # ── Charts ─────────────────────────────────────────────────────────────────
    with gr.Row():
        temp_scatter  = gr.Plot(label="Air vs Process Temperature (coloured by failure)")
        wear_scatter  = gr.Plot(label="Tool Wear vs Torque (coloured by failure)")

    with gr.Row():
        failure_bar   = gr.Plot(label="Failure Rate by Machine Type")
        tool_hist     = gr.Plot(label="Tool Wear Distribution")

    gr.Markdown("### Top At-Risk Machines (High Wear / High Torque)")
    risk_table = gr.DataFrame(label="At-Risk Machines", height=300)

    # ── STRETCH GOAL: Health Scorecard ─────────────────────────────────────────
    with gr.Accordion("Asset Health Scorecard (Stretch Goal)", open=False):
        gr.Markdown(
            "Each machine receives a composite 0–100 health score combining "
            "tool wear normalisation and failure history."
        )
        scorecard_btn  = gr.Button("Generate Scorecard", variant="secondary")
        scorecard_plot = gr.Plot(label="Health Score Distribution")
        scorecard_tbl  = gr.DataFrame(label="Per-Type Score Summary", height=200)

    # ── Event handlers ─────────────────────────────────────────────────────────
    def load_data(machine_type_label: str, request: gr.Request):
        user  = auth_service.get_user_from_request(request)
        email = user["email"] if user else "unknown"
        role  = user["role"]  if user else "viewer"

        type_map = {
            "H (High Quality)": "H",
            "M (Medium Quality)": "M",
            "L (Low Quality)": "L",
        }
        mtype = type_map.get(machine_type_label)  # None = All

        try:
            df = db_service.get_cnc_scatter_data(machine_type=mtype, limit=5000)

            # Temperature scatter
            temp_fig = px.scatter(
                df, x="air_temp_k", y="proc_temp_k",
                color="failure_label",
                color_discrete_map={"Normal": "#43A047", "Failure": "#E53935"},
                opacity=0.5,
                title="Air vs Process Temperature",
                labels={"air_temp_k": "Air Temp (K)", "proc_temp_k": "Process Temp (K)",
                        "failure_label": "Status"},
            )

            # Wear vs torque scatter
            wear_fig = px.scatter(
                df, x="tool_wear_min", y="torque_nm",
                color="failure_label",
                color_discrete_map={"Normal": "#1976D2", "Failure": "#E53935"},
                opacity=0.5,
                title="Tool Wear vs Torque",
                labels={"tool_wear_min": "Tool Wear (min)", "torque_nm": "Torque (Nm)",
                        "failure_label": "Status"},
            )

            # Failure rate by type bar
            typ_df   = db_service.get_cnc_failure_by_type()
            fail_fig = px.bar(
                typ_df, x="machine_type", y="failure_rate_pct",
                color="failure_rate_pct", color_continuous_scale="Reds",
                text="failure_rate_pct",
                title="Failure Rate (%) by Machine Type",
                labels={"machine_type": "Type", "failure_rate_pct": "Failure Rate (%)"},
            )
            fail_fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fail_fig.update_layout(coloraxis_showscale=False)

            # Tool wear histogram
            hist_fig = px.histogram(
                df, x="tool_wear_min",
                color="failure_label",
                color_discrete_map={"Normal": "#1976D2", "Failure": "#E53935"},
                nbins=30,
                title="Tool Wear Distribution",
                labels={"tool_wear_min": "Tool Wear (min)", "count": "Count"},
                barmode="overlay",
                opacity=0.7,
            )

            # At-risk table
            risk_df = db_service.get_cnc_anomalies(limit=50)
            if mtype:
                risk_df = risk_df[risk_df["machine_type"] == mtype]

            audit_service.log_event(
                action_type="QUERY",
                user_email=email, user_role=role,
                source_tables=db_service.CNC_TABLE,
                query_text=f"CNC analysis: type={mtype or 'All'}",
                row_count=len(df),
            )
            return temp_fig, wear_fig, fail_fig, hist_fig, risk_df, ""

        except Exception as exc:
            logger.error("CNC tab load error: %s", exc)
            empty = go.Figure()
            return empty, empty, empty, empty, gr.update(), f"⚠️ {str(exc)[:120]}"

    def generate_scorecard(request: gr.Request):
        """STRETCH: Compute composite health score per machine type."""
        try:
            typ_df = db_service.get_cnc_failure_by_type()
            # Normalize tool wear (lower is better) and failure rate (lower is better)
            max_wear = typ_df["avg_tool_wear_min"].max()
            max_fail = typ_df["failure_rate_pct"].max()
            typ_df["wear_score"]   = (1 - typ_df["avg_tool_wear_min"] / max_wear) * 50
            typ_df["fail_score"]   = (1 - typ_df["failure_rate_pct"]  / max_fail) * 50
            typ_df["health_score"] = (typ_df["wear_score"] + typ_df["fail_score"]).round(1)

            score_fig = px.bar(
                typ_df, x="machine_type", y="health_score",
                color="health_score",
                color_continuous_scale="RdYlGn",
                range_color=[0, 100],
                text="health_score",
                title="Composite Health Score by Machine Type (0 = worst, 100 = best)",
                labels={"machine_type": "Machine Type", "health_score": "Health Score"},
            )
            score_fig.update_traces(texttemplate="%{text:.0f}", textposition="outside")
            summary_df = typ_df[["machine_type", "health_score", "avg_tool_wear_min",
                                  "failure_rate_pct"]].rename(columns={
                "machine_type": "Type", "health_score": "Health Score",
                "avg_tool_wear_min": "Avg Tool Wear (min)", "failure_rate_pct": "Failure Rate (%)",
            })
            return score_fig, summary_df
        except Exception as exc:
            return go.Figure(), pd.DataFrame({"error": [str(exc)]})

    load_btn.click(
        fn=load_data,
        inputs=[type_filter],
        outputs=[temp_scatter, wear_scatter, failure_bar, tool_hist, risk_table, status],
    )
    type_filter.change(
        fn=load_data,
        inputs=[type_filter],
        outputs=[temp_scatter, wear_scatter, failure_bar, tool_hist, risk_table, status],
    )
    scorecard_btn.click(fn=generate_scorecard, outputs=[scorecard_plot, scorecard_tbl])
