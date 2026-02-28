"""Admin tab — IT oversight dashboard with health KPIs, kill switches, and reports."""
import logging
from datetime import datetime, timezone

import gradio as gr
import pandas as pd
import plotly.express as px

from services import auth_service, audit_service

logger = logging.getLogger(__name__)

# Page registry — admin can toggle each page on/off (kill switch)
_PAGES = [
    {"page": "Overview",            "category": "main",  "enabled": True},
    {"page": "CNC Analysis",        "category": "main",  "enabled": True},
    {"page": "Engine Health",       "category": "main",  "enabled": True},
    {"page": "Electrical Monitor",  "category": "main",  "enabled": True},
    {"page": "Ask Your Data",       "category": "main",  "enabled": True},
    {"page": "Audit Log",           "category": "main",  "enabled": True},
    {"page": "Admin",               "category": "system","enabled": True},
]


def build() -> None:
    """Build the Admin tab. All event handlers enforce admin-only access."""

    gr.Markdown("## IT Oversight Dashboard")
    gr.Markdown("_Accessible to **admin** role only._")

    with gr.Row():
        load_btn = gr.Button("Refresh Dashboard", variant="primary")
        status   = gr.Markdown("")

    # ── Health KPIs ────────────────────────────────────────────────────────────
    with gr.Row():
        kpi_total_queries = gr.Markdown("### Total Queries\n# **—**")
        kpi_chat_count    = gr.Markdown("### Chat Interactions\n# **—**")
        kpi_denied_count  = gr.Markdown("### Access Denied\n# **—**")
        kpi_error_count   = gr.Markdown("### Query Errors\n# **—**")

    # ── Charts ─────────────────────────────────────────────────────────────────
    with gr.Row():
        action_chart = gr.Plot(label="Activity by Action Type")
        ai_src_chart = gr.Plot(label="AI Response Source (Genie vs Bedrock vs Mock)")

    # ── Recent Audit Entries ──────────────────────────────────────────────────
    gr.Markdown("### Recent Audit Log (last 100 entries)")
    recent_table = gr.DataFrame(height=300)

    # ── Page Registry / Kill Switch ────────────────────────────────────────────
    with gr.Accordion("Page Registry & Kill Switch", open=False):
        gr.Markdown(
            "In a full deployment, toggling `enabled: false` here hides a page "
            "from all non-admin users immediately."
        )
        page_table = gr.DataFrame(
            value=pd.DataFrame(_PAGES),
            label="Registered Pages",
            interactive=False,
            height=250,
        )

    # ── Compliance Report Export ───────────────────────────────────────────────
    with gr.Accordion("Governance Compliance Report", open=False):
        gr.Markdown(
            "Export a report summarising total queries by user/role, PII access events, "
            "access denials, and export events."
        )
        report_btn  = gr.Button("Generate Report", variant="secondary")
        report_file = gr.File(label="Download Governance Report", visible=False)
        report_msg  = gr.Markdown("")

    # ── Event handlers ─────────────────────────────────────────────────────────
    def load_dashboard(request: gr.Request):
        user = auth_service.get_user_from_request(request)
        if not user or user["role"] != "admin":
            return (
                "# **—**", "# **—**", "# **—**", "# **—**",
                _empty_fig(), _empty_fig(),
                pd.DataFrame({"message": ["Admin access required."]}),
                "⚠️ Admin access required.",
            )

        df = audit_service.read_audit_log(limit=10_000)
        if len(df) == 0:
            return (
                "### Total Queries\n# **0**", "### Chat Interactions\n# **0**",
                "### Access Denied\n# **0**", "### Query Errors\n# **0**",
                _empty_fig(), _empty_fig(), df, "No audit data yet.",
            )

        total_q  = len(df[df["action_type"] == "QUERY"])
        total_c  = len(df[df["action_type"] == "CHAT"])
        denied   = len(df[df["action_type"] == "ACCESS_DENIED"])
        errors   = len(df[df["action_type"] == "QUERY_FAILED"])

        action_counts = df["action_type"].value_counts().reset_index()
        action_counts.columns = ["action_type", "count"]
        act_fig = px.bar(
            action_counts, x="action_type", y="count",
            title="Activity by Action Type",
            color="count", color_continuous_scale="Blues",
        )
        act_fig.update_layout(coloraxis_showscale=False)

        ai_df = df[df["ai_source"].notna() & (df["ai_source"] != "")]
        if len(ai_df) > 0:
            ai_counts = ai_df["ai_source"].value_counts().reset_index()
            ai_counts.columns = ["source", "count"]
            ai_fig = px.pie(ai_counts, names="source", values="count",
                            title="AI Response Source", hole=0.4)
        else:
            ai_fig = _empty_fig("No chat data yet")

        audit_service.log_event(
            action_type="QUERY",
            user_email=user["email"], user_role=user["role"],
            source_tables="logs/audit_trail.csv",
            query_text="Admin dashboard load",
            row_count=len(df),
            pii_accessed=True,
        )

        return (
            f"### Total Queries\n# **{total_q:,}**",
            f"### Chat Interactions\n# **{total_c:,}**",
            f"### Access Denied\n# **{denied:,}**",
            f"### Query Errors\n# **{errors:,}**",
            act_fig, ai_fig,
            df.head(100),
            f"Loaded at {datetime.now(timezone.utc).strftime('%H:%M UTC')}",
        )

    def gen_report(request: gr.Request):
        user = auth_service.get_user_from_request(request)
        if not user or user["role"] != "admin":
            return gr.update(visible=False), "⚠️ Admin access required."

        import tempfile
        df = audit_service.read_audit_log(limit=50_000)
        summary = df.groupby(["user_email", "user_role", "action_type"]).size().reset_index(name="count")
        tmp = tempfile.NamedTemporaryFile(
            suffix=".csv", prefix="governance_report_", delete=False, mode="w", encoding="utf-8"
        )
        summary.to_csv(tmp.name, index=False)
        tmp.close()

        audit_service.log_event(
            action_type="EXPORT",
            user_email=user["email"], user_role=user["role"],
            source_tables="logs/audit_trail.csv",
            query_text="Governance compliance report export",
            row_count=len(summary),
        )
        return gr.update(value=tmp.name, visible=True), f"Report generated — {len(summary)} summary rows."

    def _empty_fig(title: str = "No data"):
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.update_layout(title=title)
        return fig

    load_btn.click(
        fn=load_dashboard,
        outputs=[kpi_total_queries, kpi_chat_count, kpi_denied_count, kpi_error_count,
                 action_chart, ai_src_chart, recent_table, status],
    )
    report_btn.click(fn=gen_report, outputs=[report_file, report_msg])
