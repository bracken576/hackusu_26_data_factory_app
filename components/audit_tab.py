"""Audit Log tab — analyst and admin only, append-only query history."""
import logging

import gradio as gr
import pandas as pd

from services import auth_service, audit_service

logger = logging.getLogger(__name__)


def build() -> None:
    """Build the Audit Log tab. Role enforcement happens inside event handlers."""

    gr.Markdown("## Audit Log")
    gr.Markdown(
        "Governance trail of all queries, chat interactions, and access events.  \n"
        "_Accessible to **analyst** and **admin** roles only._"
    )

    with gr.Row():
        refresh_btn  = gr.Button("Refresh Log", variant="primary")
        export_btn   = gr.Button("Export CSV", variant="secondary")
        status       = gr.Markdown("")

    with gr.Row():
        action_filter = gr.Dropdown(
            choices=["All", "QUERY", "CHAT", "EXPORT", "LOGIN",
                     "ACCESS_DENIED", "QUERY_FAILED"],
            value="All",
            label="Filter by Action",
        )

    audit_table = gr.DataFrame(label="Audit Trail", height=500)
    export_file = gr.File(label="Download CSV", visible=False)

    def load_log(action_type_filter: str, request: gr.Request):
        user = auth_service.get_user_from_request(request)
        if not user or user["role"] not in ("analyst", "admin"):
            audit_service.log_event(
                action_type="ACCESS_DENIED",
                user_email=user["email"] if user else "unknown",
                user_role=user["role"] if user else "viewer",
            )
            return (
                pd.DataFrame({"message": ["Access denied. Analyst or Admin role required."]}),
                gr.update(visible=False),
                "⚠️ Access denied.",
            )

        df = audit_service.read_audit_log(limit=500)
        if action_type_filter != "All" and len(df) > 0:
            df = df[df["action_type"] == action_type_filter]
        return df, gr.update(visible=False), f"Showing {len(df):,} entries."

    def export_log(request: gr.Request):
        user = auth_service.get_user_from_request(request)
        if not user or user["role"] not in ("analyst", "admin"):
            return gr.update(visible=False), "⚠️ Export requires analyst or admin role."

        import tempfile, os
        df = audit_service.read_audit_log(limit=10_000)
        tmp = tempfile.NamedTemporaryFile(
            suffix=".csv", prefix="audit_export_", delete=False, mode="w", encoding="utf-8"
        )
        df.to_csv(tmp.name, index=False)
        tmp.close()
        audit_service.log_event(
            action_type="EXPORT",
            user_email=user["email"], user_role=user["role"],
            source_tables="logs/audit_trail.csv",
            row_count=len(df),
        )
        return gr.update(value=tmp.name, visible=True), f"Exported {len(df):,} rows."

    refresh_btn.click(
        fn=load_log,
        inputs=[action_filter],
        outputs=[audit_table, export_file, status],
    )
    action_filter.change(
        fn=load_log,
        inputs=[action_filter],
        outputs=[audit_table, export_file, status],
    )
    export_btn.click(fn=export_log, outputs=[export_file, status])
