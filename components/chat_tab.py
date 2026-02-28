"""Ask Your Data tab — Genie → Bedrock conversational interface."""
import logging

import gradio as gr
import pandas as pd

from services import ai_service, auth_service, audit_service

logger = logging.getLogger(__name__)

_EXAMPLE_QUESTIONS = [
    "Which machine type has the highest failure rate?",
    "Show engines with RUL under 50 cycles",
    "What is the average torque when a machine failure occurs?",
    "How many power failures happened across all CNC machines?",
    "Which sensor readings are highest before engine failure?",
]


def build(conv_id_state: gr.State, schema_context: str = "") -> None:
    """Build the Ask Your Data tab inside a gr.Blocks context."""

    gr.Markdown("## Ask Your Data")
    gr.Markdown(
        "Ask questions in plain English about any asset — CNC machines, engines, or "
        "electrical systems. Powered by **Genie** (primary) with **Bedrock fallback**.  \n"
        "_Responses are shown as tables and explanations — no raw code is returned._"
    )

    chatbot = gr.Chatbot(
        label="",
        height=400,
        type="messages",
        render_markdown=True,
        show_label=False,
    )

    results_table = gr.DataFrame(
        label="Query Results",
        height=280,
        visible=False,
    )

    with gr.Row():
        msg_box = gr.Textbox(
            placeholder="e.g. Which CNC machines have the most tool wear failures?",
            label="Your question",
            max_lines=3,
            scale=4,
        )
        submit_btn = gr.Button("Ask", variant="primary", scale=1)

    with gr.Row():
        clear_btn    = gr.Button("Clear Chat", variant="secondary")
        source_label = gr.Markdown("")

    gr.Markdown("**Example questions:**")
    with gr.Row():
        for q in _EXAMPLE_QUESTIONS[:3]:
            gr.Button(q, size="sm").click(
                fn=lambda question=q: question,
                outputs=[msg_box],
            )

    # ── Respond handler ────────────────────────────────────────────────────────
    def respond(message: str, history: list, conv_id: str | None,
                request: gr.Request) -> tuple:
        message = message.strip()
        if not message or len(message) > 2000:
            return history, message, conv_id, "", gr.update(visible=False)

        user  = auth_service.get_user_from_request(request)
        email = user["email"] if user else "unknown"
        role  = user["role"]  if user else "viewer"

        result = ai_service.chat_with_data(
            question=message,
            conversation_id=conv_id,
            schema_context=schema_context,
        )

        # Explanation text only — no SQL/code in chat window
        reply = result.text if not result.error else (
            "Sorry, I could not answer that question. Please try rephrasing it."
        )

        audit_service.log_event(
            action_type="CHAT",
            user_email=email, user_role=role,
            ai_source=result.source,
            source_tables="(AI-generated query)",
            query_text=result.sql or message,
            row_count=len(result.dataframe) if result.dataframe is not None else 0,
        )

        new_history = history + [
            {"role": "user",      "content": message},
            {"role": "assistant", "content": reply},
        ]

        source_md = f"_Answered by: **{result.source.capitalize()}**_"

        if result.dataframe is not None and len(result.dataframe) > 0:
            table_update = gr.update(value=result.dataframe, visible=True)
        else:
            table_update = gr.update(value=None, visible=False)

        return new_history, "", result.conversation_id, source_md, table_update

    submit_btn.click(
        fn=respond,
        inputs=[msg_box, chatbot, conv_id_state],
        outputs=[chatbot, msg_box, conv_id_state, source_label, results_table],
    )
    msg_box.submit(
        fn=respond,
        inputs=[msg_box, chatbot, conv_id_state],
        outputs=[chatbot, msg_box, conv_id_state, source_label, results_table],
    )
    clear_btn.click(
        fn=lambda: ([], "", None, "", gr.update(value=None, visible=False)),
        outputs=[chatbot, msg_box, conv_id_state, source_label, results_table],
    )
