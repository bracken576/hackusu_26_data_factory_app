"""
AI/LLM integration — 3-tier architecture:
  Tier 1: Genie API (Unity Catalog native, primary)
  Tier 2: Bedrock / Claude Text-to-SQL (fallback)
  Tier 3: Mock (local dev, USE_MOCK_AI=true)
"""
import os
import re
import time
import logging
import requests
import sqlparse
from dataclasses import dataclass

logger = logging.getLogger(__name__)

DATABRICKS_HOST  = os.getenv("DATABRICKS_HOST", "")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN", "")
GENIE_SPACE_ID   = os.getenv("GENIE_SPACE_ID", "")
BEDROCK_REGION   = os.getenv("BEDROCK_REGION", "us-east-1")
USE_MOCK         = os.getenv("USE_MOCK_AI", "false").lower() == "true"

GENIE_HEADERS = {
    "Authorization": f"Bearer {DATABRICKS_TOKEN}",
    "Content-Type": "application/json",
}

BLOCKED_SQL_KEYWORDS = {
    "INSERT", "UPDATE", "DELETE", "DROP", "TRUNCATE",
    "ALTER", "CREATE", "EXEC", "EXECUTE", "GRANT", "REVOKE",
}
MAX_ROW_LIMIT = 10_000


@dataclass
class ChatResponse:
    text: str                               # plain-English explanation shown in chat
    dataframe: "pd.DataFrame | None" = None # query results rendered as table in UI
    sql: str | None = None                  # generated SQL — audit log ONLY, never shown in chat
    conversation_id: str | None = None      # Genie conversation ID for multi-turn
    source: str = "genie"                   # "genie" | "bedrock" | "mock"
    error: str | None = None                # set only when all tiers fail


# ── PUBLIC ENTRY POINT ────────────────────────────────────────────────────────
def chat_with_data(
    question: str,
    conversation_id: str | None = None,
    schema_context: str = "",
) -> ChatResponse:
    """
    Single entry point for all conversational queries.
    Tries Genie first; falls back to Bedrock Text-to-SQL; falls back to mock.
    """
    if USE_MOCK:
        return _mock_response(question)

    # Tier 1: Genie
    if GENIE_SPACE_ID:
        try:
            if conversation_id:
                return _continue_genie(conversation_id, question)
            else:
                return _start_genie(question)
        except Exception as exc:
            logger.warning("Genie failed (%s), falling back to Bedrock: %s", type(exc).__name__, exc)

    # Tier 2: Bedrock
    try:
        return _bedrock_text_to_sql(question, schema_context)
    except Exception as exc:
        logger.error("Bedrock fallback also failed: %s", exc)
        return ChatResponse(
            text="I was unable to answer your question. Please try rephrasing it.",
            error=str(exc),
            source="error",
        )


# ── Tier 1: Genie ─────────────────────────────────────────────────────────────
def _start_genie(question: str) -> ChatResponse:
    url = f"{DATABRICKS_HOST}/api/2.0/genie/spaces/{GENIE_SPACE_ID}/start-conversation"
    resp = requests.post(url, headers=GENIE_HEADERS, json={"content": question}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return _poll_genie(data["conversation_id"], data["message_id"])


def _continue_genie(conversation_id: str, question: str) -> ChatResponse:
    url = (f"{DATABRICKS_HOST}/api/2.0/genie/spaces/{GENIE_SPACE_ID}"
           f"/conversations/{conversation_id}/messages")
    resp = requests.post(url, headers=GENIE_HEADERS, json={"content": question}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return _poll_genie(conversation_id, data["message_id"])


def _poll_genie(conversation_id: str, message_id: str, max_wait: int = 60) -> ChatResponse:
    url = (f"{DATABRICKS_HOST}/api/2.0/genie/spaces/{GENIE_SPACE_ID}"
           f"/conversations/{conversation_id}/messages/{message_id}")
    wait, elapsed = 1, 0
    while elapsed < max_wait:
        data = requests.get(url, headers=GENIE_HEADERS, timeout=10).json()
        status = data.get("status")
        if status == "COMPLETED":
            text, sql = "", None
            for att in data.get("attachments", []):
                if att.get("type") == "text":
                    text = att.get("content", "")
                elif att.get("type") == "query":
                    sql = att.get("query", {}).get("query")
            df = None
            if sql:
                try:
                    from services import db_service
                    df = db_service._sql_query(validate_sql(sql))
                except Exception as exc:
                    logger.warning("Could not execute Genie SQL for DataFrame: %s", exc)
            return ChatResponse(
                text=text or "Here are your results.",
                dataframe=df,
                sql=sql,
                conversation_id=conversation_id,
                source="genie",
            )
        elif status == "FAILED":
            raise RuntimeError(f"Genie FAILED: {data.get('error', 'unknown')}")
        elif status == "UNABLE_TO_ANSWER":
            raise RuntimeError("Genie could not answer this question.")
        time.sleep(wait)
        elapsed += wait
        wait = min(wait * 2, 5)
    raise TimeoutError("Genie timed out.")


# ── Tier 2: Bedrock Text-to-SQL ───────────────────────────────────────────────
_BEDROCK_AGENT = None


def _get_agent():
    global _BEDROCK_AGENT
    if _BEDROCK_AGENT is None:
        from strands import Agent
        from strands.models import BedrockModel
        _BEDROCK_AGENT = Agent(model=BedrockModel(
            model_id="global.anthropic.claude-sonnet-4-6",
            region_name=BEDROCK_REGION,
        ))
    return _BEDROCK_AGENT


_TEXT_TO_SQL_PROMPT = """\
You are a SQL expert for Databricks SQL Warehouse analyzing predictive maintenance data.
Convert the user question to a single valid SELECT statement.

STRICT RULES:
- Output ONLY the SQL statement, nothing else — no explanation, no markdown
- Only SELECT is allowed — never INSERT, UPDATE, DELETE, DROP, ALTER, or any DDL
- Always use fully qualified table names: catalog.schema.table
- Always include LIMIT {limit} unless the query already has a LIMIT
- If the question cannot be answered with the available schema, output exactly:
  UNABLE_TO_ANSWER: <one sentence explanation>

Available schema:
{schema_context}

User question: {question}"""


def _bedrock_text_to_sql(question: str, schema_context: str) -> ChatResponse:
    from services import db_service
    agent = _get_agent()
    raw_sql = str(agent(_TEXT_TO_SQL_PROMPT.format(
        question=question,
        schema_context=schema_context or "(no schema — use best judgement)",
        limit=MAX_ROW_LIMIT,
    ))).strip()

    if raw_sql.startswith("UNABLE_TO_ANSWER"):
        return ChatResponse(
            text=f"I couldn't find data to answer that. {raw_sql.split(':', 1)[-1].strip()}",
            source="bedrock",
        )

    safe_sql = validate_sql(raw_sql)
    df = db_service._sql_query(safe_sql)

    summary = f"Found {len(df):,} row(s)."
    if len(df) > 0:
        try:
            summary = _generate_insight(df.head(10).to_string(index=False), question)
        except Exception:
            pass

    return ChatResponse(
        text=summary,
        dataframe=df,
        sql=safe_sql,
        conversation_id=None,
        source="bedrock",
    )


def _generate_insight(data_description: str, context: str) -> str:
    agent = _get_agent()
    prompt = (
        f"A maintenance engineer asked: '{context}'\n"
        f"Query results (sample):\n{data_description}\n\n"
        "Write 2-3 bullet-point insights for a non-technical operations manager. "
        "Be specific with numbers. Do not mention SQL or technical terms."
    )
    return str(agent(prompt))


# ── SQL Validator ─────────────────────────────────────────────────────────────
def validate_sql(sql: str) -> str:
    """Validate and sanitize LLM-generated SQL before execution. Raises ValueError on failure."""
    sql = sql.strip().rstrip(";")
    if ";" in sql:
        raise ValueError("Multi-statement SQL is not allowed.")
    parsed = sqlparse.parse(sql)
    if not parsed:
        raise ValueError("Could not parse the generated SQL.")
    if parsed[0].get_type() != "SELECT":
        raise ValueError("Only SELECT queries are allowed.")
    upper = sql.upper()
    for kw in BLOCKED_SQL_KEYWORDS:
        if re.search(rf"\b{kw}\b", upper):
            raise ValueError(f"Blocked keyword detected: {kw}")
    if not re.search(r"\bLIMIT\b", upper):
        sql = f"{sql} LIMIT {MAX_ROW_LIMIT}"
    return sql


# ── Tier 3: Mock ──────────────────────────────────────────────────────────────
def _mock_response(question: str) -> ChatResponse:
    import pandas as pd
    mock_df = pd.DataFrame({
        "failure_mode": ["Tool Wear Failure", "Heat Dissipation", "Power Failure", "Overstrain", "Random"],
        "count": [45, 112, 95, 78, 8],
        "pct_of_total": [13.3, 33.0, 28.0, 23.0, 2.4],
    })
    return ChatResponse(
        text=(
            f"[MOCK] Simulated response for: '{question}'\n\n"
            "• Heat Dissipation Failure is the most common failure mode at 33% of incidents.\n"
            "• Power Failure accounts for 28% — consider reviewing power supply stability.\n"
            "• Tool Wear Failure at 13% suggests scheduled tool replacement intervals may need adjustment."
        ),
        dataframe=mock_df,
        sql="SELECT failure_mode, COUNT(*) AS count FROM main.predictive_maintenance.cnc_machine_data GROUP BY 1 LIMIT 10",
        conversation_id="mock-conv-001",
        source="mock",
    )
