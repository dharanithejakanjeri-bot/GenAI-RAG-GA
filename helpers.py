"""
utils/helpers.py — Formatting and display utilities
"""


def format_sources(sources: list[dict]) -> str:
    """Render retrieved source chunks as styled HTML for Streamlit."""
    if not sources:
        return ""

    html_parts = []
    for i, src in enumerate(sources, 1):
        score_pct = int(src.get("score", 0) * 100)
        excerpt = src.get("text", "")[:220].replace("\n", " ")
        doc_name = src.get("doc_name", "Unknown")
        html_parts.append(f"""
        <div class="source-card">
            <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                <strong style="color:#e8c547; font-size:12px;">📄 {doc_name}</strong>
                <span class="score-badge">relevance {score_pct}%</span>
            </div>
            <div style="color:#b0ada8; font-size:12px; font-style:italic;">
                "…{excerpt}…"
            </div>
        </div>
        """)

    return "\n".join(html_parts)


def truncate(text: str, max_chars: int = 300) -> str:
    """Truncate text for preview display."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "…"
