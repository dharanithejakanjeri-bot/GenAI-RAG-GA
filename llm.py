"""
utils/llm.py — LLM query with RAG context + chat memory
"""

import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """You are a precise, helpful document Q&A assistant.
Answer questions ONLY using the provided document context.
If the answer is not in the context, say "I couldn't find that in the provided documents."
Be concise and cite which document your answer comes from."""


def _build_context_block(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(
            f"[Source {i} — {c['doc_name']} | score {c['score']:.2f}]\n{c['text']}"
        )
    return "\n\n---\n\n".join(parts)


def ask_llm(
    question: str,
    context_chunks: list[dict],
    history: list[dict] | None = None,
    model: str = "gpt-4o-mini",
) -> tuple[str, list[dict]]:
    """
    Send question + RAG context + chat history to OpenAI.

    Returns:
        answer (str)       — the model's response
        sources (list)     — the context chunks used (for citation display)
    """
    context_block = _build_context_block(context_chunks)

    # Build message list
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    # Inject prior conversation (trimmed to last 6 turns for token safety)
    if history:
        for msg in history[-6:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

    # Final user message with context
    messages.append({
        "role": "user",
        "content": (
            f"DOCUMENT CONTEXT:\n{context_block}\n\n"
            f"QUESTION: {question}"
        ),
    })

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
        max_tokens=800,
    )

    answer = response.choices[0].message.content.strip()
    return answer, context_chunks
