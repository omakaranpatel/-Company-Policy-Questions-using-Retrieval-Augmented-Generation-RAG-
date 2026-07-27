import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import src.utils  # noqa: F401 - Must be imported FIRST to patch sys.stderr.flush for Windows Streamlit threads
import src.config  # noqa: F401 - Loads .env variables into environment

import streamlit as st

from src.pipeline import PolicyRAGPipeline

st.set_page_config(
    page_title="Policy Assistant",
    page_icon="📋",
    layout="wide",
)

POLICY_FILTERS = {
    "All policies": None,
    "Expense Policy": "expense",
    "Travel Policy": "travel",
    "Finance Policy": "finance",
    "Employee Handbook": "handbook",
}


@st.cache_resource
def get_pipeline() -> PolicyRAGPipeline:
    return PolicyRAGPipeline()


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_result" not in st.session_state:
        st.session_state.last_result = None


def main() -> None:
    init_session_state()
    pipeline = get_pipeline()

    st.title("Employee Policy & Reimbursement Assistant")
    st.caption("Answers are grounded in company policy documents with citations.")

    with st.sidebar:
        st.header("Settings")
        policy_label = st.selectbox("Filter by policy", list(POLICY_FILTERS.keys()))
        policy_filter = POLICY_FILTERS[policy_label]
        use_expansion = st.toggle("Query expansion", value=True)
        use_reranking = st.toggle("Reranking", value=True)
        streaming = st.toggle("Streaming responses", value=True)

        st.divider()
        if pipeline.is_indexed():
            st.success(f"Index ready ({pipeline.vector_store.count()} chunks)")
        else:
            st.error("Index not found. Run: python scripts/index_documents.py")
            if st.button("Build index now"):
                with st.spinner("Indexing documents..."):
                    count = pipeline.index_documents(Path(__file__).parent / "documents")
                    st.success(f"Indexed {count} chunks")
                    st.rerun()

        st.divider()
        st.markdown("**Sample questions**")
        samples = [
            "What is the hotel cap for domestic travel?",
            "How long do I have to submit expense reports?",
            "What is the wellness stipend amount?",
            "Can I get reimbursed for a gym membership?",
            "What approval is needed for expenses over $5,000?",
        ]
        for q in samples:
            if st.button(q, key=f"sample_{q[:20]}"):
                st.session_state.pending_question = q

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("citations"):
                with st.expander("Citations"):
                    for cite in message["citations"]:
                        st.markdown(f"**{cite['document_title']} — {cite['section']}**")
                        st.caption(cite["document_id"])
                        st.text(cite["excerpt"])

    prompt = st.chat_input("Ask about reimbursement or policy...")
    if "pending_question" in st.session_state:
        prompt = st.session_state.pop("pending_question")

    if prompt:
        if not pipeline.is_indexed():
            st.error("Please build the index first.")
            return

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[:-1]
        ]

        with st.chat_message("assistant"):
            if streaming and pipeline.generator.use_openai:
                stream_gen, hits = pipeline.stream_ask(
                    question=prompt,
                    history=history,
                    policy_filter=policy_filter,
                    use_query_expansion=use_expansion,
                    use_reranking=use_reranking,
                )
                placeholder = st.empty()
                full_answer = ""
                for token in stream_gen:
                    full_answer += token
                    placeholder.markdown(full_answer)
                result = pipeline.generator.generate(prompt, hits, history)
                result.answer = full_answer
            else:
                with st.spinner("Searching policies..."):
                    result = pipeline.ask(
                        question=prompt,
                        history=history,
                        policy_filter=policy_filter,
                        use_query_expansion=use_expansion,
                        use_reranking=use_reranking,
                    )
                st.markdown(result.answer)

            citation_dicts = [
                {
                    "document_title": c.document_title,
                    "section": c.section,
                    "document_id": c.document_id,
                    "excerpt": c.excerpt,
                }
                for c in result.citations
            ]
            if citation_dicts:
                with st.expander("Citations"):
                    for cite in citation_dicts:
                        st.markdown(f"**{cite['document_title']} — {cite['section']}**")
                        st.caption(cite["document_id"])
                        st.text(cite["excerpt"])

            if result.insufficient_info:
                st.warning("Insufficient information found in policy documents.")

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result.answer,
                "citations": citation_dicts,
            }
        )
        st.session_state.last_result = {"question": prompt, "answer": result.answer}

        with st.expander("Rate this answer"):
            rating = st.slider("Helpfulness", 1, 5, 3)
            comment = st.text_input("Optional comment")
            if st.button("Submit feedback"):
                pipeline.record_feedback(prompt, result.answer, rating, comment)
                st.success("Thank you for your feedback!")


if __name__ == "__main__":
    main()
