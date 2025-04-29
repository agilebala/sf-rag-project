
import os
import numpy as np
import streamlit as st
from dotenv import load_dotenv

# Import retrieval and generation functions
from scripts.retrieval import retrieve_top_k
from utils.embedding import generate_answer

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(page_title="Salesforce Earnings RAG", layout="wide")
st.title("📊 Salesforce Earnings Call Q&A Assistant")

# Input from user
query = st.text_input("Ask a question or request a summary:")

if st.button("Submit") and query:
    with st.spinner("🔍 Retrieving relevant documents..."):
        context_chunks = retrieve_top_k(query)

    if context_chunks:
        with st.spinner("🧠 Generating answer..."):
            try:
                answer = generate_answer(query, context_chunks)
                st.markdown("### ✅ Answer")
                st.write(answer)

                st.markdown("### 📄 Context Used")
                for chunk in context_chunks:
                    st.markdown(f"**Source:** {chunk['source']}")
                    st.write(chunk["text"][:500] + "...")
                    st.markdown("---")
            except Exception as e:
                st.error(f"Error generating answer: {e}")
    else:
        st.warning("⚠️ No relevant documents found for your query.")

# Optional examples to guide users
st.markdown("---")
st.subheader("💡 Example Questions")
st.markdown("- What were the key financial highlights?")
st.markdown("- Summarize the main strategic initiatives discussed.")
st.markdown("- What risks were mentioned in the latest earnings call?")

