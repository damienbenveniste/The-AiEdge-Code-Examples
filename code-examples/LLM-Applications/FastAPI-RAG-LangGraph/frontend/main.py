"""
Application Frontend - Main Streamlit Interface

This module provides a unified web interface for the AI assistant system
with two main functionalities:
1. Chat Interface - Interactive conversation with the RAG-powered AI assistant
2. Indexing Interface - Administrative tool for adding content to the knowledge base

The app uses Streamlit's sidebar navigation to switch between these modes,
providing a complete end-to-end user experience for both content consumption
and content management.
"""

import streamlit as st
import requests
from chat import chat_interface

# Configure the Streamlit page
st.set_page_config(
    page_title="AI Assistant", 
    layout="wide",
    page_icon="🤖"
)

# Sidebar navigation
st.sidebar.title("🤖 AI Assistant")
st.sidebar.markdown("---")
page = st.sidebar.selectbox(
    "Choose a page", 
    ["💬 Chat", "📚 Indexing"],
    help="Select between chatting with the AI or managing the knowledge base"
)

if page == "💬 Chat":
    # Launch the conversational AI interface
    chat_interface()
    
elif page == "📚 Indexing":
    st.title("📚 Knowledge Base Management")
    st.markdown("Add new content to the AI's knowledge base for enhanced responses.")
    
    # Input form for indexing new content
    with st.form("indexing_form"):
        url = st.text_input(
            "Enter URL to index:",
            placeholder="https://example.com",
            help="URL of the website content to add to the knowledge base"
        )
        user = st.text_input(
            "Enter user:",
            placeholder="admin",
            help="User identifier for tracking indexing operations"
        )
        
        submitted = st.form_submit_button("🔄 Index Content")
        
        if submitted:
            if url and user:
                # Show progress indicator
                with st.spinner("Processing and indexing content..."):
                    try:
                        response = requests.post(
                            "http://localhost:8000/indexing/index",
                            json={"url": url, "user": user}
                        )
                        
                        if response.status_code == 200:
                            st.success("✅ " + response.json()["message"])
                            st.balloons()  # Celebration animation
                        else:
                            st.error(f"❌ Server error: {response.status_code}")
                            
                    except requests.exceptions.RequestException as e:
                        st.error(f"🔌 Connection error: {e}")
                        st.info("💡 Make sure the backend server is running on localhost:8000")
            else:
                st.warning("⚠️ Please enter both URL and user to proceed")

    # Add some helpful information
    with st.expander("ℹ️ How does indexing work?"):
        st.markdown("""
        **The indexing process involves several steps:**
        
        1. **Document Parsing** - Extract text content from the provided URL
        2. **AI Summarization** - Generate concise summaries using GPT models
        3. **Vector Embedding** - Create semantic embeddings for search
        4. **Database Storage** - Store in ChromaDB for fast retrieval
        
        Once indexed, this content becomes available to the AI assistant 
        for generating more accurate and contextual responses.
        """)