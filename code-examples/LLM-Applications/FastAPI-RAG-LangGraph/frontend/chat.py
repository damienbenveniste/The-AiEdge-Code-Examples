import streamlit as st
import requests

def get_response(user_input, user_id):
    """Get response from the chat API"""
    try:
        response = requests.post(
            "http://localhost:8000/chat/message",
            json={
                "message": user_input,
                "user_id": user_id
            }
        )
        
        if response.status_code == 200:
            return response.json()["response"]
        else:
            return f"Sorry, I encountered an error: {response.status_code}"
            
    except requests.exceptions.RequestException as e:
        return f"Connection error: {e}"

def chat_interface(chat_title="AI Assistant"):
    st.title(chat_title)

    # Add user ID input in sidebar
    user_id = st.sidebar.number_input("User ID", min_value=1, value=1)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What is your message?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Get response from API
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                full_response = get_response(prompt, user_id)
            st.markdown(full_response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

# This module is imported by main.py