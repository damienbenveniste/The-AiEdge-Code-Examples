import streamlit as st
from langserve.client import RemoteRunnable

def get_response(user_input, url, username):
    response_placeholder = st.empty()
    full_response = ""
    chain = RemoteRunnable(url)
    stream = chain.stream(input={'question': user_input, 'username': username})
    for chunk in stream:
        full_response += chunk
        response_placeholder.markdown(full_response)

    return full_response

def chat_interface(chat_title, page_hash ,url):
    st.title(chat_title)

    # Add username input at the top of the page
    username = st.text_input("Enter your username:", key="username_input", value="Guest")

    # Initialize page-specific chat history
    if "chat_histories" not in st.session_state:
        st.session_state.chat_histories = {}
    
    if page_hash not in st.session_state.chat_histories:
        st.session_state.chat_histories[page_hash] = []

    # Display chat messages from history for the current page
    for message in st.session_state.chat_histories[page_hash]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What is your message?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.chat_histories[page_hash].append({"role": "user", "content": prompt})

        # Get streaming response
        with st.chat_message("assistant"):
            full_response = get_response(prompt, url, username)

        # Add assistant response to chat history
        st.session_state.chat_histories[page_hash].append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    chat_interface()


st.title('RAG Chat App')

# Add username input at the top of the page
username = st.text_input(
    "Enter your username:", 
    key="username_input", 
    value="Guest"
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is your message?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({
        "role": "user", "content": prompt
    })

    # Get streaming response
    with st.chat_message("assistant"):
        full_response = get_response(prompt, url, username)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})


def get_response(user_input, url, username):
    response_placeholder = st.empty()
    full_response = ""
    chain = RemoteRunnable(url)
    stream = chain.stream(
        input={'question': user_input, 'username': username}
    )
    for chunk in stream:
        full_response += chunk
        response_placeholder.markdown(full_response)

    return full_response