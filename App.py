import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from github_loader import load_github_repo

# Page configuration
st.set_page_config(
    page_title="GitHub Repo Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# Title
st.title("🤖 GitHub Repository Chatbot")
# Hide Streamlit menu and footer
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# GitHub repo URL (hardcoded in code)
github_repo = "https://github.com/Bhavana-Radhakrishna/chat-app.git"  # Change this URL to your desired repository

# Use default model
model_name = "llama3"

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "repo_loaded" not in st.session_state:
    st.session_state.repo_loaded = False

# Auto-load repository on first run
if not st.session_state.repo_loaded and github_repo:
    with st.spinner("Loading repository... This may take a few minutes."):
        try:
            # Load and process the GitHub repository
            st.info("Step 1/3: Cloning repository...")
            documents = load_github_repo(github_repo)
            
            if documents:
                st.info(f"Step 2/3: Processing {len(documents)} document chunks...")
                
                # Create embeddings using HuggingFace (much faster than Ollama)
                try:
                    embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2",
                    )
                    
                    st.info("Step 3/3: Creating vector database...")
                    # Create vector store
                    st.session_state.vectorstore = FAISS.from_documents(
                        documents, 
                        embeddings
                    )
                    
                    st.session_state.repo_loaded = True
                    st.success(f"✅ Successfully loaded {len(documents)} documents from the repository!")
                    
                except Exception as ollama_error:
                    st.error(f"Error: {str(ollama_error)}")
                    st.error("Make sure required packages are installed.")
                    
            else:
                st.error("No documents found in the repository.")
                
        except Exception as e:
            st.error(f"Error loading repository: {str(e)}")

# Chat interface
if st.session_state.repo_loaded:
    st.markdown("### Chat with the Repository")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the repository..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                # Create retriever
                retriever = st.session_state.vectorstore.as_retriever(
                    search_kwargs={"k": 3}  # Reduced from 5 to 3 for faster responses
                )
                
                # Get relevant context
                relevant_docs = retriever.invoke(prompt)
                context = "\n\n".join([doc.page_content for doc in relevant_docs])
                
                # Create prompt
                full_prompt = f"""You are a helpful assistant that answers questions about a GitHub repository.
Use the following context from the repository to answer the question. If you don't know the answer,
just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {prompt}

Answer:"""
                
                # Initialize Ollama LLM with streaming
                llm = OllamaLLM(model=model_name, temperature=0.7)
                
                # Stream the response
                for chunk in llm.stream(full_prompt):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
                
                # Add assistant message to chat history
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

else:
    st.info("Please enter a GitHub repository URL and click 'Load Repository' to start chatting.")
    
    # Example section
    st.markdown("### Example Usage")
    st.markdown("""
    1. Enter a GitHub repository URL (e.g., `https://github.com/python/cpython`)
    2. Click 'Load Repository' to index the repository
    3. Ask questions about the code, documentation, or structure
    
    **Example Questions:**
    - What is this repository about?
    - How do I install this project?
    - What are the main features?
    - Explain the architecture of this project
    - Show me how to contribute to this project
    """)
