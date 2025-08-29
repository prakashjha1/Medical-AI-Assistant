import streamlit as st
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
from src.prompts import system_prompt
import os

# Page configuration
st.set_page_config(
    page_title="Medical Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize the RAG system (cached to avoid reloading on every interaction)
@st.cache_resource
def initialize_rag_system():
    """Initialize and cache the RAG system components"""
    try:
        # Get API keys
        PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        
        if not PINECONE_API_KEY or not GOOGLE_API_KEY:
            st.error("Missing API keys. Please check your environment variables.")
            return None
        
        os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
        os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
        
        # Initialize embeddings
        embeddings = download_embeddings()
        
        # Initialize vector store
        index_name = "medical-assistant"
        docsearch = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embeddings
        )
        
        # Create retriever
        retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        
        # Initialize chat model
        chat_model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",  # or "gemini-2.5-pro", or other model IDs
            temperature=0.7,
            max_tokens=None,
            timeout=None,
            max_retries=2
            )
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        # Create chains
        # Initialize memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )

        # Updated prompt template with chat history
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(chat_model, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        return rag_chain, memory
        
        
    except Exception as e:
        st.error(f"Error initializing RAG system: {str(e)}")
        return None

# Main app
def main():
    # Header
    st.title("🏥 Medical Assistant")
    st.markdown("Ask me any medical questions and I'll provide helpful information based on medical knowledge.")
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.info(
            "This medical chatbot uses RAG (Retrieval Augmented Generation) "
            "to provide accurate medical information from a curated knowledge base."
        )
        
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.session_state.memory.clear()  # Clear LangChain memory
            st.rerun()
    
    # Initialize RAG system
    rag_chain, memory = initialize_rag_system()
    
    if rag_chain is None:
        st.error("Failed to initialize the medical chatbot. Please check your configuration.")
        return
    
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if "memory" not in st.session_state:
        st.session_state.memory = memory
    
    # Accept user input
    if prompt := st.chat_input("Ask your medical question..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Get response from RAG chain
                    response = rag_chain.invoke({
                        "input": prompt,
                        "chat_history": st.session_state.memory.chat_memory.messages
                    })
                    answer = response["answer"]
                    # After getting the response, add to memory:
                    st.session_state.memory.chat_memory.add_user_message(prompt)
                    st.session_state.memory.chat_memory.add_ai_message(answer)
                    
                    
                    # Display the response
                    st.markdown(answer)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                    # Optional: Display source documents in an expander
                    if "context" in response and response["context"]:
                        with st.expander("📚 Source Documents"):
                            for i, doc in enumerate(response["context"]):
                                st.markdown(f"**Source {i+1}:**")
                                st.markdown(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                                st.markdown("---")
                
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()