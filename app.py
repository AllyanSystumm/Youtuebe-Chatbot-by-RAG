import os
import streamlit as st
from groq import Groq
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

# --- 1. CONFIGURATION ---

# ⚠️ SECURITY NOTE: Never hardcode API keys in a production app.
# For local testing, we can use the environment variables or Streamlit secrets.
# You can replace the placeholders with your actual keys for local testing,
# but it's better to use os.environ.get() to load them.

# Replace with your actual keys for quick local testing if they are not in environment variables
GROQ_API_KEY = ""
# HUGGINGFACE_API_KEY is not strictly needed for the embeddings model
# but keep this in mind if you use an API-based embedding service.

# --- 2. CORE RAG FUNCTIONS (CACHED) ---

# Use st.cache_resource for objects that should be persisted and not re-run on every interaction
@st.cache_resource
def get_groq_client():
    """Initializes and returns the Groq client."""
    try:
        client = Groq(api_key=GROQ_API_KEY)
        return client
    except Exception as e:
        st.error(f"Error initializing Groq client: {e}")
        return None

@st.cache_resource(show_spinner="Processing transcript and creating knowledge base...")
def build_vector_store(video_id):
    """Fetches transcript, splits, and creates a FAISS vector store."""
    try:
        # 1. Fetch Transcript
        ytt_api = YouTubeTranscriptApi()
        fetched_transcript = ytt_api.fetch(video_id, languages=['en'])
        transcript_data = fetched_transcript.to_raw_data()
        full_transcript_text = " ".join([item['text'] for item in transcript_data])
        
        # 2. Split Text
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.create_documents([full_transcript_text])

        # 3. Create Embeddings and Vector Store
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(chunks, embeddings)
        
        return vector_store, f"Successfully processed {len(chunks)} chunks."

    except TranscriptsDisabled:
        return None, "Error: Transcripts are disabled for this video."
    except Exception as e:
        return None, f"An unexpected error occurred during processing: {e}"

def generate_rag_answer(client, retriever, question):
    """Performs RAG to get the final answer from Groq."""
    
    # 1. Retrieve Context
    retrieved_docs = retriever.invoke(question)
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

    # 2. Define Prompt Template (matching your existing prompt)
    prompt = PromptTemplate(
        template="""
You are a helpful assistant that answers questions based ONLY on the provided context from PDF documents.

CONTEXT:
{context}

INSTRUCTIONS:
1. Answer the question using ONLY the information from the context above
2. If the question is not related to the domain or topics covered in the context, respond with: "I am unable to respond because this question is irrelevant to my domain."
3. If the context does not contain specific information to answer a relevant question, respond with "I don't know"
4. Do not use any external knowledge or make assumptions

QUESTION: {question}

ANSWER:
        """,
        input_variables=['context', 'question']
    )
    final_prompt = prompt.invoke({"context": context_text, "question": question})

    # 3. Call Groq API
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": final_prompt.text}],
        max_tokens=1000
    )
    return response.choices[0].message.content

# --- 3. STREAMLIT APPLICATION LAYOUT ---

def main():
    st.set_page_config(page_title="⚡ Groq RAG Chatbot (YouTube Transcript)", layout="wide")
    st.title("⚡ YouTube Transcript RAG Chatbot")
    st.caption("Powered by Groq Llama-3.3 and FAISS Embeddings")

    # Initialize Groq client
    groq_client = get_groq_client()
    if not groq_client:
        return # Stop execution if client fails to initialize

    # Sidebar for Configuration
    with st.sidebar:
        st.subheader("Configuration")
        
        # The YouTube video ID you were using
        default_video_id = "7rWGv8f2B2Y" 
        
        video_url = st.text_input(
            "Enter YouTube Transcript ID :", 
            placeholder="e.g., 7rWGv8f2B2Y or full URL",
            value=default_video_id
        )
        
        st.markdown("**Note:** Once the ID is entered, the app automatically processes the transcript. If you change the ID, the processing will re-run.")
        st.divider()
        st.subheader("Instructions")
        st.markdown(
            """
            1. Enter a YouTube Video ID/URL.
            2. The app fetches the transcript and creates a RAG knowledge base.
            3. Ask questions about the video content in the chat box!
            """
        )

    # --- Main Chat Area Logic ---

    if 'messages' not in st.session_state:
        st.session_state.messages = []
        
    current_video_id = video_url.split("v=")[-1] if "v=" in video_url else video_url
    
    if not current_video_id:
        st.info("Please enter a valid YouTube Transcript ID or URL in the sidebar to begin.")
        return

    # 1. Build or retrieve the Vector Store
    vector_store, status_message = build_vector_store(current_video_id)
    
    if vector_store:
        st.success(status_message)
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # 2. Handle User Input
        if prompt := st.chat_input("Ask a question about the video..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking (Powered by Groq)..."):
                    # Call the RAG function
                    response = generate_rag_answer(groq_client, retriever, prompt)
                    st.markdown(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

    else:
        st.error(status_message)


if __name__ == "__main__":
    main()
