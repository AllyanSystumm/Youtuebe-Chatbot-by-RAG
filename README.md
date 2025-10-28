YouTube Transcript RAG Chatbot 

A chatbot powered by Groq Llama-3.3 and FAISS embeddings that allows users to interact with YouTube video transcripts using Retrieve-and-Generate (RAG). This app automatically fetches the transcript of a YouTube video, processes it into a knowledge base, and answers user questions based on the content of the video.

Features:

Automatic Transcript Processing: Fetches YouTube transcript for the provided video ID.

Embedding & Vector Store: Uses FAISS and HuggingFace embeddings to convert transcript text into vector form for efficient retrieval.

Retrieve-and-Generate (RAG): Combines context retrieval with Groq’s Llama-3.3 model to answer user questions.

Interactive Chat Interface: Users can ask questions and get responses based on the video’s transcript.

