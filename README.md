# YouTube Transcript RAG Chatbot

## Overview

The **YouTube Transcript RAG Chatbot** is an interactive chatbot powered by **Groq Llama-3.3** and **FAISS embeddings**. It allows users to interact with YouTube video transcripts using **Retrieve-and-Generate (RAG)**. The app fetches a YouTube video’s transcript, processes it into a knowledge base, and enables users to ask questions based on the content of the video.

This tool is ideal for anyone looking to explore and query YouTube videos in-depth through a conversational interface.

## Features

- **Automatic Transcript Processing**:  
  The app automatically fetches the transcript of a YouTube video using the provided video ID.

- **Embedding & Vector Store**:  
  Converts the transcript text into vector form using **FAISS** and **HuggingFace embeddings** for efficient retrieval of relevant information.

- **Retrieve-and-Generate (RAG)**:  
  Combines **context retrieval** with **Groq’s Llama-3.3** model to generate accurate answers to user queries based on the video’s transcript.

- **Interactive Chat Interface**:  
  Allows users to ask questions, and receive responses directly from the video’s transcript content, enabling a deeper understanding of the video material.
  <img width="1840" height="1010" alt="CHATBOT CHATBOT" src="https://github.com/user-attachments/assets/2a58f1ac-5f68-4bc0-8292-ca89cbac946f" />


## Prerequisites

Before you start, ensure you have the following:

- A **YouTube video ID** for fetching the transcript.
- API keys for **Groq** and **HuggingFace** (stored in a `.env` file).
