# RAG-Based Application Using Gemini Model

 Overview
This project implements a Retrieval-Augmented Generation (RAG) application using Google Gemini for question-answering tasks. The system allows users to upload PDF documents, processes them into vector embeddings, and enables question-answering using a retrieval + generative model approach.

Features
PDF Upload & Processing: Extracts text from uploaded PDFs.
 Text Chunking: Splits documents into manageable chunks.
 Vector Storage: Embeds and stores document chunks using Google Gemini embeddings.
 Retrieval & Question Answering: Fetches relevant document chunks and generates concise answers.
 Chat Interface: Displays chat history for better user interaction.

Tech Stack
Python
Streamlit (for UI)
LangChain (for document processing, retrieval, and generation)
ChromaDB (for vector storage)
Google Gemini (for embeddings & question-answering)
