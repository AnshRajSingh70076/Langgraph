##This README is for chatbot.py
This project is an intelligent chatbot that answers medical questions using both structured database info and unstructured PDF content. It is built using LangGraph, LangChain, and Groq’s LLaMA 3 model.

🔧 Features
📄 Upload PDFs, extract content, and embed them using FAISS

🧬 Query a SQLite database of treatments

🧠 Final answer generation using Groq's LLaMA3-8B via ChatGroq

🔁 Maintains recent chat history per user

🕸️ Uses LangGraph to orchestrate PDF, SQL, and Final Answering nodes

🧰 Tech Stack
Feature	Tech Used
LLM	Groq LLaMA 3 8B via langchain_groq
Document Embeddings	HuggingFace all-MiniLM-L6-v2
Vector Search	FAISS
DB	SQLite (ansh.db)
Chat History	SQLite (chat_history.db)
Document Parser	PyPDF2
LLM Orchestration	LangGraph
UI (optional)	Streamlit
