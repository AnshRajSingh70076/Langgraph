import os
import sqlite3
import streamlit as st
import pandas as pd
from datetime import datetime
from typing import TypedDict, Optional

from langchain_core.runnables import RunnableConfig
from langchain_core.documents import Document
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph

# === Set API Key ===
os.environ["GROQ_API_KEY"] = "gsk_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"

# === Paths ===
db_path = r"C:\\Users\\rajs1\\Downloads\\newdata.db\\ansh.db"
chat_db_path = "chat_history.db"

# === Chat History DB ===
def init_db():
    conn = sqlite3.connect(chat_db_path)
    conn.execute('''CREATE TABLE IF NOT EXISTS ChatHistory (
        user_id TEXT,
        timestamp TEXT,
        question TEXT,
        answer TEXT
    );''')
    conn.commit()
    conn.close()

def save_chat(user_id, question, answer):
    conn = sqlite3.connect(chat_db_path)
    conn.execute("INSERT INTO ChatHistory (user_id, timestamp, question, answer) VALUES (?, ?, ?, ?)",
                 (user_id, datetime.utcnow().isoformat(), question, answer))
    conn.commit()
    conn.close()

def get_recent_history(user_id, limit=5):
    conn = sqlite3.connect(chat_db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT question, answer FROM ChatHistory WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?", (user_id, limit))
    rows = cursor.fetchall()
    conn.close()
    return list(reversed(rows))

# === PDF Utilities ===
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

def get_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    vectorstore.save_local("faiss_index")

def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# === SQL Utilities ===
def read_sql_query(sql, db):
    try:
        conn = sqlite3.connect(db)
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        conn.close()
        return rows
    except Exception as e:
        return [(f"SQL Error: {e}",)]

def get_disease_list():
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT DISTINCT Disease FROM Treatment", conn)
    conn.close()
    return df["Disease"].str.lower().tolist()

def match_disease(question):
    question_lower = question.lower()
    for disease in get_disease_list():
        if disease in question_lower:
            return disease
    return None

def query_sql(question):
    disease = match_disease(question)
    if disease:
        sql = f"SELECT treat FROM Treatment WHERE LOWER(Disease) = '{disease}'"
        results = read_sql_query(sql, db_path)
        if results and not results[0][0].startswith("SQL Error"):
            return f"Based on the database, treatment for {disease.title()}: {results[0][0]}"
    return ""

# === LangGraph State and Nodes ===
class QAState(TypedDict):
    question: str
    pdf_answer: Optional[str]
    sql_answer: Optional[str]
    final_answer: Optional[str]

def pdf_node(state: QAState, config: RunnableConfig) -> QAState:
    vectorstore = load_vector_store()
    docs = vectorstore.similarity_search(state["question"], k=2)
    return {
        **state,
        "pdf_answer": "\n\n".join([d.page_content if isinstance(d, Document) else str(d) for d in docs]) if docs else ""
    }

def sql_node(state: QAState) -> QAState:
    sql_response = query_sql(state["question"])
    return {**state, "sql_answer": sql_response}

def final_node(state: QAState) -> QAState:
    llm = ChatGroq(
        groq_api_key=os.environ["GROQ_API_KEY"],
        model_name="llama3-8b-8192",
        temperature=0.3
    )

    history = get_recent_history("guest")
    memory_context = "\n".join([f"User: {q}\nAssistant: {a}" for q, a in history])

    context = f"PDF Info:\n{state['pdf_answer']}\n\nDB Info:\n{state['sql_answer']}"
    prompt = PromptTemplate(
        template="""
You are a helpful assistant.

Here is the previous conversation:
{memory_context}

Answer the question as detailed as possible from the provided context.
If the answer is not in the context, say "answer is not available in the context".

Context: {context}
Question: {question}

Answer:
""",
        input_variables=["memory_context", "question", "context"]
    )

    chain = create_stuff_documents_chain(llm, prompt)

    result = chain.invoke({
        "memory_context": memory_context,
        "question": state["question"],
        "context": [Document(page_content=context)]
    })

    return {**state, "final_answer": result.strip()}



def create_graph():
    builder = StateGraph(QAState)
    builder.add_node("pdf", pdf_node)
    builder.add_node("sql", sql_node)
    builder.add_node("final", final_node)

    builder.set_entry_point("pdf")
    builder.add_edge("pdf", "sql")
    builder.add_edge("sql", "final")
    builder.set_finish_point("final")
    return builder.compile()

# === Streamlit App ===
def handle_user_input(user_id, question):
    graph = create_graph()
    state = graph.invoke({"question": question})

    answer = state.get("final_answer", "Sorry, I couldn’t find relevant information.")
    save_chat(user_id, question, answer)

    st.write("💬 **Reply:**")
    st.success(answer)

def main():
    st.set_page_config(page_title="Ayushveda: PDF + DB Chat with LangGraph", layout="centered")
    st.title("🩺 Ayushveda – Chat with PDFs + Medical DB (LangGraph)")
    st.markdown("Ask about your uploaded PDFs or known diseases.")

    init_db()

    user_id = st.text_input("👤 Enter your name or ID:", value="guest")
    user_question = st.text_input("❓ Ask your question:")

    if user_question and user_id:
        handle_user_input(user_id, user_question)

    with st.sidebar:
        st.title("📚 Upload PDFs")
        pdf_docs = st.file_uploader("Upload one or more PDF files", accept_multiple_files=True)
        if st.button("🔍 Process PDFs"):
            with st.spinner("Extracting and embedding..."):
                raw_text = get_pdf_text(pdf_docs)
                chunks = get_text_chunks(raw_text)
                get_vector_store(chunks)
                st.success("PDFs processed and indexed!")

if __name__ == "__main__":
    main()
