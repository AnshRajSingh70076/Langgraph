from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableLambda
from typing import TypedDict
import sqlite3
import db  # Ensure DB setup is done
import os

# Set Google Gemini API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyA8-Q1tO01v3RN3OW_VXezySZ9EVxIN4Ho"

# LangChain + Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# Initialize DB
db.setup_db()

# Stronger prompt to enforce clean SQL output
prompt = ChatPromptTemplate.from_template(
    """
You are a SQL assistant. Convert the following natural language question into a **valid SQLite SELECT SQL query only**. 
Respond with the SQL query ONLY — no explanation, no formatting, no markdown. 
If it’s unrelated or impossible, return: `SELECT "Invalid or unsupported query"`.

Question: {messages}
SQL:
"""
)

# LLM setup
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
check_generated_query = prompt | model

# State format
class State(TypedDict):
    messages: str
    result: str

# Execute SQL query
def execute_query(query: str) -> str:
    try:
        conn = sqlite3.connect("mydb.db")
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()
        if rows:
            return "\n".join(str(row) for row in rows)
        else:
            return "No rows returned."
    except Exception as e:
        return f"Error executing query:\n{e}"

# Node logic
def generate_and_execute_query(state: State) -> State:
    response = check_generated_query.invoke({"messages": state["messages"]})
    generated_sql = response.content.strip()

    # Remove backticks, markdown, and extra prefix lines (if any)
    if "```" in generated_sql:
        generated_sql = generated_sql.split("```")[1].strip()
    if generated_sql.lower().startswith("sql"):
        generated_sql = generated_sql[3:].strip()

    # Execute only SELECT queries
    if not generated_sql.lower().startswith("select"):
        return {"messages": state["messages"], "result": f"Invalid SQL:\n{generated_sql}"}

    result = execute_query(generated_sql)
    return {"messages": state["messages"], "result": result}

# LangGraph wiring
workflow = StateGraph(State)
workflow.add_node("GenerateQuery", RunnableLambda(generate_and_execute_query))
workflow.set_entry_point("GenerateQuery")
workflow.set_finish_point("GenerateQuery")
app = workflow.compile()

# CLI loop
if __name__ == "__main__":
    while True:
        user_input = input("Ask a database query (or type 'exit'): ")
        if user_input.lower() == "exit":
            break
        output = app.invoke({"messages": user_input})
        print("\nResult:\n", output["result"])
