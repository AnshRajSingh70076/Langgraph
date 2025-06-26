from typing import Annotated, TypedDict
from langgraph.graph import StateGraph,START,END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

import os

os.environ["GOOGLE_API_KEY"] = "-Q1tO01*******88888888888OW"     

class GraphState(TypedDict):
    msg: Annotated[list[str], "Conversation history"]


llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")


def call_llm(state: GraphState) -> GraphState:
    message = state["msg"][-1]  
    response = llm.invoke([HumanMessage(content=message)])
    return {"msg": state["msg"] + [response.content]}


graph_builder = StateGraph(GraphState)
graph_builder.add_node("llm", call_llm)
graph_builder.set_entry_point("llm")
graph_builder.set_finish_point("llm")


from IPython.display import display
import matplotlib.pyplot as plt

fig = graph_builder.get_graph().draw()
display(fig)
plt.show()

graph = graph_builder.compile()


input_state = {"msg": ["Hello Gemini!"]}
output = graph.invoke(input_state)
print(output)

