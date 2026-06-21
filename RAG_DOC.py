from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint,HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langgraph.graph import StateGraph,START,END 
from typing import Annotated,TypedDict 
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage,BaseMessage
from langgraph.prebuilt import ToolNode,tools_condition

import os

#import HuggingFace 
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

llm = HuggingFaceEndpoint(
    repo_id= "OBLITERATUS/Gemma-4-12B-OBLITERATED",
    task="text-generation",
    max_new_tokens=100
)

model = ChatHuggingFace(llm = llm)

# step - 1
# __________________________load_pdf______________________________#
PDF_PATH = r'C:\Users\Admin\Desktop\Deep+Learning+Ian+Goodfellow.pdf'

loader = PyPDFLoader(PDF_PATH)
docs = loader.load()

# split 

splitter = RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlap = 200)
chunks = splitter.split_documents(docs)



# embedding #
embedded = HuggingFaceEmbeddings(model="google/embeddinggemma-300m")

# vector_store
vector_store = FAISS.from_documents(chunks,embedded)


# reteriver
retriever = vector_store.as_retriever(
    search_type = "similarity",
    search_kwargs = {"k" :2}
)

# ------------- Now working of langraph ------------ #
@tool
def rag_tool(query: str) -> dict:
    """
    Retrieve relevant information from the PDF.
    """

    result = retriever.invoke(query)

    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        "query": query,
        "context": context,
        "metadata": metadata
    }
tools = [rag_tool]
llm_with_tools = model.bind_tools(tools)


#_____________________ Langgraph ________________#
class Chatstate(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]

#_______________ Node_____________________________#
def Chat_node(state:Chatstate):
    message = state['messages']
    response = llm_with_tools.invoke(message)
    return{'messages':[response]}

tool_node = ToolNode(tools)



#_________________Graph____________________________#
graph = StateGraph(Chatstate)

graph.add_node('chat_node',Chat_node)
graph.add_node('tools',tool_node)

graph.add_edge(START,'chat_node')
graph.add_conditional_edges(
    "chat_node",
    tools_condition
)
graph.add_edge('tools','chat_node')

chatbot = graph.compile()

result = chatbot.invoke(
    {
        'messages':[
            HumanMessage(
                content = ("what's deep learning")
            )
        ]
    }
)



print(result['messages'][-1].content)

