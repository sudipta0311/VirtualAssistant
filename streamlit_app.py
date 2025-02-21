from langchain_openai import OpenAIEmbeddings
import streamlit as st


import os

# Retrieve secrets using st.secrets
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
LANGCHAIN_TRACING_V2 = st.secrets.get("LANGCHAIN_TRACING_V2")
LANGCHAIN_API_KEY = st.secrets.get("LANGCHAIN_API_KEY")
PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY")


# Load existing vector store

from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings

pc = Pinecone(api_key=PINECONE_API_KEY)


embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# vector store 
index_name = "knowlodgebase"

index = pc.Index(index_name)

vector_store = PineconeVectorStore(index=index, embedding=embeddings)

retriever = vector_store.as_retriever()

from langchain.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_blog_posts",
    "Search and return information abput telecom products.",
)

tools = [retriever_tool]

############################# Utility tasks ############################################
from typing import Annotated, Sequence
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage

from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    # The add_messages function defines how an update should be processed
    # Default is to replace. add_messages says "append"
    messages: Annotated[Sequence[BaseMessage], add_messages]

from typing import Annotated, Literal, Sequence
from typing_extensions import TypedDict

from langchain import hub
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from pydantic import BaseModel, Field


from langgraph.prebuilt import tools_condition

def get_latest_user_question(messages):
    # Iterate over the messages in reverse order
    for role, content in reversed(messages):
        if role.lower() == "user":
            return content
    return ""

### Edges


def grade_documents(state) -> Literal["generate", "rewrite"]:
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (messages): The current state

    Returns:
        str: A decision for whether the documents are relevant or not
    """

    print("---CHECK RELEVANCE---")

    # Data model
    class grade(BaseModel):
        """Binary score for relevance check."""

        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    # LLM
    model = ChatOpenAI(temperature=0, model="gpt-4-0125-preview", streaming=True)

    # LLM with tool and validation
    llm_with_tool = model.with_structured_output(grade,method="function_calling")

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )

    chain = prompt | llm_with_tool

    messages = state["messages"]
    last_message = messages[-1]

    #question = messages[0].content
    #question = get_latest_user_question(messages)
    question = get_latest_user_question(st.session_state.conversation)

    docs = last_message.content

    scored_result = chain.invoke({"question": question, "context": docs})

    score = scored_result.binary_score

    if score == "yes":
        print("---DECISION: DOCS RELEVANT---")
        return "generate"

    else:
        print("---DECISION: DOCS NOT RELEVANT---")
        print(score)
        return "rewrite"


### Nodes


def agent(state):
    """
    Invokes the agent model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply end.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with the agent response appended to messages
    """
    print("---CALL AGENT---")
    messages = state["messages"]
    model = ChatOpenAI(temperature=0, streaming=True, model="gpt-4-turbo")
    model = model.bind_tools(tools)
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list

    return {"messages": [response]}


def rewrite(state):
    """
    Transform the query to produce a better question contextualized for telmore DENMARK.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with a re-phrased question specific to telmore DENMARK
    """

    print("---TRANSFORM QUERY FOR telmore DENMARK---")

    messages = state["messages"]
    #question = get_latest_user_question(messages)
    question = get_latest_user_question(st.session_state.conversation)


    # Prompt to force contextualization for telmore DENMARK
    msg = [
        HumanMessage(
            content=f"""
        You are a virtual assistant specializing in telmore denmark.
        Your job is to refine the user's question to be more specific to telmore Denmark’s services, plans, network, or offers.

        **User's Original Question:**
        {question}

        **Rewritten Question (must be relevant to telmore denmark):**
        """,
        )
    ]

    # Invoke the model to rephrase the question with Airtel context
    model = ChatOpenAI(temperature=0, model="gpt-4-0125-preview", streaming=True)
    response = model.invoke(msg)
    print("relevent conextualized question=" + response.content)

    print(response.content)
    return {"messages": [response]}


def generate(state):
    print("---GENERATE---")
    messages = state["messages"]

    #question = get_latest_user_question(messages)
    question = get_latest_user_question(st.session_state.conversation)
    # Assume the last assistant message (or retrieved content) holds the context.
    last_message = messages[-1]
    docs = last_message.content

    prompt = PromptTemplate(
        template = """
            You are an intelligent virtual assistant for a telecom company, leveraging real customer reviews and feedback to support business teams. Your primary role is to analyze customer sentiment, extract insights from reviews, provide actionable recommendations, and generate sales proposals when a customer inquires about a product.

            **Context Information (Retrieved Customer Reviews & Feedback):**
            {context}

            **User Inquiry:**
            {question}

            **Instructions:**
            - Analyze the provided customer feedback and extract relevant complaints, sentiments, feature requests, or competitive insights.
            - Provide a **concise summary** of the key issue or trend based on real customer experiences.
            - Offer **insightful recommendations** tailored to the specific business function (customer support, marketing, product management, competitor analysis, or sales).
            - If the inquiry is about a product or service, generate a **sales proposal** that highlights key benefits, features, pricing (if available), and relevant customer feedback.
            - If applicable, suggest improvements, new features, or messaging strategies based on customer sentiment.
            - If there is insufficient information, respond with: "I'm sorry, but I don't have enough information to provide a detailed response."
            - Format your response in a structured, **business-ready manner** that can be directly used for decision-making.

            **Response Format:**
            - **Summary:** A brief analysis of the customer concern, trend, or sentiment.
            - **Recommendation:** Suggested product, service, or improvement based on feedback.
            - **Sales Proposal (if applicable):** If the inquiry is product-related, generate a structured sales proposal including:
                - **Product Name & Description**
                - **Key Features & Benefits**
                - **Pricing (if available)**
                - **Relevant Customer Testimonials**
                - **Next Steps for Purchase**
            - **Additional Notes:** Any extra insights, follow-up actions, or areas requiring further investigation.

            **Example Use Cases (Supported Scenarios):**
            1. **AI-Powered Customer Support Agent** – Retrieve past feedback to auto-generate customer service responses.
            2. **Dynamic Review Analysis & Sentiment-Based Insights** – Summarize real-time customer sentiment trends.
            3. **Intelligent Product Feedback & Feature Request Analyzer** – Extract and prioritize new feature requests.
            4. **AI-Generated Review Summarization & Marketing Insights** – Summarize key strengths for marketing teams.
            5. **Automated Review Response Generation** – Generate personalized responses to customer reviews.
            6. **Competitor Benchmarking & Industry Analysis** – Compare your brand with competitors using sentiment trends.
            7. **Fake Review Detection & Authenticity Scoring** – Identify and flag potential fake or bot-generated reviews.
            8. **Proactive Business Decision Support** – Provide AI-driven recommendations for retention and growth.

        """
,
        input_variables=["context", "question"],
    )

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True)
    rag_chain = prompt | llm | StrOutputParser()
    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [response]}

################################# GRAPH##################################
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, AIMessage
from langgraph.graph.message import add_messages
import streamlit as st

# Initialize session state for conversation history if it doesn't exist.
if "conversation" not in st.session_state:
    st.session_state.conversation = []  # List of tuples like ("user", "question") or ("assistant", "response")
    # Initialize session state for retry count.
if "retry_count" not in st.session_state:
    st.session_state.retry_count = 0

# Define AgentState (we don't include retry_count in AgentState because we'll use session state)
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# New wrapper to limit retries using session state.
def grade_documents_limited(state) -> str:
    # Use the retry count from session state



    decision = grade_documents(state)  # This function must be defined elsewhere.
    retry_count = st.session_state.retry_count +1
    print("---TEST retry count is ---", retry_count)

    if decision == "rewrite":
        if retry_count >= 1:
            # Maximum retries reached: return a special decision "final"
            print("---Maximum retries reached: switching to final response---")
            return "final"
        else:
            # Increment the retry counter in session state.
            st.session_state.retry_count = retry_count + 1
            print("---after increment, retry count is ---", st.session_state.retry_count)
            return "rewrite"
    else:
        return decision
    
    # New node to handle the final response.
def final_response(state):
    final_msg = ("Sorry, this question is beyond my knowledge, as a virtual assistant I can only assist you "
                 "with your needs on telecom service")
    return {"messages": [AIMessage(content=final_msg)]}

# Define a new graph.
workflow = StateGraph(AgentState)

# Define the nodes (agent, retrieve, rewrite, generate, and final_response).
workflow.add_node("agent", agent)         # Agent node; function 'agent' must be defined.
retrieve = ToolNode([retriever_tool])       # 'retriever_tool' must be defined.
workflow.add_node("retrieve", retrieve)     # Retrieval node.
workflow.add_node("rewrite", rewrite)       # Rewriting the question; function 'rewrite' must be defined.
workflow.add_node("generate", generate)     # Generating the response; function 'generate' must be defined.
workflow.add_node("final_response", final_response)  # Final response node.

# Build the edges.
workflow.add_edge(START, "rewrite")
workflow.add_edge("rewrite", "agent")
workflow.add_conditional_edges(
    "agent",
    tools_condition,  # Function 'tools_condition' must be defined.
    {
        "tools": "retrieve",
        END: END,
    },
)
# In the retrieval branch, use the limited grade_documents function.
workflow.add_conditional_edges(
    "retrieve",
    grade_documents_limited,
    {
        "rewrite": "rewrite",
        "generate": "generate",
        "final": "final_response"
    }
)
workflow.add_edge("generate", END)
workflow.add_edge("rewrite", "agent")

# Compile the graph.
memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)

#############################################GUI#################################################
import uuid
import streamlit as st

# Generate a thread_id dynamically if it doesn't exist in session state.
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# Now use the dynamically generated thread_id in your config.
config = {"configurable": {"thread_id": st.session_state.thread_id}}

if "history" not in st.session_state:
    st.session_state.history = ""


import pprint


# Initialize session state for conversation history if it doesn't exist.
if "conversation" not in st.session_state:
    st.session_state.conversation = []  # List of tuples like ("user", "question") or ("assistant", "response")

def run_virtual_assistant():
    st.title("Virtual assitant to support service desk agent")

    # Display conversation history if available.
    if st.session_state.conversation:
        with st.expander("Click here to see the old conversation"):
            st.subheader("Conversation History")
            st.markdown({st.session_state.history})

    # Use a form to handle user input and clear the field after submission.
    with st.form(key="qa_form", clear_on_submit=True):
        user_input = st.text_input("Ask me your question (or type 'reset' to clear):")
        submit_button = st.form_submit_button(label="Submit")

    if submit_button and user_input:
        # Allow the user to reset the conversation.
        if user_input.strip().lower() == "reset":
            st.session_state.conversation = []
            st.session_state.retry_count = 0
            st.experimental_rerun()
        else:
            # Append the user's question to the conversation history.
            st.session_state.conversation.append(("user", user_input))
            st.session_state.retry_count = 0

            # Prepare the input for the graph using the entire conversation history.
            inputs = {
                "messages": st.session_state.conversation,            }
            
            final_message_content = ""
            # Process the input through the graph (assumes 'graph' is defined globally).
            for output in graph.stream(inputs, config):
                for key, value in output.items():
                    # Check if the value is a dict containing messages.
                    if isinstance(value, dict) and "messages" in value:
                        for msg in value["messages"]:
                            if hasattr(msg, "content"):
                                final_message_content = msg.content + "\n"
                                # Append the assistant response to conversation history.
                                st.session_state.conversation.append(("assistant", msg.content))
                            else:
                                final_message_content = str(msg) + "\n"
                                st.session_state.conversation.append(("assistant", str(msg)))

            # Render the final response.
            st.markdown(final_message_content)
            st.session_state.history+="################MESSAGE###############"
            st.session_state.history+=final_message_content


if __name__ == "__main__":
    run_virtual_assistant()

