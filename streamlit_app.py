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
index_name = "demoindex"

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
from langchain_core.prompts import ChatPromptTemplate


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
    Transform the query to produce a better question contextualized for YOUSEE DENMARK.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with a re-phrased question specific to YOUSEE DENMARK
    """

    print("---TRANSFORM QUERY FOR YOUSEE DENMARK---")

    messages = state["messages"]
    #question = get_latest_user_question(messages)
    question = get_latest_user_question(st.session_state.conversation)


    # Prompt to force contextualization for YOUSEE DENMARK
    msg = [
        HumanMessage(
            content=f"""
        You are a virtual assistant specializing in Yousee denmark.
        Your job is to refine the user's question to be more specific to Yousee Denmark’s services, plans, network, or offers.

        **User's Original Question:**
        {question}

        **Rewritten Question (must be relevant to Yousee denmark):**
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
        template="""You are a telecom sales agent specializing in providing the best offers and plans for customers.
        Your goal is to assist customers by answering their questions, providing relevant information based on the available context,
        and creating a compelling sales proposal that convinces them to choose a product or service.
        
        **Context Information:**
        {context}
        **Customer's Question:**
        {question}
        
        **Instructions:**
        - If the context contains relevant details, use them to craft a persuasive sales pitch.
        - Highlight the key benefits, special offers, and why the customer should choose this product or service.
        - If no relevant information is available, politely inform the customer:
          "I'm sorry, but I don't have the details for that request at the moment."
        """,
        input_variables=["context", "question"],
    )

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True)
    rag_chain = prompt | llm | StrOutputParser()
    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [response]}

### Hallucination Grader


# Data model
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


# LLM with function call
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeHallucinations)

# Prompt
system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

hallucination_grader = hallucination_prompt | structured_llm_grader
hallucination_grader.invoke({"documents": docs, "generation": generation})



### Answer Grader


# Data model
class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


# LLM with function call
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeAnswer)

# Prompt
system = """You are a grader assessing whether an answer addresses / resolves a question \n 
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

answer_grader = answer_prompt | structured_llm_grader
answer_grader.invoke({"question": question, "generation": generation})

### Question Re-writer

# LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Prompt
system = """You a question re-writer that converts an input question to a better version that is optimized \n 
     for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Here is the initial question: \n\n {question} \n Formulate an improved question.",
        ),
    ]
)

question_rewriter = re_write_prompt | llm | StrOutputParser()
question_rewriter.invoke({"question": question})


#####################################

def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    state["question"]
    filtered_documents = state["documents"]

    if not filtered_documents:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
        )
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"


def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score.binary_score

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score.binary_score
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        pprint("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"
    
#######################################

def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]

    # Re-write question
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}


################################# GRAPH##################################
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, AIMessage
from langgraph.graph.message import add_messages
import streamlit as st

# Initialize session state if not already set.
if "conversation" not in st.session_state:
    st.session_state.conversation = []  # List of (role, message)
if "generation_retry_count" not in st.session_state:
    st.session_state.generation_retry_count = 0
if "max_generation_retries" not in st.session_state:
    st.session_state.max_generation_retries = 2  # Configurable maximum retries

# Define AgentState (we don't include retry_count in AgentState because we'll use session state)
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

    
    # New node to handle the final response.
def final_response(state):
    final_msg = ("Sorry, this question is beyond my knowledge, as a virtual assistant I can only assist you "
                 "with your needs on telecom service")
    return {"messages": [AIMessage(content=final_msg)]}

def grade_generation_and_retry(state):
    """
    Invokes grade_generation_v_documents_and_question to assess whether the generation is
    grounded in the retrieved documents and addresses the question. If not, it checks
    the generation retry counter and either triggers a query transformation or returns a
    final decision.
    """
    print("---CHECKING GENERATED ANSWER QUALITY---")
    # Invoke your provided method to assess generation quality.
    decision = grade_generation_v_documents_and_question(state)
    
    if decision == "useful":
        # Answer is good: stop processing.
        return "useful"
    else:
        # For "not useful" or "not supported", decide whether to re-run the query.
        max_retries = st.session_state.max_generation_retries
        current_retry = st.session_state.generation_retry_count
        if current_retry < max_retries:
            st.session_state.generation_retry_count = current_retry + 1
            print(f"---Retry {st.session_state.generation_retry_count} of {max_retries}: Transforming query---")
            return "transform_query"
        else:
            print("---Maximum generation retries reached: returning final response---")
            return "final"
# --- Build the Graph ---

workflow = StateGraph(AgentState)

# Add nodes.
workflow.add_node("agent", agent)         # Agent node (must be defined elsewhere).
retrieve = ToolNode([retriever_tool])       # Retrieval tool node (retriever_tool must be defined).
workflow.add_node("retrieve", retrieve)     # Retrieval node.
workflow.add_node("rewrite", rewrite)       # Rewrite node (function 'rewrite' must be defined).
workflow.add_node("generate", generate)     # Generate node (function 'generate' must be defined).
workflow.add_node("final_response", final_response)  # Final response node.
workflow.add_node("transform_query", transform_query)  # Transform query node.
workflow.add_node("grade_generation", grade_generation_and_retry)  # Grade generation node.

# Define graph edges.
workflow.add_edge(START, "rewrite")
workflow.add_edge("rewrite", "agent")
workflow.add_conditional_edges(
    "agent",
    tools_condition,  # Decides whether to retrieve; must be defined.
    {
        "tools": "retrieve",
         END: END,
    },
)
workflow.add_conditional_edges(
    "retrieve",
    grade_documents,  # Now using the original grade_documents without retry logic.
    {
        "rewrite": "rewrite",
        "generate": "generate",
        "final": "final_response"
    }
)
# After generation, assess the quality of the answer.
workflow.add_edge("generate", "grade_generation")
workflow.add_conditional_edges(
    "grade_generation",
    lambda state: state,  # This node returns a branch key.
    {
        "useful": END,
        "transform_query": "transform_query",
        "final": "final_response"
    }
)
# If the generation is not useful and we still have retries left,
# transform the query and restart the cycle.
workflow.add_edge("transform_query", "agent")

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
    st.title("Virtual Agent")

    # Display conversation history if available.
    if st.session_state.conversation:
        with st.expander("Click here to see the old conversation"):
            st.subheader("Conversation History")
            st.markdown({st.session_state.history})

    # Use a form to handle user input and clear the field after submission.
    with st.form(key="qa_form", clear_on_submit=True):
        user_input = st.text_input("Ask me anything about telco offers (or type 'reset' to clear):")
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