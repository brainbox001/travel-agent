import asyncio
import os
from typing import Annotated
from langchain_groq import ChatGroq
from typing_extensions import TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from agent import agent
from tool_executor import ToolExecutorNode
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage
from memory import memory


class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]
    context: Annotated[list, "a list of relevant context from different tools called to be given to the LLM to generate a response"]
    tool_calls_made: Annotated[dict, "a dictionary of tool calls made by the brain node"]

graph_builder = StateGraph(State)

try:
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable not set.")
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0.3
    )
except ValueError as e:
    print(f"Error initializing LLM: {e}")
    llm = None
except Exception as e:
    print(f"An unexpected error occurred during LLM initialization: {e}")
    llm = None

# Tools definitions
# ----------------------------------------------------------------------------------------------------
@tool # type: ignore
async def faqs_and_policy_agent(question: Annotated[str, "The question or questions from the user's input directly related to this tool"]):
    """
    This tool explicitly handles faqs, questions about the company and airlines in general including their policies, and travel guidelines.
    """
    try:
        query_prompt = await agent.query(question)
        return {"context": query_prompt, "question": question}
    except Exception as e:
        print(f"Error in faqs_and_policy_agent: {e}")
        return {"context": f"Error: Failed to process the request due to a system error. Please try again later.", "question": question}


@tool # type: ignore
async def flights_finder(from_: Annotated[str, "The city or town of the flight depature, the takeoff point, extracted from the question"], to_: Annotated[str, "The destination point of the flight extracted from the question"], question: Annotated[str, "The question related to the tool"]):
    """
    This tool finds the best flight deals from the database based on the user's input.
    """
    try:
        # Mock flight search results from a database or API
        mock_result = [
            {"flight": "Flight 1", "price": "$200"},
            {"flight": "Flight 2", "price": "$250"},
            {"flight": "Flight 3", "price": "$300"},
        ]
        return {"context": f"Here is a list of best flight deals from {from_} to {to_} : {mock_result}", "question": question}
    except Exception as e:
        print(f"Error in flights_finder: {e}")
        return {"context": f"Error: Failed to find flights from {from_} to {to_}. Please check the details and try again.", "question": question}


@tool # type: ignore
async def unfamiliar(question: Annotated[str, "questions or tasks the llm has marked as not directly or indirectly linked to any other tool"]):
    """"
    All the questions or request that is not related to the other tools available to the llm should be sent to this unfamiliar tool.
    """

    return {"context": "Unfamiliar question", "question": question}


tools = [faqs_and_policy_agent, flights_finder, unfamiliar]
tools_llm = llm.bind_tools(tools) if llm else None

# ----------------------------------------------------------------------------------------------------

# Node definitions
# ----------------------------------------------------------------------------------------------------

tool_node = ToolExecutorNode(tools=tools)

async def response_node(state: State) -> str:
    try:
        if not agent:
            raise RuntimeError("Agent is not initialized.")
        return {"messages" : [await agent.get_response(state["context"], state["messages"])], "context" : [], "tool_calls_made" : {}}
    except Exception as e:
        print(f"Error in response_node: {e}")
        return {"messages": [AIMessage(content=f"An error occurred while generating the response: {e}")], "context" : [], "tool_calls_made" : {}} 

async def brain_node(state : State) -> dict:
    try:
        if not tools_llm:
            raise RuntimeError("LLM is not initialized. Cannot proceed.")
        user_input = state["messages"][-1].content
        template = """
        You are the brain of this travel agent system.
        The "Past_Chats" at the end of the prompt is a list of past interactions you've had with this user.

        Extract as many questions a user's input(attached in the "Input" towards the end of the prompt) might contain.

        Before making a tool call, you should do the following:
            1. Check "Past_Chats" to see if you have any past interactions with the user.

            2. If you have past interactions with the user, do the following:
                - The chat priority should go in descending order, that is from the last chat to the first.
                - If you're referencing a pronoun, you should reference it to the first appropriate reference you find on the chat list(starting from the last to first chat)
                - For each question you extracted, check for pronouns that are not a type of personal   pronoun(examples, "I", "me", "we", "us"), and replace them with their appropriate reference from "Past_Chats" and update the question to reflect the change.

                - If there is no appropriate reference for a pronoun, just ignore and move on.
                - Then based on the updated question, decide which tool call to make.

            3. unfamiliar questions or questions you marked as not having a function call should be sent to the unfamiliar tools

            4. If there are no past interactions with the user, you can go ahead to make the tool calls for all the questions you extracted.

        Input: {user_input}
        Past_Chats: {messages}
        """
        prompt = ChatPromptTemplate.from_template(template)
        rendered_prompt = await prompt.ainvoke({"user_input": user_input, "messages": state["messages"]})
        return {"tool_calls_made" : await tools_llm.ainvoke(rendered_prompt)}
    except RuntimeError as e:
        print(f"Error in brain_node: {e}")
        return {"messages": [AIMessage(content=f"Error: {e}")]}
    except Exception as e:
        print(f"An unexpected error of type {type(e)} occurred in brain_node: {e}")
        return {"messages": [AIMessage(content=f"An unexpected error occurred on the brain. Please try again.")]}


# ----------------------------------------------------------------------------------------------------

graph_builder.add_node("brain", brain_node)
graph_builder.add_node("tools", tool_node)
graph_builder.add_node("response", response_node)
graph_builder.add_edge(START, "brain")
graph_builder.add_edge("brain", "tools")
graph_builder.add_edge("tools", "response")
graph_builder.add_edge("response", END)

graph = graph_builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "1"}}

async def stream_graph_updates(user_input: str):
    try:
        async for event in graph.astream({"messages": [{"role": "user", "content": user_input}]}, config=config):
            for value in event.values():
                print("Assistant:", value)
    except Exception as e:
        print(f"An error occurred during graph execution: {e}")

async def main():
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["exit", "quit", "stop"]:
                print("Exiting the chat. Goodbye!")
                break
            if not user_input.strip():
                print("Please enter a valid question.")
                continue
            await stream_graph_updates(user_input)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting the chat. Goodbye!")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())
