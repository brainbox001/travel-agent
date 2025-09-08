import asyncio

# from langchain_core.messages import ToolMessage


class ToolExecutorNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    async def __call__(self, inputs: dict):
        message = inputs.get("tool_calls_made", {})

        if not message :
            return {"context" : "No tool call was made for this question"}

        print(f"tools made by the brain llm : - {message}")
        # Create a list of tasks for each tool call
        tasks = []
        for tool_call in message.tool_calls:
            tasks.append(
            self.tools_by_name[tool_call["name"]].ainvoke(tool_call["args"])
            )            

        # tasks = [
        #     self.tools_by_name[tool_call["name"]].ainvoke(tool_call["args"])
        #     for tool_call in message.tool_calls
        # ]
     
        tool_results = await asyncio.gather(*tasks)
        # print(f"This is tool_results {tool_results}")
        return {"context": tool_results}
