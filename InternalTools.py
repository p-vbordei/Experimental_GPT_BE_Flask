from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain import OpenAI, SerpAPIWrapper, LLMChain, LLMMathChain
     

import os



search = SerpAPIWrapper()
llm_math_chain = LLMMathChain(llm=lora_llm, verbose=True)

tools = [
    Tool(
        name = "Search",
        func=search.run,
        description="useful for when you need to answer questions about current events"
    ),
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="useful for when you need to answer questions about Divide, Multiply, Add and Subtract"
    )
]
     


prefix = """Answer the following questions as best you can. Think through step by step.
You have access to the following tools:"""
suffix = """Begin! Remember to think step by step

Question: {input}
{agent_scratchpad}"""

prompt = ZeroShotAgent.create_prompt(
    tools, 
    prefix=prefix, 
    suffix=suffix, 
    input_variables=["input", "agent_scratchpad"]
)


agent_chain = LLMChain(llm=lora_llm, prompt=prompt)
tool_names = [tool.name for tool in tools]
agent = ZeroShotAgent(llm_chain=agent_chain, 
                      allowed_tools=tool_names)
     


agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, 
                                                    tools=tools, 
                                                    verbose=True,
                                                    max_iterations=3)
     