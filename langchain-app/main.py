import os
import logging
import time
import logging
import random
import langchain
from langchain.agents import Tool, initialize_agent, load_tools
from langchain.llms import OpenAI
import graphsignal


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Graphsignal: configure
#   expects GRAPHSIGNAL_API_KEY environment variable
graphsignal.configure(deployment='langchain-app-example')


@graphsignal.trace_function
def solve(task):
    llm = OpenAI(temperature=0)
    tools = load_tools(["llm-math"], llm=llm)
    agent = initialize_agent(
        tools, llm, agent="zero-shot-react-description", verbose=True
    )
    agent.run(task)


while True:
    num = random.randint(0, 100)
    solve(f"What is {num} raised to .123243 power?")

    time.sleep(20 * random.random())
