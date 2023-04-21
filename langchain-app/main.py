import logging
import time
import random
from langchain.agents import initialize_agent, load_tools
from langchain.chat_models import ChatOpenAI

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# import and initialize graphsignal
# add GRAPSIGNAL_API_KEY to your environment variables
import graphsignal
graphsignal.configure(deployment='langchain-demo')

def solve(task):
    llm = ChatOpenAI(temperature=0)
    tools = load_tools(["llm-math"], llm=llm)
    agent = initialize_agent(
        tools, llm, agent="zero-shot-react-description", verbose=True
    )
    agent.run(task)


# simulate some requests
while True:
    num = random.randint(0, 100)
    
    try:
        solve(f"What is {num} raised to .123243 power?")
        logger.debug('Task solved')
    except:
        logger.error("Error while solving task", exc_info=True)

    time.sleep(5 * random.random())
