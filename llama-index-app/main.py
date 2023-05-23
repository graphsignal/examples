import logging
import time
import random
from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# import and initialize graphsignal
# add GRAPSIGNAL_API_KEY to your environment variables
import graphsignal
graphsignal.configure(deployment='llama-index-example-demo')


# initialize index
documents = SimpleDirectoryReader("./data").load_data()
index = GPTVectorStoreIndex.from_documents(documents)

# answer questions
def answer_question(question):
  query_engine = index.as_query_engine()
  return query_engine.query(question)


# simulate requests
questions = [
    'Who proposed the concept of quantization and introduced the idea of energy being emitted in discrete packets or "quanta"?',
    'What is the significance of the uncertainty principle in quantum mechanics?',
    'Which two formulations of quantum mechanics were shown to be mathematically equivalent?',
    'What are some practical applications of quantum mechanics mentioned in the text?',
    'What is the field of study that aims to use quantum systems to simulate and understand complex phenomena?'
]

while True:
    try:
        q_num = random.randint(0, len(questions) - 1)
        print(f'Question: {questions[q_num]}')
        answer = answer_question(questions[q_num])
        print(f'Answer: {answer}')
    except:
        logger.error("Error while solving task", exc_info=True)

    time.sleep(5 * random.random())
