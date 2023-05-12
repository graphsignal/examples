import logging
import time
import random
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# import and initialize graphsignal
# add GRAPSIGNAL_API_KEY to your environment variables
import graphsignal
graphsignal.configure(deployment='langchain-chroma-example')


loader = TextLoader('data.txt')
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(texts, embeddings)


def answer_question(question):
    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=vectordb.as_retriever())

    return qa.run(question)


# simulate some requests

questions=[
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
