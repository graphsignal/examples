import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

import graphsignal

logging.basicConfig(level=logging.DEBUG)

# Graphsignal: configure
#   expects GRAPHSIGNAL_API_KEY environment variable
graphsignal.configure(deployment='fastapi-langchain-example')


prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)

app = FastAPI()

@app.post("/generate")
async def generate(request: Request):
    body = await request.json()

    chat = ChatOpenAI(temperature=0.5)
    chain = LLMChain(llm=chat, prompt=prompt)
    output = await chain.arun(body['idea'])

    return JSONResponse(content={"output": output})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="debug")
