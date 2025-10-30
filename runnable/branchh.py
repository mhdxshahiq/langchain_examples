from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableLambda, RunnablePassthrough, RunnableBranch
import os
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

def shit(text):
    return "lenghty joke"

llm = ChatGoogleGenerativeAI(
    model='gemini-2.0-flash',
    temperature=0.3,
    google_api_key=api_key
)

prompt = PromptTemplate(
    template = 'write a joke about {topic}',
    input_variables = ['topic']

)

prompt2 = PromptTemplate(
    template ="Summaries the following {text}",
    input_variables=["text"]

)

parser = StrOutputParser()

joke_chain = RunnableSequence(prompt, llm, parser)

chain = RunnableBranch(
    (lambda x : len(x.split())>10,RunnableLambda(shit)),
    RunnablePassthrough()
)

final = RunnableSequence(joke_chain, chain)

result = final.invoke({"topic": "car"})

print(result)