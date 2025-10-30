from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence
import os

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model='gemini-2.0-flash',
    temperature=0.3,
    max_output_tokens=200,
    google_api_key=api_key
)

template = PromptTemplate(
    template = 'write a joke about {topic}',
    input_variables = ['topic']

)

tempalte2 = PromptTemplate(
    template ="what is the meaning of the following joke {text}",
    input_variables=["text"]

)

parser = StrOutputParser()

chain = RunnableSequence(template, llm , parser, tempalte2, llm,parser)

result = chain.invoke({'topic':"ai"})
print(result)