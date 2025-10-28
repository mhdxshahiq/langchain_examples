from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model='gemini-2.0-flash',
    temperature=0.3,
    max_output_tokens=200,
    google_api_key=api_key
)

# 1st prompt - report
template1 = PromptTemplate(
    template="write a detailed report on {topic}",
    input_variables=['topic']
)

# 2nd prompt - summary
template2 = PromptTemplate(
    template="write a 5 line summary on the following text {text}",
    input_variables=['text']
)

parser = StrOutputParser()

chain = template1 | llm | parser | template2 | llm | parser

result = chain.invoke({'topic':'black hole'})
print(result)