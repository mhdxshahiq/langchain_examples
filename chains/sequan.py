from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")


llm = ChatGoogleGenerativeAI(
    model='gemini-2.0-flash',
    google_api_key=api_key
)

prompt1 = PromptTemplate(
    template='Generate a short 5 interesting facts about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template = 'summary in one line {text}',
    input_variables=['text']
)

parser = StrOutputParser()

chain = prompt1 | llm | parser | prompt2 | llm | parser

result = chain.invoke({"topic":"cricket"})
print(result)


print(chain.get_graph().print_ascii())
