from langchain_community.document_loaders import TextLoader
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
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

prompt = PromptTemplate(
    input_variables=["text"],
    template="What is the highest degree of the person and his work experience : {text}"
)

parser = StrOutputParser()

loader = TextLoader(r"C:\FILES\projects\langchain chatbot\document_loader\CV.txt")

doc = loader.load()

chain = prompt | llm | parser

print(chain.invoke({'text': doc[0].page_content}))#in zeroth postion the content is availale

print(doc[0].metadata)
