from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel
import os
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model='gemini-2.0-flash',
    temperature=0.3,
    max_output_tokens=200,
    google_api_key=api_key
)

Prompt1 = PromptTemplate(
    template = 'Generate a tweet about {topic}',
    input_variables=["topic"]
)

Prompt2 = PromptTemplate(
    template = 'Generate a linkedin post about {topic}',
    input_variables=["topic"]
)

parser = StrOutputParser()

parallel_chain = RunnableParallel(
    {'tweet': RunnableSequence(Prompt1, llm ,parser),
     'linkedin': RunnableSequence(Prompt2, llm , parser)}
)

result = parallel_chain.invoke({'topic':"AI"})

print(result)