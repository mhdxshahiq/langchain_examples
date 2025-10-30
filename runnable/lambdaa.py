from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableLambda, RunnablePassthrough, RunnableParallel
import os
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# def count(text):
#     return len(text.split())

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

joke_chain = RunnableSequence(template, llm, parser)

# final_chain = RunnableParallel({
#     "joke" : RunnablePassthrough(),
#     "text_count" : RunnableLambda(count)}

# )

final_chain = RunnableParallel(
    {
        "joke" : RunnablePassthrough(),
        "text_count" : RunnableLambda(lambda x: len(x.split()))
    }
)

chain = RunnableSequence(joke_chain, final_chain)

result = chain.invoke({"topic":"AI"})

print(result)