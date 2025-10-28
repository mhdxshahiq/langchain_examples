from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel
import os


load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")


llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=api_key
)


prompt1 = PromptTemplate(
    template="Generate one short and interesting fact about {topic}.",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Summarize this in one short line: {text}",
    input_variables=["text"]
)

prompt3 = PromptTemplate(
    template=(
        "Based on the following information, create one simple question:\n"
        "Fact: {facts}\n"
        "Summary: {summary}"
    ),
    input_variables=["facts", "summary"]
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    "facts": prompt1 | llm | parser,
    "summary": prompt2 | llm | parser
})

mega_chain = parallel_chain | (prompt3 | llm | parser)


result = mega_chain.invoke({
    "topic": "black holes",
    "text": "Black holes are dense objects with gravity so strong that not even light can escape."
})


print(result)

print(mega_chain.get_graph().print_ascii())
