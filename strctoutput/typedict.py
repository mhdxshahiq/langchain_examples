from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict, Annotated
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=api_key
)

#structured output using typedict
class review(TypedDict):
    summary: str
    sentiment: str
    example: Annotated[str, "A real world example illustrating the statement"]

# crearing a model that have structured output
strct_model = llm.with_structured_output(review)
#passing input tot the model
result = strct_model.invoke("iphones are good for money give an example and battery ")

print(result)



