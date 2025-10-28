from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict, Annotated,Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=api_key
)

#class structured output using pydantic
class review(BaseModel):
    summary: str
    sentiment: str
    example: Annotated[str, "A real-world example illustrating the statement"] # using annoted we can give command to LLM
    battery: float = Field(gt=3500, lt=6000, description="Battery capacity in mAh")
    rate: Optional[float] = Field(gt=3500,lt=6000,description="rate of the phone  (optional field)")

#carete a model that have structured output
strct_model = llm.with_structured_output(review)
#passing input into the model
result = strct_model.invoke("iphones are good for money give an example and battery is 3600 and the rate is above 3500")

print(result)



