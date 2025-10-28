from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Initialize model
llm = ChatGoogleGenerativeAI(
    model='gemini-2.0-flash',
    temperature=0.3,
    max_output_tokens=200,
    google_api_key="YOUR_API_KEY"
)

# -------- Single-turn static message --------
single_static = "Explain what a car is."
response = llm.invoke(single_static)
print("Single-turn static:", response.content)

# -------- Single-turn dynamic message --------
from langchain.prompts import PromptTemplate
single_dynamic_template = PromptTemplate(
    input_variables=["topic"],
    template="Explain {topic} in simple words."
)
single_dynamic_msg = single_dynamic_template.format(topic="bicycle")
response = llm.invoke(single_dynamic_msg)
print("Single-turn dynamic:", response.content)

# -------- Multi-turn static messages --------
messages = [
    SystemMessage(content="You are a helpful tutor."),
    HumanMessage(content="What is photosynthesis?")
]
response = llm.invoke(messages)
print("Multi-turn static:", response.content)

# -------- Multi-turn dynamic messages (ChatPromptTemplate) --------
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful tutor."),
    ("human", "Explain {topic} in simple words.")
])
formatted_messages = chat_template.format_messages(topic="gravity")
response = llm.invoke(formatted_messages)
print("Multi-turn dynamic:", response.content)
