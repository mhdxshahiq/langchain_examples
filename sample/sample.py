from langchain_google_genai import ChatGoogleGenerativeAI
from sample.system_prompt import system_instruction
from dotenv import load_dotenv
import os
import json

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Structured instruction for "what is" questions
structured_instruction = (
    "\nWhen you answer, always use this structure:\n"
    "Definition: [short and simple]\n"
    "Example: [give one simple example]\n"
    "Importance: [why it matters in short]\n"
)

llm = ChatGoogleGenerativeAI(
    model='gemini-2.0-flash',
    temperature=0.3,
    max_output_tokens=200,
    google_api_key=api_key
)

chat_history = []

def chat_loop():
    while True:
        user_input = input("Human: ")
        if user_input.lower() == "exit":
            print("Chat ended.")
            break

        if user_input.lower().startswith("what is"):
            system_msg = f"{system_instruction}\n{structured_instruction}"
        else:
            system_msg = system_instruction

        # Messages with roles
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_input}
        ]

        # AI response
        response = llm.invoke(messages)
        ai_message = response.content.strip()#only content is printing

        print("AI:", ai_message, "\n")

        # Save chat history with roles
        chat_history.append({
            "system": system_msg,
            "human": user_input,
            "ai": ai_message
        })

    # Save chat history to JSON
    with open("chat_history.json", "w") as f:
        json.dump(chat_history, f, ensure_ascii=False, indent=4)
    print("Chat history saved to chat_history.json")

chat_loop()
