# chatbot.py
import os
from dotenv import load_dotenv
from utils import load_faiss_store
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("❌ OPENAI_API_KEY not found in .env file")

    print("🧠 Loading vectorstore...")
    store = load_faiss_store(openai_api_key=api_key)

    retriever = store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    llm = ChatOpenAI(openai_api_key=api_key, model="gpt-3.5-turbo")
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

    print("🤖 Chatbot ready! Type 'exit' to quit.\n")

    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break
        result = qa_chain.run(query)
        print(f"Bot: {result}\n")
