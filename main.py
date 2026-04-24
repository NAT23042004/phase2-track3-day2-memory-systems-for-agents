import os
from dotenv import load_dotenv
from src.agent import MultiMemoryAgent

# Load environment variables
load_dotenv()

def main():
    agent = MultiMemoryAgent()
    print("Chào mừng bạn đến với Multi-Memory Agent CLI! (Gõ 'exit' hoặc 'quit' để thoát)")
    
    while True:
        query = input("\nBạn: ")
        if query.lower() in ["exit", "quit"]:
            break
            
        result = agent.run(query)
        print(f"\nAgent: {result['response']}")
        print(f"(Ý định: {result['intent']})")

if __name__ == "__main__":
    main()
