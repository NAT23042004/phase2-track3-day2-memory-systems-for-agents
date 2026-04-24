import os
from dotenv import load_dotenv
from src.agent import MultiMemoryAgent

# Load environment variables
load_dotenv()

def main():
    agent = MultiMemoryAgent()
    print("Welcome to Multi-Memory Agent CLI! (Type 'exit' to quit)")
    
    while True:
        query = input("\nYou: ")
        if query.lower() in ["exit", "quit"]:
            break
            
        result = agent.run(query)
        print(f"\nAgent: {result['response']}")
        print(f"(Intent: {result['intent']})")

if __name__ == "__main__":
    main()
