import time
import os
import json
from dotenv import load_dotenv
from src.agent import MultiMemoryAgent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()

class BenchmarkRunner:
    def __init__(self):
        self.agent = MultiMemoryAgent()

    def run_scenario(self, name: str, turns: list[str]):
        print(f"--- Scenario: {name} ---")
        # Clear short-term memory for each new scenario to ensure clean turns
        # but keep long-term/episodic/semantic for cross-turn tests if needed.
        # However, rubric says "multi-turn conversations", implying a continuous session
        # or separate sessions. Let's do separate scenarios for clarity.
        self.agent.short_term.clear()
        
        for i, query in enumerate(turns):
            print(f"Turn {i+1} User: {query}")
            result = self.agent.run(query)
            print(f"Turn {i+1} Agent (Intent: {result['intent']}): {result['response']}\n")
        print("-" * 30)

def run_benchmarks():
    runner = BenchmarkRunner()
    
    # 1. Profile Recall
    runner.run_scenario("Profile Recall", [
        "Hi, my name is Natus.",
        "What is my name?"
    ])

    # 2. Conflict Update (MANDATORY TEST)
    runner.run_scenario("Conflict Update", [
        "I am allergic to cow milk.",
        "Wait, I was wrong, I am allergic to soy, not cow milk.",
        "What am I allergic to?"
    ])

    # 3. Episodic Recall
    runner.run_scenario("Episodic Recall", [
        "I am struggling with Docker setup today. The container won't start.",
        "Can you help me fix the issue I mentioned earlier?"
    ])

    # 4. Semantic Retrieval
    # Pre-populate semantic memory first for testing
    runner.agent.semantic.save({"text": "The company policy states that employees get 20 days of PTO per year.", "metadata": {"source": "policy_doc"}})
    runner.run_scenario("Semantic Retrieval", [
        "How many days of PTO do I get per year according to policy?"
    ])

    # 5. Trim/Token Budget (Simulating long history)
    long_history_scenario = ["Talk to me about random topic " + str(i) for i in range(15)]
    long_history_scenario.append("What was the very first thing we talked about in this scenario?")
    runner.run_scenario("Context Trimming", long_history_scenario)

    # 6. Combined Preference & Logic
    runner.run_scenario("Preference Logic", [
        "I prefer dark mode in UI.",
        "Suggest a color scheme for my IDE based on my preferences."
    ])

    # 7. Cross-Session Consistency (Simulated by not clearing long-term)
    runner.run_scenario("Cross-Session Persistence", [
        "Do you still remember my name from the first scenario?"
    ])

    # 8. Complex Experience
    runner.run_scenario("Complex Experience", [
        "Last week I had a hard time understanding the GIL in Python.",
        "Summarize the technical challenges I face recently."
    ])

    # 9. Knowledge Discovery
    runner.run_scenario("Knowledge Discovery", [
        "I need to know about the capital of France.",
        "Tell me a fun fact about that city."
    ])

    # 10. Final Summary
    runner.run_scenario("Final Summary", [
        "Summarize everything you know about my preferences and past struggles today."
    ])

if __name__ == "__main__":
    # Clear all memories for a fresh start
    agent = MultiMemoryAgent()
    agent.long_term.clear()
    agent.episodic.clear()
    agent.semantic.clear()
    
    run_benchmarks()
