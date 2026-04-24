import json
from typing import Annotated, TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from src.memory.short_term import ShortTermMemory
from src.memory.long_term import LongTermMemory
from src.memory.episodic import EpisodicMemory
from src.memory.semantic import SemanticMemory
from src.router import MemoryRouter
from src.context_manager import ContextManager
from src.extractor import PreferenceExtractor

class AgentState(TypedDict):
    query: str
    intent: str
    retrieved_content: List[str]
    preferences: str
    short_term_history: List[BaseMessage]
    final_context: str
    response: str
    messages: List[BaseMessage]

class MultiMemoryAgent:
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.llm = ChatOpenAI(model=model_name)
        self.router = MemoryRouter(model_name=model_name)
        self.context_mgr = ContextManager(model_name=model_name)
        self.extractor = PreferenceExtractor(model_name=model_name)
        
        # Initialize Memories
        self.short_term = ShortTermMemory()
        # Redis might need a running server, but I'll instantiate it
        # In a real environment, I'd check connectivity
        self.long_term = LongTermMemory() 
        self.episodic = EpisodicMemory()
        self.semantic = SemanticMemory()
        
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("analyze_intent", self.analyze_intent)
        workflow.add_node("retrieve_memory", self.retrieve_memory)
        workflow.add_node("trim_context", self.trim_context)
        workflow.add_node("generate_response", self.generate_response)
        workflow.add_node("update_memory", self.update_memory)

        workflow.set_entry_point("analyze_intent")
        workflow.add_edge("analyze_intent", "retrieve_memory")
        workflow.add_edge("retrieve_memory", "trim_context")
        workflow.add_edge("trim_context", "generate_response")
        workflow.add_edge("generate_response", "update_memory")
        workflow.add_edge("update_memory", END)

        return workflow.compile()

    def analyze_intent(self, state: AgentState):
        intent_obj = self.router.route(state["query"])
        return {"intent": intent_obj.intent_type}

    def retrieve_memory(self, state: AgentState):
        intent = state["intent"]
        query = state["query"]
        retrieved = []
        
        # Always load preferences
        prefs_data = self.long_term.load()
        prefs = json.dumps(prefs_data)
        print(f"DEBUG: Loaded Preferences: {prefs}")
        
        if intent == "FACTUAL":
            retrieved = self.semantic.load(query)
            print(f"DEBUG: Retrieved Semantic Content: {retrieved}")
        elif intent == "EXPERIENCE":
            # For simplicity, load all episodes and let context manager handle it
            episodes = self.episodic.load()
            retrieved = [json.dumps(e) for e in episodes]
            print(f"DEBUG: Retrieved {len(retrieved)} Episodes")
            
        history = self.short_term.load()
        
        return {
            "preferences": prefs,
            "retrieved_content": retrieved,
            "short_term_history": history
        }

    def trim_context(self, state: AgentState):
        system_prompt = "You are a helpful assistant with multi-level memory. Use the provided USER PREFERENCES and RETRIEVED KNOWLEDGE/EXPERIENCES to personalize your response. If you see a user's name or allergy in PREFERENCES, acknowledge it when relevant."
        final_context = self.context_mgr.manage_context(
            system_prompt,
            state["preferences"],
            state["short_term_history"],
            state["retrieved_content"]
        )
        return {"final_context": final_context}

    def generate_response(self, state: AgentState):
        messages = [
            HumanMessage(content=f"Context:\n{state['final_context']}\n\nQuery: {state['query']}")
        ]
        response = self.llm.invoke(messages)
        return {"response": response.content}

    def update_memory(self, state: AgentState):
        intent = state["intent"]
        query = state["query"]
        response = state["response"]
        
        # Always update short-term
        self.short_term.save({"input": query, "output": response})
        
        # Robust Preference Extraction: check even if it's not PREFERENCE intent
        # (Corrections are often classified as EXPERIENCE)
        if intent in ["PREFERENCE", "EXPERIENCE", "GENERAL"]:
            pref_update = self.extractor.extract(query)
            if pref_update:
                print(f"DEBUG: Extracted Preference - Key: {pref_update.key}, Value: {pref_update.value}")
                self.long_term.save({pref_update.key: pref_update.value})
        
        if intent == "EXPERIENCE":
            self.episodic.save({"episode": query, "response": response})
            print(f"DEBUG: Saved Episode: {query[:50]}...")

        return {}

    def run(self, query: str):
        initial_state = {
            "query": query,
            "messages": [HumanMessage(content=query)]
        }
        return self.graph.invoke(initial_state)
