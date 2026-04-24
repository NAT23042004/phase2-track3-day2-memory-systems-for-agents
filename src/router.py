from typing import Literal

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class Intent(BaseModel):
    intent_type: Literal["PREFERENCE", "FACTUAL", "EXPERIENCE", "GENERAL"] = Field(
        description="The type of memory interaction needed based on the user query."
    )
    reasoning: str = Field(description="Brief explanation for the chosen intent.")


class MemoryRouter:
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        self.parser = PydanticOutputParser(pydantic_object=Intent)

        self.prompt = ChatPromptTemplate.from_template(
            "Analyze the following user query and determine the most relevant memory interaction type.\n"
            "Types:\n"
            "- PREFERENCE: User sharing personal likes, dislikes, or settings (e.g., 'I love Python').\n"
            "- FACTUAL: User asking for factual information that might be in the knowledge base.\n"
            "- EXPERIENCE: User mentioning a past event, struggle, or specific session detail.\n"
            "- GENERAL: Conversational filler or queries that don't fit the above.\n"
            "\nQuery: {query}\n"
            "\n{format_instructions}"
        )

    def route(self, query: str) -> Intent:
        chain = self.prompt | self.llm | self.parser
        return chain.invoke({"query": query, "format_instructions": self.parser.get_format_instructions()})
