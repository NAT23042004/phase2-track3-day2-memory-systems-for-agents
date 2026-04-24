from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

class PreferenceUpdate(BaseModel):
    key: str = Field(description="The specific category of preference (e.g., 'allergy', 'programming_language', 'name'). Use snake_case.")
    value: str = Field(description="The current value for this preference. If the user corrected a previous value, provide the new one.")
    is_correction: bool = Field(description="Set to true if the user is correcting or updating a previously stated preference.")

class PreferenceExtractor:
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        self.parser = PydanticOutputParser(pydantic_object=PreferenceUpdate)
        
        self.prompt = ChatPromptTemplate.from_template(
            "Extract user preferences from the query as key-value pairs.\n"
            "If the user provides a correction (e.g., 'No, I meant X not Y'), identify the key and the new value.\n"
            "\nQuery: {query}\n"
            "\n{format_instructions}"
        )

    def extract(self, query: str) -> Optional[PreferenceUpdate]:
        chain = self.prompt | self.llm | self.parser
        try:
            return chain.invoke({
                "query": query,
                "format_instructions": self.parser.get_format_instructions()
            })
        except Exception:
            return None
