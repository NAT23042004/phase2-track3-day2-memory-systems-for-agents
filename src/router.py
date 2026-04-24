from typing import Literal, List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

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
            "Phân tích câu hỏi sau của người dùng và xác định loại tương tác bộ nhớ phù hợp nhất.\n"
            "Các loại (Types):\n"
            "- PREFERENCE: Người dùng chia sẻ sở thích cá nhân, điều không thích, hoặc cài đặt (ví dụ: 'Tôi thích Python').\n"
            "- FACTUAL: Người dùng hỏi thông tin thực tế có thể có trong kho kiến thức.\n"
            "- EXPERIENCE: Người dùng nhắc về một sự kiện đã qua, khó khăn đã gặp, hoặc chi tiết cụ thể trong phiên làm việc.\n"
            "- GENERAL: Các câu chào hỏi xã giao hoặc yêu cầu không thuộc các loại trên.\n"
            "\nCâu hỏi: {query}\n"
            "\n{format_instructions}"
        )

    def route(self, query: str) -> Intent:
        chain = self.prompt | self.llm | self.parser
        return chain.invoke({
            "query": query,
            "format_instructions": self.parser.get_format_instructions()
        })
