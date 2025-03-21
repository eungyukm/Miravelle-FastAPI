from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class LLMResult(Base):
    __tablename__ = "llm_results"
    __table_args__ = {"schema": "fastapi_schema"}

    id = Column(Integer, primary_key=True, index=True)
    input_text = Column(String, nullable=False)
    output_text = Column(String, nullable=False)
