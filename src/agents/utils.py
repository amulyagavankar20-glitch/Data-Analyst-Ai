from typing import TypedDict, Annotated, List, Union
from langchain_core.messages import BaseMessage
import operator

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    dataset_schema: str
    plan: List[str]
    code: str
    code_output: str
    explanation: str
    figures: List[object] # Matplotlib figures
    dataframe: object # The actual pandas DataFrame
    error: str # Error message from execution if any
    retry_count: int # To prevent infinite loops
