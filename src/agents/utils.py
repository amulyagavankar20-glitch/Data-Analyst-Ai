from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage
import operator

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    dataset_schema: str
    plan: List[str]
    code: str
    code_output: str
    explanation: str
    charts: List[object] # Interactive Plotly figures
    dataframe: object # The actual pandas DataFrame
    error: str # Error message from execution if any
    retry_count: int # To prevent infinite loops
    retry_history: List[dict]
    intent: str
    analysis_failed: bool
    share_values: bool
