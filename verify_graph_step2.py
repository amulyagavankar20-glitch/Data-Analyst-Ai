"""Deterministic workflow smoke check using mocked LLM and worker responses."""
import os
from unittest.mock import patch

os.environ.setdefault("GROQ_API_KEY", "test-key")

import pandas as pd
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage

from src.agents.graph import app


class FakeLLM:
    def __init__(self):
        self.responses = iter([
            "ANALYSIS",
            "Calculate the average age",
            "result = df['Age'].mean()\nprint(result)",
            "The average age is 37.",
        ])

    def invoke(self, _messages):
        return AIMessage(content=next(self.responses))


def main():
    frame = pd.DataFrame({"Age": [22, 35, 58, 25, 45]})
    initial_state = {
        "messages": [HumanMessage(content="What is the average Age?")],
        "dataset_schema": "Columns: Age; Types: Age int64; Shape: (5, 1)",
        "dataframe": frame,
        "plan": [], "code": "", "code_output": "", "explanation": "",
        "charts": [], "error": None, "retry_count": 0,
        "retry_history": [], "intent": "", "analysis_failed": False,
        "share_values": True,
    }

    def fake_worker(code, dataframe):
        assert "df['Age'].mean()" in code
        return {"output": "37.0", "dataframe": dataframe.copy(), "charts": []}

    with patch("src.agents.graph.llm", FakeLLM()), \
            patch("src.agents.graph.run_isolated_code", fake_worker):
        result = app.invoke(initial_state)
    assert result["explanation"] == "The average age is 37."
    assert not result.get("analysis_failed")
    print("Deterministic workflow smoke check passed.")


if __name__ == "__main__":
    main()
