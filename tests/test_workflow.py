from unittest.mock import patch

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage

from src.agents.graph import (app, explainer_node, query_node, route_execution,
                              route_query, validate_code)


def test_code_validation_allows_basic_analysis():
    assert validate_code("result = df['Age'].mean()\nprint(result)") is None


def test_code_validation_rejects_imports_and_introspection():
    assert validate_code("import os")
    assert validate_code("x = df.__class__")
    assert validate_code("x = open('/etc/passwd').read()")
    assert validate_code("df['Age'].mean()")


def test_retry_router_retries_only_within_limit():
    assert route_execution({"error": "failure", "retry_count": 1}) == "retry"
    assert route_execution({"error": "failure", "retry_count": 3}) == "failure"
    assert route_execution({"error": None, "retry_count": 1}) == "explain"


def test_greeting_is_routed_to_chat():
    state = {"messages": [HumanMessage(content="Hi!")], "dataset_schema": "Age: int"}
    fake_llm = type("FakeLLM", (), {"invoke": lambda self, _messages: AIMessage(content="CHAT")})()
    with patch("src.agents.graph.llm", fake_llm):
        classified = query_node(state)
    assert classified["intent"] == "CHAT"
    assert route_query({**state, **classified}) == "direct"


def test_classifier_failure_does_not_trigger_analysis():
    state = {"messages": [HumanMessage(content="hello")], "dataset_schema": "Age: int"}
    fake_llm = type("FailingLLM", (), {"invoke": lambda self, _messages: (_ for _ in ()).throw(
        RuntimeError("offline")
    )})()
    with patch("src.agents.graph.llm", fake_llm):
        classified = query_node(state)
    assert classified["intent"] == "ERROR"
    assert route_query({**state, **classified}) == "direct"


def test_result_values_are_not_sent_for_summary_without_opt_in():
    fake_llm = type("FailingLLM", (), {"invoke": lambda self, _messages: (_ for _ in ()).throw(
        AssertionError("LLM must not receive result values without consent")
    )})()
    state = {"share_values": False, "code_output": "secret result"}
    with patch("src.agents.graph.llm", fake_llm):
        result = explainer_node(state)
    assert "shown below" in result["explanation"]


def test_planner_provider_failure_is_terminal():
    state = {
        "messages": [HumanMessage(content="Average age?")],
        "dataset_schema": "Age: int", "dataframe": pd.DataFrame({"Age": [1]}),
        "plan": [], "code": "", "code_output": "", "explanation": "",
        "charts": [], "error": None, "retry_count": 0,
        "retry_history": [], "intent": "", "analysis_failed": False,
    }

    class FailingLLM:
        calls = 0

        def invoke(self, messages):
            self.calls += 1
            if self.calls == 1:
                return AIMessage(content="ANALYSIS")
            raise RuntimeError("provider unavailable")

    with patch("src.agents.graph.llm", FailingLLM()):
        result = app.invoke(state)
    assert result["analysis_failed"] is True
    assert "plan could not be created" in result["explanation"]
