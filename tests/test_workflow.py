from unittest.mock import patch

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage

from src.agents.graph import (app, direct_node, explainer_node, query_node,
                              route_execution, route_query, validate_code)


def test_code_validation_allows_basic_analysis():
    assert validate_code("result = df['Age'].mean()\nprint(result)") is None
    assert validate_code("fig = px.bar(df, x='Age')") is None


def test_code_validation_rejects_imports_and_introspection():
    assert validate_code("import os")
    assert validate_code("x = df.__class__")
    assert validate_code("x = open('/etc/passwd').read()")
    assert validate_code("df['Age'].mean()")


def test_analysis_code_cannot_mutate_dataframe_but_modify_intent_can():
    mutation = "df['Age'] = df['Age'].fillna(df['Age'].median())\nresult = df['Age'].mean()"
    inplace_mutation = "df.dropna(inplace=True)\nresult = len(df)"
    assert validate_code(mutation)
    assert validate_code(inplace_mutation)
    assert validate_code(mutation, allow_data_mutation=True) is None
    assert validate_code("analysis_df = df.copy()\nanalysis_df['Age'] = 0\nresult = analysis_df['Age'].mean()") is None


def test_retry_router_retries_only_within_limit():
    assert route_execution({"error": "failure", "retry_count": 1}) == "retry"
    assert route_execution({"error": "failure", "retry_count": 3}) == "failure"
    assert route_execution({"error": None, "retry_count": 1}) == "explain"
    assert route_execution({"error": "worker unavailable", "error_category": "infrastructure",
                            "retry_count": 1}) == "failure"


def test_greeting_is_routed_to_chat():
    state = {"messages": [HumanMessage(content="Hi!")], "dataset_schema": "Age: int"}
    fake_llm = type("FakeLLM", (), {"invoke": lambda self, _messages: AIMessage(content="CHAT")})()
    with patch("src.agents.graph.llm", fake_llm):
        classified = query_node(state)
    assert classified["intent"] == "CHAT"
    assert route_query({**state, **classified}) == "direct"


def test_explicit_modification_intent_routes_to_planning():
    state = {"messages": [HumanMessage(content="Remove rows with missing values")],
             "dataset_schema": "Age: int"}
    fake_llm = type("FakeLLM", (), {"invoke": lambda self, _messages: AIMessage(content="MODIFY")})()
    with patch("src.agents.graph.llm", fake_llm):
        classified = query_node(state)
    assert classified["intent"] == "MODIFY"
    assert route_query({**state, **classified}) == "planner"


def test_dataset_based_machine_learning_recommendation_routes_to_advice():
    state = {"messages": [HumanMessage(content="Best machine learning models to train on this dataset")],
             "dataset_schema": "target: Churn; numeric and categorical features"}

    class FakeLLM:
        def invoke(self, messages):
            prompt = messages[0].content.lower()
            assert "machine-learning model families" in prompt
            assert "asks for a recommendation" in prompt
            return AIMessage(content="ADVISE")

    with patch("src.agents.graph.llm", FakeLLM()):
        classified = query_node(state)
    assert classified["intent"] == "ADVISE"
    assert route_query({**state, **classified}) == "direct"


def test_metric_comparisons_cannot_be_misrouted_as_dataset_advice():
    questions = [
        "Does churn differ between the East and West regions?",
        "Which payment methods have the highest churn rates?",
    ]
    frame = pd.DataFrame({"churned": [False, True], "region": ["East", "West"],
                          "payment_method": ["card", "invoice"]})
    for question in questions:
        state = {"messages": [HumanMessage(content=question)],
                 "dataset_schema": "churned: boolean churn outcome", "dataframe": frame}
        fake_llm = type("FakeLLM", (), {"invoke": lambda self, _messages: AIMessage(content="ADVISE")})()
        with patch("src.agents.graph.llm", fake_llm):
            classified = query_node(state)
        assert classified["intent"] == "ANALYZE"


def test_dataset_advice_uses_schema_without_executor():
    state = {"intent": "ADVISE", "dataset_schema": "text: string; label: category",
             "messages": [HumanMessage(content="Which models should I try?")]}

    class FakeLLM:
        def invoke(self, messages):
            assert "text: string; label: category" in messages[0].content
            return AIMessage(content="Start with a TF-IDF logistic regression baseline.")

    with patch("src.agents.graph.llm", FakeLLM()):
        result = direct_node(state)
    assert "TF-IDF" in result["explanation"]


def test_classifier_failure_does_not_trigger_analysis():
    state = {"messages": [HumanMessage(content="hello")], "dataset_schema": "Age: int"}
    fake_llm = type("FailingLLM", (), {"invoke": lambda self, _messages: (_ for _ in ()).throw(
        RuntimeError("offline")
    )})()
    with patch("src.agents.graph.llm", fake_llm):
        classified = query_node(state)
    assert classified["intent"] == "ERROR"
    assert route_query({**state, **classified}) == "direct"


def test_empty_classifier_response_is_handled_as_service_error():
    state = {"messages": [HumanMessage(content="hello")], "dataset_schema": "Age: int"}
    fake_llm = type("EmptyLLM", (), {"invoke": lambda self, _messages: AIMessage(content="   ")})()
    with patch("src.agents.graph.llm", fake_llm):
        classified = query_node(state)
    assert classified["intent"] == "ERROR"
    assert route_query({**state, **classified}) == "direct"


def test_churn_query_asks_for_ambiguous_target_definition():
    frame = pd.DataFrame({"Churn": ["Yes", "No"],
                          "Status": ["Active", "Inactive"],
                          "Plan": ["monthly", "annual"]})
    state = {"messages": [HumanMessage(content="Compare churn by plan")],
             "dataset_schema": "Churn, Status, Plan", "dataframe": frame}
    fake_llm = type("FakeLLM", (), {"invoke": lambda self, _messages: AIMessage(content="ANALYZE")})()
    with patch("src.agents.graph.llm", fake_llm):
        classified = query_node(state)
    assert classified["intent"] == "CLARIFY"
    assert "Churn" in classified["clarification"]
    result = direct_node({**state, **classified})
    assert result["explanation"] == classified["clarification"]


def test_churn_query_uses_explicit_data_dictionary_definition():
    frame = pd.DataFrame({"churn_date": [None, "2026-01-01"],
                          "churned": [False, True],
                          "plan": ["monthly", "annual"]})
    schema = (
        "churn_date: cancellation date for churned customers; blank for active customers.\n"
        "churned: current binary churn outcome (true means churned).\n"
        "plan: subscription term."
    )
    state = {"messages": [HumanMessage(content="What percentage of customers churned?")],
             "dataset_schema": schema, "dataframe": frame}
    fake_llm = type("FakeLLM", (), {"invoke": lambda self, _messages: AIMessage(content="ANALYZE")})()
    with patch("src.agents.graph.llm", fake_llm):
        classified = query_node(state)
    assert classified["intent"] == "ANALYZE"
    assert classified["clarification"] == ""


def test_churn_clarification_reply_resumes_original_request():
    frame = pd.DataFrame({"ChurnStatus": ["Yes", "No"], "Plan": ["monthly", "annual"]})
    state = {
        "messages": [HumanMessage(content="Compare churn by plan"),
                     AIMessage(content="Which column defines churn?"),
                     HumanMessage(content="ChurnStatus; Yes means churn.")],
        "dataset_schema": "ChurnStatus, Plan", "dataframe": frame,
        "clarification_context": "Compare churn by plan",
    }

    class FakeLLM:
        def invoke(self, messages):
            assert "Original data request: Compare churn by plan" in messages[1].content
            return AIMessage(content="ANALYZE")

    with patch("src.agents.graph.llm", FakeLLM()):
        classified = query_node(state)
    assert classified["intent"] == "ANALYZE"
    assert classified["clarification"] == ""


def test_result_values_are_not_sent_for_summary_without_opt_in():
    fake_llm = type("FailingLLM", (), {"invoke": lambda self, _messages: (_ for _ in ()).throw(
        AssertionError("LLM must not receive result values without consent")
    )})()
    state = {"share_values": False, "code_output": "secret result"}
    with patch("src.agents.graph.llm", fake_llm):
        result = explainer_node(state)
    assert "secret result" in result["explanation"]


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
