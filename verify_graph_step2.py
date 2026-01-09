import os
import sys

# Set dummy key to avoid init errors if validation happens aggressively
os.environ["GROQ_API_KEY"] = "gsk_dummy_key_for_testing_structure"

# Add project root to path
sys.path.append(os.getcwd())

from src.agents.graph import app
from langchain_core.messages import HumanMessage

print("Graph compiled successfully.")

# Test State
initial_state = {
    "messages": [HumanMessage(content="What is the average age?")],
    "dataset_schema": "Age: int, Name: str",
    "plan": [],
    "code": "",
    "code_output": "",
    "explanation": "",
    "figures": []
}

print("Invoking graph...")
try:
    result = app.invoke(initial_state)
    print("Graph execution completed.")
    print("Final Explanation:", result.get("explanation"))
    print("Messages Log:", [m.content for m in result['messages']])
except Exception as e:
    print(f"Graph execution failed: {e}")
