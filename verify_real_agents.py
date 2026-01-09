import os
import sys
import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

# Load env for API key
load_dotenv()

# Add project root to path
sys.path.append(os.getcwd())

from src.agents.graph import app

print("Initializing verification...")

# Create dummy dataframe
df = pd.DataFrame({
    'Age': [22, 35, 58, 25, 45],
    'Name': ['A', 'B', 'C', 'D', 'E'],
    'Fare': [7.25, 71.0, 8.05, 53.0, 8.05]
})

schema_info = f"""
Columns: {list(df.columns)}
Types: {df.dtypes.to_dict()}
"""

initial_state = {
    "messages": [HumanMessage(content="What is the average Age and Fare?")],
    "dataset_schema": schema_info,
    "dataframe": df,
    "plan": [],
    "code": "",
    "code_output": "",
    "explanation": "",
    "figures": []
}

print("Invoking graph with REAL LLM calls (Groq)...")
try:
    result = app.invoke(initial_state)
    print("Graph execution completed.")
    print("-" * 20)
    print("Plan:", result['plan'])
    print("-" * 20)
    print("Code:", result['code'])
    print("-" * 20)
    print("Output:", result['code_output'])
    print("-" * 20)
    print("Explanation:", result['explanation'])
    print("-" * 20)
    if result['figures']:
        print(f"Captured {len(result['figures'])} figures.")
except Exception as e:
    print(f"Graph execution failed: {e}")
