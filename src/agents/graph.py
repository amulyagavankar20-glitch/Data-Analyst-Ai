import os
import io
import contextlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from src.agents.utils import AgentState
from dotenv import load_dotenv

load_dotenv()

# Initialize LLM
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    # Fallback placeholder to allow import without crashing, 
    # but app.py should have already stopped execution.
    api_key = "MISSING"

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=api_key,
    temperature=0
)

# --- Node Definitions ---

def query_node(state: AgentState):
    """Parses user intent."""
    if state.get('dataset_schema'):
        return {"messages": [SystemMessage(content="Intent detected: Analysis")]}
    return {"messages": [SystemMessage(content="Intent detected: Chit-chat")]}

def planner_node(state: AgentState):
    """Creates a step-by-step plan."""
    schema = state['dataset_schema']
    user_query = ""
    for m in reversed(state['messages']):
        if isinstance(m, HumanMessage):
            user_query = m.content
            break
            
    system_prompt = f"""You are a Lead Data Scientist.
    Dataset Schema:
    {schema}
    
    User Question: "{user_query}"
    
    Your Goal: Create a precise plan to answer the question using Python (pandas/matplotlib/seaborn).
    
    IMPORTANT Rules:
    1. The plan MUST consist of short, executable actions (e.g., "Calculate correlation", "Plot histogram").
    2. Do NOT explain "how" to do it theoretically. Just list the steps.
    3. If the user asks "What is this data?", the plan should be: "1. Print head, 2. Print info, 3. Describe columns".
    4. Do not include standard markdown or conversational filler.
    """
    
    msg = [SystemMessage(content=system_prompt), HumanMessage(content=user_query)]
    response = llm.invoke(msg)
    plan_text = response.content
    plan = [line.strip() for line in plan_text.split('\n') if line.strip()]
    
    # Initialize retry count
    return {"plan": plan, "messages": [SystemMessage(content=f"Plan generated.")], "retry_count": 0, "error": None}

def codegen_node(state: AgentState):
    """Writes code."""
    plan = state['plan']
    schema = state['dataset_schema']
    error = state.get('error')
    
    plan_str = "\n".join(plan)
    
    system_prompt = f"""You are a Python Data Science Expert.
    You have a pandas DataFrame named `df` loaded.
    Schema: {schema}
    
    Plan:
    {plan_str}
    
    Task: Write Python code.
    - Use `df` directly.
    - MODIFY `df` in place if cleaning data (e.g. `df.dropna(inplace=True)` or `df = df.dropna()`).
    - Use `plt.figure()` for plots.
    - Define `result` variable for text outputs.
    - ALWAYS `print()` the key results or `df.head()` so the output is not empty.
    - WRAP CODE IN MARKDOWN ```python ... ```
    """
    
    if error:
        system_prompt += f"\n\nNOTE: The previous code failed with this error:\n{error}\nPlease FIX the code."
    
    response = llm.invoke([SystemMessage(content=system_prompt)])
    content = response.content
    
    if "```python" in content:
        code = content.split("```python")[1].split("```")[0].strip()
    elif "```" in content:
        code = content.split("```")[1].split("```")[0].strip()
    else:
        code = content
        
    return {"code": code, "messages": [SystemMessage(content="Code generated.")]}

def executor_node(state: AgentState):
    """Executes code and captures global df updates."""
    code = state['code']
    df = state['dataframe']
    retry_count = state.get("retry_count", 0)
    
    plt.clf()
    buffer = io.StringIO()
    
    # Pass 'df' in, and we'll check if it changed
    local_scope = {'df': df, 'pd': pd, 'plt': plt, 'sns': sns}
    
    try:
        with contextlib.redirect_stdout(buffer):
            exec(code, {}, local_scope)
            
        output = buffer.getvalue()
        
        # Capture updated DataFrame
        new_df = local_scope.get('df')
        
        if 'result' in local_scope:
            output += f"\nResult Variable: {local_scope['result']}"
            
        figures = []
        if plt.get_fignums():
             figures = [plt.gcf()]
             output += "\n[Plot Generated Successfully]"
        
        # Success - clear error
        return {
            "code_output": output, 
            "figures": figures, 
            "dataframe": new_df, # Update global state
            "error": None,
            "messages": [SystemMessage(content="Code executed successfully.")]
        }
        
    except Exception as e:
        return {
            "code_output": f"Error: {e}", 
            "error": str(e),
            "retry_count": retry_count + 1,
            "messages": [SystemMessage(content=f"Execution Error: {e}")]
        }

def viz_node(state: AgentState):
    return {"messages": [SystemMessage(content="Visualization pass completed.")]}

def explainer_node(state: AgentState):
    output = state.get('code_output', 'No output')
    plan = state.get('plan', [])
    
    user_query = ""
    for m in reversed(state['messages']):
        if isinstance(m, HumanMessage):
            user_query = m.content
            break

    system_prompt = f"""You are a Data Analyst.
    Question: "{user_query}"
    Output: {output}
    Explain findings clearly.
    """
    response = llm.invoke([SystemMessage(content=system_prompt)])
    return {"explanation": response.content, "messages": [SystemMessage(content="Explanation generated.")]}

# --- Graph ---

workflow = StateGraph(AgentState)

workflow.add_node("query_agent", query_node)
workflow.add_node("planner_agent", planner_node)
workflow.add_node("codegen_agent", codegen_node)
workflow.add_node("executor_agent", executor_node)
workflow.add_node("viz_agent", viz_node)
workflow.add_node("explainer_agent", explainer_node)

workflow.set_entry_point("query_agent")

def route_query(state: AgentState):
    if state.get('dataset_schema'):
        return "planner_agent"
    return END

def route_error(state: AgentState):
    if state.get("error") and state.get("retry_count", 0) < 3:
        return "codegen_retry"
    return "success"

workflow.add_conditional_edges("query_agent", route_query, {"planner_agent": "planner_agent", END: END})

workflow.add_edge("planner_agent", "codegen_agent")
workflow.add_edge("codegen_agent", "executor_agent")

# Loop back on error
workflow.add_conditional_edges(
    "executor_agent",
    route_error,
    {
        "codegen_retry": "codegen_agent",
        "success": "viz_agent"
    }
)

workflow.add_edge("viz_agent", "explainer_agent")
workflow.add_edge("explainer_agent", END)

app = workflow.compile()
