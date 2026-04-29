from langgraph.graph import StateGraph, END

# NOTE: These imports assume Person 2's agents module is accessible.
# When integrating, ensure all three person folders are in the same directory
# or installed as packages, then adjust imports accordingly.
from memory import DebateState
from agents import pro_agent, con_agent
from judge import judge_agent

def should_continue(state: DebateState):
    """Conditional edge logic: checks if max rounds are reached."""
    if state["round_count"] >= state["max_rounds"]:
        return "judge_agent"
    return "pro_agent"

def build_debate_graph():
    # 1. Initialize Graph
    workflow = StateGraph(DebateState)
    
    # 2. Add Nodes
    workflow.add_node("pro_agent", pro_agent)
    workflow.add_node("con_agent", con_agent)
    workflow.add_node("judge_agent", judge_agent)
    
    # 3. Define the Flow (Edges)
    workflow.set_entry_point("pro_agent")
    
    # Pro always goes to Con
    workflow.add_edge("pro_agent", "con_agent")
    
    # Con goes to either Pro (next round) or Judge (if rounds finished)
    workflow.add_conditional_edges(
        "con_agent",
        should_continue,
        {
            "pro_agent": "pro_agent",
            "judge_agent": "judge_agent"
        }
    )
    
    # Judge ends the debate
    workflow.add_edge("judge_agent", END)
    
    return workflow.compile()
