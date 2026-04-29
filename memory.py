from typing import TypedDict, Annotated, List
import operator

class DebateState(TypedDict):
    topic: str
    max_rounds: int
    round_count: int
    # Annotated with operator.add means elements returned will be appended to the list
    history: Annotated[List[str], operator.add] 
    winner: str
    judge_reason: str
