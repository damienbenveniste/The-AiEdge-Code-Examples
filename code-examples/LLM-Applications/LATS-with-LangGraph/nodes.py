from tree import Node
from typing_extensions import TypedDict
from tools import tool_node
from reflection import Reflection
from chains import answer_chain, reflection_chain


class TreeState(TypedDict):
    root: Node
    input: str
    solution: Node


def initialize(state: TreeState) -> dict:
    root_reflection = Reflection(
        reflections='', 
        score=0, 
        found_solution=False
    )
    root = Node(messages=[], reflection=root_reflection)
    return {"root": root}


def expand(state: TreeState) -> dict:
    # we start from the root
    root = state['root']
    # we select the best leaf node
    best_candidate = root.select_best()
    # we get all the messages from the root
    messages = best_candidate.get_trajectory()
    # Generate N candidates from the single child candidate
    new_candidates = answer_chain.batch([
        {"input": state["input"], "messages": messages}
    ] * 5)
    # we get the tool responses from the actions
    tool_responses = tool_node.batch([
        {"messages": [candidate]} 
        for candidate in new_candidates
    ])
    # we append the tool response to the action
    output_messages = []
    for candidate, response in zip(new_candidates, tool_responses):
        output_message = [candidate]
        output_message.extend(response['messages'])
        output_messages.append(output_message)

    # Reflect on each candidate
    reflections = reflection_chain.batch([
        {"input": state["input"], "candidate": msges} 
        for msges in output_messages  
    ])

    # Grow the tree
    child_nodes = [
        Node(
            output_message, 
            parent=best_candidate, 
            reflection=reflection
        )
        for output_message, reflection 
        in zip(output_messages, reflections)
    ]
    best_candidate.children.extend(child_nodes)
    return {"root": root}


def get_best_solution(state: TreeState):
    root = state["root"]
    best_solution = root.select_best_solution()
    return {"solution": best_solution}


def should_continue(state: TreeState):
    """Determine whether to continue the tree search."""
    root = state["root"]
    if root.is_solved or root.height > 5:
        return 'terminate'
    return "expand"