from tree import Node
from tools import search
from chains import answer_chain, reflection_chain
from multiprocessing import Pool, cpu_count
from burr.core import action, State
from reflection import Reflection


@action(reads=[], writes=["input", "root"])
def initialize(state: State, input: str) -> State:
    root_reflection = Reflection(
        reflections='', 
        score=0, 
        found_solution=False
    )
    root = Node(messages=[], reflection=root_reflection)
    return state.update(
        input=input,
        root=root
    )


@action(reads=["input", "root"], writes=["root"])
def expand(state: State) -> State:
    # we get the best node to expand and its current trajectory
    root = state['root']
    best_candidate = root.select_best()
    messages = best_candidate.get_trajectory()

    # we get new candidates
    args_list = [(state['input'], messages)] * 5
    with Pool(cpu_count()) as pool:
        new_candidates = pool.starmap(answer_chain, args_list)

    # we call the tools if there are tool calls
    with Pool(cpu_count()) as pool:
        tool_responses = pool.map(search, new_candidates)

    # we get the messages depending on if they are tool calls or not
    output_messages = []
    for candidate, responses in zip(new_candidates, tool_responses):
        output_message = [{
            'role': 'assistant',
            'content': candidate.content,
            'tool_calls': candidate.tool_calls
        }]
        if responses:
            for tool_response in responses:
                output_message.append({
                    'role': 'tool',
                    'content': str(tool_response[0]),
                    'tool_call_id': tool_response[1]
                })
        output_messages.append(output_message)

    # Reflect on each candidate
    args = [(state['input'], msges) for msges in output_messages]

    with Pool(cpu_count()) as pool:
        reflections = pool.starmap(reflection_chain, args)

    # Grow tree
    child_nodes = [
        Node(output_message, parent=best_candidate, reflection=reflection)
        for output_message, reflection in zip(output_messages, reflections)
    ]
    best_candidate.children.extend(child_nodes)
    return state.update(root=root)


@action(reads=["root"], writes=["solution"])
def get_best_solution(state: State):
    root = state["root"]
    best_solution = root.select_best_solution()
    return state.update(solution=best_solution)
