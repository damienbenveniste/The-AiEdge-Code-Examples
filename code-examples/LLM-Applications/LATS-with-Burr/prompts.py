
reflection_system_prompt = {
    "content": """
    Analyze the trajectories of a solution to a question answering task. The trajectories are a series of actions and observations with a textual final answer at the end.
    The actions are web search input and the observations are the resulting list of urls with their text content.
                
    Given a question and a trajectory, evaluate its correctness and provide your reasoning and analysis in detail. 
    Focus on the latest thought, action, and observation. 
    Incomplete trajectories can be correct if the actions and observations can be useful to answer the question, even if the answer is not found yet. 
    Then conclude on the correctness score between 0 and 10.
    """,
    "role": "system"
}

answer_system_prompt = {
    "content": "You are an AI assistant.",
    "role": "system"
}

