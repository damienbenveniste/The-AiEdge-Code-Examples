from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
Analyze the trajectories of a solution to a question answering task. The trajectories are a series of actions and observations with a textual final answer at the end.
The actions are web search input and the observations are the resulting list of urls with their text content.
             
Given a question and a trajectory, evaluate its correctness and provide your reasoning and analysis in detail. 
Focus on the latest thought, action, and observation. 
Incomplete trajectories can be correct if the actions and observations can be useful to answer the question, even if the answer is not found yet. 
Then conclude on the correctness score between 0 and 10.
""",
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="candidate"),
    ]
)


answer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an AI assistant."
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="messages", optional=True),
    ]
)