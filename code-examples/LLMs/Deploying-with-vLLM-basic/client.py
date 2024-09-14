from openai import OpenAI

openai_api_base = "http://ec2-18-144-41-161.us-west-1.compute.amazonaws.com:8000/v1"

client = OpenAI(
    api_key='none',
    base_url=openai_api_base,
)

stream = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-chat-hf",
    messages=[
        {"role": "user", "content": "What is the history of machine learning?"}
    ],
    stream=True,
    temperature=0.7,
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")