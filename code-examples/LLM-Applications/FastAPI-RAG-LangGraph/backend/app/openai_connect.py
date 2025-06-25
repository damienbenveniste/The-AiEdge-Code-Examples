from dotenv import load_dotenv
import os
from openai import OpenAI, AsyncOpenAI

load_dotenv()

api_key = os.environ['OPENAI_API_KEY']

openai_client = OpenAI(api_key=api_key)
async_openai_client = AsyncOpenAI(api_key=api_key)