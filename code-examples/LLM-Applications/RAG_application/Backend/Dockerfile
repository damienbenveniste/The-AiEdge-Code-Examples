FROM python:3.12

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Expose the secret SECRET_EXAMPLE at buildtime and use its value as git remote URL

COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

RUN --mount=type=secret,id=PINECONE_API_KEY,mode=0444,required=true
RUN --mount=type=secret,id=OPENAI_API_KEY,mode=0444,required=true
RUN --mount=type=secret,id=HF_TOKEN,mode=0444,required=true 

COPY --chown=user ./app /app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]