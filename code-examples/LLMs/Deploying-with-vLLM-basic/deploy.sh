pip install --upgrade vllm Jinja2 jsonschema
export PATH="$PATH:$HOME/.local/bin"
export HF_TOKEN='YOUR TOKEN'
vllm serve meta-llama/Llama-2-7b-chat-hf --dtype half --max-model-len 200