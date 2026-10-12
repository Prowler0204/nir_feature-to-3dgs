from openai import OpenAI

# 1. Initialize the client pointing to NVIDIA's servers
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1/",
    api_key="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"
)

# 2. Have Python read your local file
with open("sample.txt", "r") as f:
    local_file_content = f.read()

# 3. Switch to the 8B model
completion = client.chat.completions.create(
    model="meta/llama-3.1-70b-instruct", 
    messages=[
        {"role": "system", "content": "You are a helpful local assistant."},
        {"role": "user", "content": f"Here is the content of my local file: '{local_file_content}'. Can you rewrite this sentence to sound more poetic?"}
    ],
    temperature=0.2,
    max_tokens=1024
)

print(completion.choices[0].message.content)