import requests

# Replace with your API key
TOKEN = "ABSKQmVkcm9ja0FQSUtleS1jZDk1LWF0LTU5MDE4Mzg5MTYxODphZXlpNVAwaHVmRS9ESXkyNng2R0RJakRtU3lWUWV4YlVWQVNSMWZGUDVMZW5qMVdSYjlaR09leW81RT0="

URL = "https://bedrock-runtime.us-west-2.amazonaws.com/model/global.anthropic.claude-sonnet-4-5-20250929-v1:0/invoke"

headers = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}

body = {
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 50,
    "messages": [{"role": "user", "content": "Say hello"}],
}

response = requests.post(URL, headers=headers, json=body, timeout=30)
print(f"Status: {response.status_code}")
print(f"Response: {response.text}")
