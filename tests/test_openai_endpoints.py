import os
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from src.app.config import settings

api_key = settings.AZURE_OPENAI_API_KEY_SUMMARY
azure_endpoint = str(settings.AZURE_OPENAI_ENDPOINT_SUMMARY)
api_version = settings.AZURE_OPENAI_API_VERSION_SUMMARY
model = settings.AZURE_OPENAI_DEPLOYMENT_SUMMARY

# print(f"This is the api key: {api_key}")

# url = "https://my-oai-instance.openai.azure.com/openai/deployments?api-version=2024-05-01-preview"
# headers = {"api-key": os.getenv("AZURE_OPENAI_API_KEY")}
# print(requests.get(url, headers=headers).json())

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=azure_endpoint,
    api_key=api_key
)

response = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": "I am going to Paris, what should I see?",
        }
    ],
    max_completion_tokens=40000,
    model=model
)

print(response.choices[0].message.content)