import os
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential

endpoint = "https://my-oai-instance.cognitiveservices.azure.com/"
model_name = "text-embedding-3-large"
deployment = "sentiment-scope-vectors"

api_version = "2024-02-01"

client = AzureOpenAI(
    api_version="2024-12-01-preview",
    endpoint=endpoint,
    credential=AzureKeyCredential("YOUR_AZURE_KEY_HERE")
)

response = client.embeddings.create(
    input=["first phrase","second phrase","third phrase"],
    model=deployment
)

for item in response.data:
    length = len(item.embedding)
    print(
        f"data[{item.index}]: length={length}, "
        f"[{item.embedding[0]}, {item.embedding[1]}, "
        f"..., {item.embedding[length-2]}, {item.embedding[length-1]}]"
    )
print(response.usage)