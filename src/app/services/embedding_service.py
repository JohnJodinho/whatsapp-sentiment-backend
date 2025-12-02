
import logging

from typing import List

from openai import AsyncAzureOpenAI
from src.app.config import settings 




log = logging.getLogger(__name__)

client = AsyncAzureOpenAI(
    api_version=settings.AZURE_OPENAI_API_VERSION,
    azure_endpoint=str(settings.AZURE_OPENAI_ENDPOINT),
    api_key=settings.AZURE_OPENAI_API_KEY
)

OPENAI_BATCH_SIZE = 64


async def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Calls the Azure OpenAI Embedding API in batches.
    """
    results: List[List[float]] = []
    
    for i in range(0, len(texts), OPENAI_BATCH_SIZE):
        batch = texts[i:i + OPENAI_BATCH_SIZE]
        try:
            if len(texts) == 1:
                log.info(f"Sending {texts[0]} to Azure OpenAI for embedding...")
                print(f"Sending {texts[0]} to Azure OpenAI for embedding...")
            else:
                log.info(f"Sending batch of {len(batch)} texts to Azure OpenAI for embedding...")
                print(f"Sending batch of {len(batch)} texts to Azure OpenAI for embedding...")
            
            resp = await client.embeddings.create(
                model=settings.AZURE_OPENAI_DEPLOYMENT, 
                input=batch
            )
            
            batch_embs = [r.embedding for r in resp.data]
            results.extend(batch_embs)
            
        except Exception    as e:
            log.error(f"Azure OpenAI API call failed: {e}")
            results.extend([[]] * len(batch)) 
            
    return results


embed_query = embed_texts