import asyncio
import os
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics.aio import TextAnalyticsClient

# -------------------------------------------------------------------------
# CONFIGURATION
# Best Practice: Use environment variables in production
# -------------------------------------------------------------------------
ENDPOINT = os.getenv("AZURE_LANGUAGE_ENDPOINT", "https://johns-text-analytics-api.cognitiveservices.azure.com/")
KEY = os.getenv("AZURE_LANGUAGE_KEY", "YOUR_AZURE_KEY_HERE")



DOCUMENTS = [
    {
        "id": "1", 
        "language": "en", 
        "text": "The user interface is fantastic and very intuitive. I love the new dark mode."
    },
    {
        "id": "2", 
        "language": "en", 
        "text": "The service was terrible. The waiter was rude, but the food was actually quite good."
    },
    {
        "id": "3", 
        "language": "es", 
        "text": "Estoy muy feliz con el resultado. Todo salió perfecto." 
    }
]

async def analyze_sentiment_async():
    # create credential object
    credential = AzureKeyCredential(KEY)
    
    # create the async client
    # We use 'async with' to ensure the client session is closed automatically
    async with TextAnalyticsClient(endpoint=ENDPOINT, credential=credential) as client:
        
        print(f"Analyzing {len(DOCUMENTS)} documents asynchronously...\n")
        
        # Call the API
        # show_opinion_mining=True gives you granular targets (e.g., "waiter" -> negative)
        results = await client.analyze_sentiment(
            documents=DOCUMENTS, 
            show_opinion_mining=True
        )

        # Process results
        # The result list matches the order of the input documents
        for idx, result in enumerate(results):
            doc_id = DOCUMENTS[idx]["id"]
            
            if result.is_error:
                print(f"Document ID: {doc_id} - Error: {result.error.code} - {result.error.message}")
                continue

            print(f"--- Document ID: {doc_id} ---")
            print(f"Overall Sentiment: {result.sentiment.upper()}")
            print(f"Scores: Positive={result.confidence_scores.positive:.2f}, "
                  f"Neutral={result.confidence_scores.neutral:.2f}, "
                  f"Negative={result.confidence_scores.negative:.2f}")

            # Display Sentence-level analysis
            for sentence in result.sentences:
                print(f"  Sentence: \"{sentence.text}\"")
                print(f"  Sentence Sentiment: {sentence.sentiment}")
                
                # Display Opinion Mining (Aspect-Based) results if available
                if sentence.mined_opinions:
                    print("    Opinions detected:")
                    for mined_opinion in sentence.mined_opinions:
                        target = mined_opinion.target
                        print(f"      Target: '{target.text}' -> {target.sentiment.upper()} (Confidence: {target.confidence_scores.positive if target.sentiment == 'positive' else target.confidence_scores.negative:.2f})")
                        for assessment in mined_opinion.assessments:
                            print(f"        Reason: '{assessment.text}'")
            print("\n")

if __name__ == "__main__":
    # Run the async main loop
    asyncio.run(analyze_sentiment_async())