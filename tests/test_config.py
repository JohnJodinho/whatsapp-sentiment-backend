from src.app.config import settings

def test_config_settings():
    assert settings.PROJECT_NAME == "SentimentScope API"
    assert settings.CHROMA_MODE in ["local", "cloud"]
    assert settings.GROQ_MODEL_PRIMARY == "openai/gpt-oss-120b"
    assert settings.SENTIMENT_MODEL_REPO == "JohnAlbarkaIbrahim/afroxlmr-mini-nigerian-sentiment"
    assert settings.EMBEDDING_MODEL_REPO == "Davlan/afro-xlmr-mini"