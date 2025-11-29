from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

model_id = "Davlan/naija-twitter-sentiment-afriberta-large"
tokenizer = AutoTokenizer.from_pretrained(model_id)

ort_model = ORTModelForSequenceClassification.from_pretrained(model_id, export=True)
ort_model.save_pretrained("./onnx_model")
tokenizer.save_pretrained("./onnx_model")

print("✅ ONNX model exported successfully.")
