import os
import shutil
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoTokenizer

# --- Configuration ---
MODEL_ID = "Davlan/naija-twitter-sentiment-afriberta-large"
SAVE_PATH = "./onnx_model_optimized"

print(f"::::: Starting ONNX export for {MODEL_ID} ...")

# 1. Export Base ONNX (FP32)
# This downloads the model and converts it to generic ONNX format
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
ort_model = ORTModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    export=True,
    provider="CPUExecutionProvider"
)

# Save the base model (approx 500MB)
ort_model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)
print("::::: Base ONNX exported successfully.")

# 2. 🚀 QUANTIZE DIRECTLY (FP32 -> INT8)
# We skip 'ORTOptimizer' here because it breaks the graph for the quantizer.
# The Quantizer will handle the necessary optimizations itself.
print("::::: Running Dynamic Quantization (INT8)...")

# Load the base 'model.onnx' we just saved
quantizer = ORTQuantizer.from_pretrained(SAVE_PATH, file_name="model.onnx")

# Create Dynamic Quantization Config (AVX2 is standard for modern CPUs)
q_config = AutoQuantizationConfig.avx2(is_static=False, per_channel=True)

# This generates 'model_quantized.onnx'
quantizer.quantize(save_dir=SAVE_PATH, quantization_config=q_config)

print("::::: Quantization complete.")

# 3. Cleanup & Rename
# We replace the heavy 'model.onnx' with the light 'model_quantized.onnx'
# so your worker loads the fast one automatically.
base_model = os.path.join(SAVE_PATH, "model.onnx")
quantized_model = os.path.join(SAVE_PATH, "model_quantized.onnx")

if os.path.exists(quantized_model):
    print("::::: Replacing base model with quantized model...")
    os.remove(base_model) # Delete heavy file
    os.rename(quantized_model, base_model) # Rename quantized to default
    print(f"::::: ✅ SUCCESS! Optimized INT8 model is ready at: {SAVE_PATH}")
    print("::::: Please restart your Celery worker now.")
else:
    print("::::: ❌ Error: Quantized model file not found.")