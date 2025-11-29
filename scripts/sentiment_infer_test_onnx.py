"""
Optimized Sentiment Benchmark Script for ONNXRuntime
----------------------------------------------------
Runs large-scale WhatsApp sentiment analysis with performance metrics and reporting.
Designed for 300k+ messages and Docker deployment.
"""

import os
import csv
import random
import time
import statistics
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional

from transformers import pipeline, AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification

from src.app.utils.raw_txt_parser import CleanedMessage, WhatsAppChatParser
from src.app.utils.pre_process import clean_messages


# ============================================================
# MODEL LOADING
# ============================================================

def load_model(model_path: str = "./onnx_model", device: str = "cpu"):
    """Load ONNX model and tokenizer for inference."""
    print(f"🚀 Loading ONNX model from {model_path} on {device.upper()} ...")

    ort_model = ORTModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    sentiment_pipeline = pipeline(
        "text-classification",
        model=ort_model,
        tokenizer=tokenizer,
        top_k=None,
        device=-1 if device == "cpu" else 0,  # ONNX CPU or GPU (if available)
        truncation=True,
        max_length=512
    )

    print("✅ Model loaded successfully.")
    return sentiment_pipeline


# ============================================================
# BENCHMARK CORE
# ============================================================

def run_benchmark(messages: List[CleanedMessage],
                  model_path="./onnx_model",
                  batch_size=128,
                  report_dir="./reports",
                  device="cpu",
                  high_conf_threshold=0.7):
    """
    Run benchmark and save performance + sentiment reports.
    Suitable for very large datasets (300k+ messages).
    """

    os.makedirs(report_dir, exist_ok=True)
    pipe = load_model(model_path, device=device)

    # --- Preprocess text data ---
    texts = [msg.text.strip() for msg in messages if msg.text and msg.text.strip()]
    total_msgs = len(texts)
    if total_msgs == 0:
        raise ValueError("No valid text messages found.")

    print(f"\n🧠 Starting inference on {total_msgs:,} messages (batch={batch_size})")

    all_results = []
    batch_times = []
    start_time = time.perf_counter()

    # --- Inference in batches ---
    for i in range(0, total_msgs, batch_size):
        batch = texts[i:i + batch_size]

        bstart = time.perf_counter()
        results = pipe(batch, batch_size=batch_size)
        bend = time.perf_counter()

        batch_times.append(bend - bstart)
        all_results.extend(results)

        if (i // batch_size + 1) % 20 == 0 or i + batch_size >= total_msgs:
            print(f"Processed {i + len(batch):,}/{total_msgs:,} messages...")

    end_time = time.perf_counter()

    # ============================================================
    # PERFORMANCE METRICS
    # ============================================================
    total_time = end_time - start_time
    avg_batch_time = statistics.mean(batch_times)
    messages_per_sec = total_msgs / total_time
    latency_per_msg = total_time / total_msgs
    num_batches = len(batch_times)

    # ============================================================
    # RESULTS AGGREGATION
    # ============================================================
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    txt_filename = os.path.join(report_dir, f"sentiment_benchmark_{timestamp}.txt")
    csv_filename = os.path.join(report_dir, f"sentiment_results_{timestamp}.csv")

    label_counts = {}
    high_conf, low_conf = 0, 0

    for res in all_results:
        pred = res[0]
        label_counts[pred["label"]] = label_counts.get(pred["label"], 0) + 1
        if pred["score"] >= high_conf_threshold:
            high_conf += 1
        else:
            low_conf += 1

    # ============================================================
    # SAVE CSV RESULTS
    # ============================================================
    with open(csv_filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "timestamp", "sender", "text", "label", "confidence"])
        for i, (msg, res) in enumerate(zip(messages, all_results)):
            pred = res[0]
            writer.writerow([
                i + 1,
                msg.timestamp.isoformat() if msg.timestamp else "",
                msg.sender or "",
                msg.text,
                pred["label"],
                round(pred["score"], 4)
            ])

    # ============================================================
    # RANDOM SAMPLE EXTRACTION
    # ============================================================
    samples = {}
    for label in label_counts.keys():
        candidates = [m.text for m, r in zip(messages, all_results)
                      if r[0]["label"] == label and m.text.strip()]
        random.shuffle(candidates)
        samples[label] = candidates[:min(20, len(candidates))]

    # ============================================================
    # WRITE TEXT REPORT
    # ============================================================
    with open(txt_filename, "w", encoding="utf-8") as f:
        f.write(f"TITLE: WhatsApp Sentiment Benchmark Report\n")
        f.write(f"Generated At: {datetime.now().isoformat()}\n\n")
        f.write(f"Batch Size: {batch_size}\nModel: {model_path}\nDevice: {device.upper()}/ONNXRuntime\n\n")

        f.write(f"1. Time taken to perform analysis: {total_time:.2f} s\n")
        f.write(f"2. Messages analyzed: {total_msgs:,}\n\n")

        f.write("=== PERFORMANCE ===\n")
        f.write(f"Total wall time (s): {total_time:.2f}\n")
        f.write(f"Average batch time (s): {avg_batch_time:.3f}\n")
        f.write(f"Messages / second: {messages_per_sec:.2f}\n")
        f.write(f"Average latency per message (s): {latency_per_msg:.5f}\n")
        f.write(f"Number of batches: {num_batches}\n\n")

        f.write("=== LABEL DISTRIBUTION ===\n")
        for label, count in label_counts.items():
            pct = (count / total_msgs) * 100
            f.write(f"{label}: {count:,} ({pct:.2f}%)\n")

        f.write(f"\nHigh-confidence threshold: confidence >= {high_conf_threshold}\n")
        f.write(f"High-confidence count: {high_conf:,}\n")
        f.write(f"Low-confidence count: {low_conf:,}\n\n")

        f.write("=== SAMPLES FOR MANUAL ASSESSMENT (n=20 per label) ===\n\n")
        for label, examples in samples.items():
            f.write(f"\n{label.upper()}:\n")
            for i, text in enumerate(examples, start=1):
                f.write(f"{i}. {text[:200]}...\n")

    print(f"\n✅ Benchmark complete!")
    print(f"📊 Report saved: {txt_filename}")
    print(f"📁 CSV results: {csv_filename}")


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    # Example: Parse WhatsApp chat text
    parser = WhatsAppChatParser(dayfirst=True)
    file_path = "C:\\Users\\user\\Downloads\\WhatsApp Chat with SUPER EAGLES🤡🤡\\WhatsApp Chat with SUPER EAGLES🤡🤡.txt"

    print(f"📂 Parsing chat file: {file_path}")
    messages = parser.parse_file(file_path)
    print(f"Parsed {len(messages):,} raw messages.")

    cleaned_messages = clean_messages(messages)
    print(f"Cleaned {len(cleaned_messages):,} valid text messages.")

    # Run benchmark
    run_benchmark(
        cleaned_messages,
        batch_size=16,
        model_path="./onnx_model_optimized",
        device="cpu"
    )
