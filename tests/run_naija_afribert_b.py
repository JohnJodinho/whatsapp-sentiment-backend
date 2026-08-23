"""
run_naija_afriberta_stageB.py

Run Davlan/naija-twitter-sentiment-afriberta-large on a list of CleanedMessage objects,
measure CPU performance, and write a single .txt report + a .csv of per-message results.

Outputs (in ./outputs/):
 - vader_naija_stageB_report_YYYYMMDD_HHMMSS.txt   <-- full textual report
 - stageB_scores_YYYYMMDD_HHMMSS.csv              <-- per-message scores + label
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any
import os
import csv
import math
import random
import time

# Hugging Face / PyTorch
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from scipy.special import softmax

from src.app.utils.raw_txt_parser import CleanedMessage, WhatsAppChatParser
from src.app.utils.pre_process import clean_messages

torch.set_num_threads(2)



# -----------------------------
# Configuration
# -----------------------------
MODEL = "Davlan/naija-twitter-sentiment-afriberta-large"
OUTPUT_DIR = "outputs"
BATCH_SIZE = 32              # adjust (larger -> fewer model calls but increases memory)
SAMPLE_PER_CATEGORY = 10     # number of example messages per category to include in report
CONFIDENCE_HIGH = 0.70       # probability threshold to consider a prediction "high-confidence"
DEVICE = torch.device("cpu") # CPU-only run

# label id2label mapping (matches model card expectation: check model if ordering differs)
id2label = {0: "positive", 1: "neutral", 2: "negative"}


# -----------------------------
# Helpers
# -----------------------------
def ensure_output_dir(path: str):
    os.makedirs(path, exist_ok=True)

def batch_iterable(iterable, size):
    for i in range(0, len(iterable), size):
        yield iterable[i:i+size]

def predict_batch(texts: List[str], tokenizer, model) -> List[Dict[str, Any]]:
    """
    Tokenize and predict a batch of texts.
    Returns list of dicts: {'text':..., 'scores': np.array([p_pos,p_neu,p_neg]), 'label':..., 'confidence':...}
    """
    # Tokenize (padding to longest in the batch)
    enc = tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.inference_mode():
        out = model(**enc)
    logits = out.logits.detach().cpu().numpy()
    probs = softmax(logits, axis=1)  # shape (batch, num_labels)
    results = []
    for i, p in enumerate(probs):
        ranking = np.argsort(p)[::-1]
        top_idx = int(ranking[0])
        label = id2label[top_idx]
        conf = float(p[top_idx])
        results.append({
            "text": texts[i],
            "scores": p,
            "label": label,
            "confidence": conf,
        })
    return results


# -----------------------------
# Main runner
# -----------------------------
def run_stageB_scoring(cleaned_messages: List[CleanedMessage]):
    ensure_output_dir(OUTPUT_DIR)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(OUTPUT_DIR, f"naija_stageB_report_{ts}.txt")
    csv_path = os.path.join(OUTPUT_DIR, f"stageB_scores_{ts}.csv")

    # Load tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    model.to(DEVICE)
    model.eval()

    total_msgs = len(cleaned_messages)
    texts = [m.text for m in cleaned_messages]

    # Warm-up on a small batch to reduce initial download overhead effects on timing (optional)
    warmup_n = min(8, total_msgs)
    if warmup_n > 0:
        _ = predict_batch(texts[:warmup_n], tokenizer, model)

    # Inference with timing
    timings = []
    results_all = []  # will store dicts per message with metadata
    start_all = time.time()
    for batch in batch_iterable(list(enumerate(texts)), BATCH_SIZE):
        # batch is list of (idx, text)
        idxs, batch_texts = zip(*batch)
        t0 = time.time()
        batch_results = predict_batch(list(batch_texts), tokenizer, model)
        t1 = time.time()
        elapsed = t1 - t0
        timings.append({"batch_size": len(batch_texts), "elapsed": elapsed})
        # attach to results with indices and original metadata
        for idx_local, pred in zip(idxs, batch_results):
            cm = cleaned_messages[idx_local]
            results_all.append({
                "index": idx_local,
                "timestamp": cm.timestamp.isoformat() if cm.timestamp is not None else "",
                "sender": cm.sender or "",
                "text": cm.text,
                "label": pred["label"],
                "confidence": pred["confidence"],
                "score_pos": float(pred["scores"][0]),
                "score_neu": float(pred["scores"][1]),
                "score_neg": float(pred["scores"][2]),
            })
    end_all = time.time()

    # Performance metrics
    total_time = end_all - start_all
    avg_batch_time = sum(t["elapsed"] for t in timings) / len(timings) if timings else 0.0
    msgs_per_second = total_msgs / total_time if total_time > 0 else float("inf")
    avg_latency_per_message = total_time / total_msgs if total_msgs else 0.0

    # Distribution of labels
    from collections import Counter, defaultdict
    counter = Counter(r["label"] for r in results_all)
    dist = {k: counter.get(k, 0) for k in ["positive", "neutral", "negative"]}

    # High-confidence vs low-confidence
    high_conf = [r for r in results_all if r["confidence"] >= CONFIDENCE_HIGH]
    low_conf = [r for r in results_all if r["confidence"] < CONFIDENCE_HIGH]

    # Sample messages for manual assessment: per category (high-confidence positive/negative/neutral + low-confidence)
    def safe_sample(lst, n):
        if not lst:
            return []
        if len(lst) <= n:
            return list(lst)
        return random.sample(lst, n)

    samples = {
        "high_pos": safe_sample([r for r in high_conf if r["label"] == "positive"], SAMPLE_PER_CATEGORY),
        "high_neu": safe_sample([r for r in high_conf if r["label"] == "neutral"], SAMPLE_PER_CATEGORY),
        "high_neg": safe_sample([r for r in high_conf if r["label"] == "negative"], SAMPLE_PER_CATEGORY),
        "low_conf": safe_sample(low_conf, SAMPLE_PER_CATEGORY),
    }

    # Write CSV of all per-message results
    with open(csv_path, "w", encoding="utf-8", newline="") as cf:
        fieldnames = ["index","timestamp","sender","text","label","confidence","score_pos","score_neu","score_neg"]
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        for r in results_all:
            writer.writerow(r)

    # Write full textual report
    with open(report_path, "w", encoding="utf-8") as rf:
        rf.write("NAIJA STAGE B SENTIMENT REPORT (Davlan/naija-twitter-sentiment-afriberta-large)\n")
        rf.write(f"Generated: {datetime.now().isoformat()}\n\n")
        rf.write("=== INPUT ===\n")
        rf.write(f"Total messages processed: {total_msgs}\n")
        rf.write(f"Batch size: {BATCH_SIZE}\n")
        rf.write(f"Model: {MODEL}\n")
        rf.write(f"Device: {DEVICE}\n\n")

        rf.write("=== PERFORMANCE ===\n")
        rf.write(f"Total wall time (s): {total_time:.4f}\n")
        rf.write(f"Average batch time (s): {avg_batch_time:.4f}\n")
        rf.write(f"Messages / second: {msgs_per_second:.2f}\n")
        rf.write(f"Average latency per message (s): {avg_latency_per_message:.4f}\n")
        rf.write(f"Number of batches: {len(timings)}\n\n")

        rf.write("=== LABEL DISTRIBUTION ===\n")
        for k, v in dist.items():
            pct = (v / total_msgs * 100) if total_msgs else 0.0
            rf.write(f"  {k:8s}: {v} ({pct:.2f}%)\n")
        rf.write("\n")

        rf.write(f"High-confidence threshold: confidence >= {CONFIDENCE_HIGH}\n")
        rf.write(f"High-confidence count: {len(high_conf)} ({len(high_conf)/total_msgs*100:.2f}%)\n")
        rf.write(f"Low-confidence count: {len(low_conf)} ({len(low_conf)/total_msgs*100:.2f}%)\n\n")

        rf.write("=== SAMPLES FOR MANUAL ASSESSMENT ===\n")
        # helper to write sample lists
        def write_sample_block(title, arr):
            rf.write(f"\n-- {title} (n={len(arr)}) --\n")
            for i, a in enumerate(arr, 1):
                txt = a["text"].replace("\n", " ")
                sender = a["sender"]
                conf = a["confidence"]
                label = a["label"]
                rf.write(f"{i:2d}. [{sender}] ({label} / {conf:.3f}) {txt[:400]}\n")

        write_sample_block("High-confidence POSITIVE", samples["high_pos"])
        write_sample_block("High-confidence NEUTRAL", samples["high_neu"])
        write_sample_block("High-confidence NEGATIVE", samples["high_neg"])
        write_sample_block("Low-confidence (ambiguous)", samples["low_conf"])

        rf.write("\n=== TIPS / NEXT STEPS ===\n")
        rf.write(" - If many low-confidence samples are truly ambiguous, consider lowering CONFIDENCE_HIGH.\n")
        rf.write(" - If many low-confidence samples contain Pidgin tokens that are misinterpreted, expand lexicon or route to Stage A lexicon enrichment.\n")
        rf.write(" - You can adjust BATCH_SIZE to trade memory for throughput.\n")
        rf.write("\n")
        rf.write(f"Results CSV saved to: {csv_path}\n")

    # Return paths so caller can use them
    return {"report_path": report_path, "csv_path": csv_path, "summary": {
        "total_time": total_time,
        "msgs_per_second": msgs_per_second,
        "avg_latency_per_message": avg_latency_per_message,
        "distribution": dist,
        "high_conf_count": len(high_conf),
        "low_conf_count": len(low_conf),
    }}


# -----------------------------
# How to call this script
# -----------------------------
# In your environment (where cleaned_messages exists), run:
# from run_naija_afriberta_stageB import run_stageB_scoring
# summary = run_stageB_scoring(cleaned_messages)
#
# After completion, open the generated .txt report in outputs/ and the CSV for full details.
#
# -----------------------------
# If this script is run directly, it attempts to find 'cleaned_messages' in globals().
# This is handy if you `exec` it in REPL where cleaned_messages is present.
# -----------------------------
if __name__ == "__main__":
    parser_android = WhatsAppChatParser(dayfirst=True)
    file_path = "sample_data/WhatsApp Chat - JOHN✊🏽/_chat.txt"

    messages = parser_android.parse_file(file_path)
    print(f"Parsed {len(messages)} messages from {file_path}")
    cleaned_messages: List[CleanedMessage] = clean_messages(messages)
    # try to grab cleaned_messages from globals (if present)
    cm = globals().get("cleaned_messages")
    if cm is None:
        raise RuntimeError("Variable 'cleaned_messages' not found in globals. "
                           "Load your dataset and assign to 'cleaned_messages' before running this script.")
    summary_info = run_stageB_scoring(cm)
    # Do not print anything; paths are written into the report file.
