# save as pidgin_vader_pass.py and run in the same env where cleaned_messages is available
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import multiprocessing as mp
import math
import random
from src.app.utils.raw_txt_parser import WhatsAppChatParser
from src.app.utils.pre_process import clean_messages
import csv, os

# If your CleanedMessage is already in the environment, comment out the following re-definition.
@dataclass
class CleanedMessage:
    timestamp: Optional[datetime]
    sender: Optional[str]
    text: str
    raw: str = ""

# ---------------------------
# Minimal Pidgin lexicon seed
# ---------------------------
# Numeric scores chosen to bias obvious Pidgin sentiment tokens.
# Add tokens you observe in your data and adjust scores (-3.0 .. +3.0 typical).
PIDGIN_LEXICON = {
    # positive / praise / affirmation
    "sharp-sharp": 1.8,
    "gbam": 1.5,
    "nice one": 1.8,
    "correct": 1.2,
    "no wahala": 1.5,
    "okey": 0.8,
    "ok": 0.6,
    "well done": 1.6,
    "congratulations": 2.0,
    # negative / complaint / frustration
    "wahala": -2.5,
    "bore": -1.2,
    "na lie": -1.6,
    "chop": -0.8, # context-sensitive: keep light negative
    "scam": -2.0,
    "baka": -1.5,
    "shame": -1.8,
    # intensifiers/emphatics (boost)
    "very": 0.3,
    "too": 0.2,
    # negation / inversion heuristics (VADER handles some negation; we'll keep a few tokens)
    "no be": -0.8,  # often used for negation ("no be small")
    # filler exclamations
    "ehh": -0.2,
    "eh": 0.0,
    "lol": 1.2,
    "hahaha": 1.8,
    "lmao": 1.8,
}

# ---------------------------
# Utilities
# ---------------------------
def init_vader_with_pidgin(pidgin_lexicon: dict) -> SentimentIntensityAnalyzer:
    analyzer = SentimentIntensityAnalyzer()
    # Merge — existing entries are overwritten by our lexicon if key exists
    analyzer.lexicon.update(pidgin_lexicon)
    return analyzer

def score_message(vader: SentimentIntensityAnalyzer, msg: CleanedMessage) -> Dict[str, Any]:
    s = vader.polarity_scores(msg.text)
    return {
        "timestamp": msg.timestamp,
        "sender": msg.sender,
        "text": msg.text,
        "compound": s["compound"],
        "neg": s["neg"],
        "neu": s["neu"],
        "pos": s["pos"],
        "raw": msg.raw,
    }

# Multiprocessing helper
def _score_wrapper(args):
    vader, msg = args
    return score_message(vader, msg)

# ---------------------------
# Analysis pipeline
# ---------------------------
def run_pidgin_vader_pass(
    cleaned_messages: List[CleanedMessage],
    pidgin_lexicon: dict = PIDGIN_LEXICON,
    candidate_thresholds=(0.05, 0.10, 0.15, 0.20),
    sample_ambiguous_n: int = 40,
    use_multiprocessing: bool = True,
    output_dir: str = "outputs",
) -> Dict[str, Any]:
    """Run VADER + Pidgin lexicon sentiment pass and save all results to a report.txt"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"vader_pidgin_report_{timestamp}.txt")
    csv_path = os.path.join(output_dir, f"vader_pidgin_scores_{timestamp}.csv")

    # --- Score all messages ---
    if use_multiprocessing:
        ncpu = max(1, mp.cpu_count() - 1)
        chunk_size = max(1, math.ceil(len(cleaned_messages) / ncpu))
        slices = [cleaned_messages[i:i+chunk_size] for i in range(0, len(cleaned_messages), chunk_size)]
        with mp.Pool(processes=min(len(slices), ncpu)) as pool:
            parts = pool.starmap(_score_chunk, [(sl, pidgin_lexicon) for sl in slices])
        scored = [r for p in parts for r in p]
    else:
        vader = init_vader_with_pidgin(pidgin_lexicon)
        scored = [score_message(vader, m) for m in cleaned_messages]

    total = len(scored)
    compounds = [r["compound"] for r in scored]
    avg_compound = sum(compounds)/total if total else 0.0

    threshold_table = {}
    for t in candidate_thresholds:
        pos = sum(1 for c in compounds if c >= t)
        neg = sum(1 for c in compounds if c <= -t)
        neu = total - pos - neg
        threshold_table[t] = {
            "positive": pos, "negative": neg, "neutral": neu,
            "pos_pct": pos/total, "neg_pct": neg/total, "neu_pct": neu/total
        }

    # Pick recommended threshold heuristically
    recommended = next((t for t in sorted(candidate_thresholds)
                        if 0.30 <= threshold_table[t]["neu_pct"] <= 0.70), 0.10)

    # Sample ambiguous messages
    amb = [r for r in scored if -recommended < r["compound"] < recommended]
    random.shuffle(amb)
    amb_sample = amb[:sample_ambiguous_n]

    # --- Write text report ---
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"📊 WHATSAPP SENTIMENT REPORT (VADER + PIDGIN)\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write(f"Total messages scored: {total}\n")
        f.write(f"Average compound score: {avg_compound:.4f}\n\n")

        f.write("=== Threshold Summary ===\n")
        for t, d in threshold_table.items():
            f.write(
                f"±{t:.2f} → pos {d['positive']} ({d['pos_pct']:.1%}) | "
                f"neg {d['negative']} ({d['neg_pct']:.1%}) | "
                f"neu {d['neutral']} ({d['neu_pct']:.1%})\n"
            )

        f.write(f"\nRecommended threshold: ±{recommended:.2f}\n")
        f.write("\n=== Ambiguous message sample ===\n")
        for i, a in enumerate(amb_sample, 1):
            txt = a['text'].replace("\n", " ")
            f.write(f"{i:2d}. [{a['sender']}] ({a['compound']:+.3f}) {txt[:200]}\n")

    # --- Write full scores CSV ---
    # with open(csv_path, "w", encoding="utf-8", newline="") as cf:
    #     writer = csv.DictWriter(cf, fieldnames=["timestamp","sender","text","compound","pos","neu","neg"])
    #     writer.writeheader()
    #     for r in scored:
    #         writer.writerow(r)

    return {
        "report_path": report_path,
        "csv_path": csv_path,
        "recommended_threshold": recommended,
        "threshold_table": threshold_table,
        "ambiguous_sample": amb_sample,
    }

# ---------------------------
# How to use:
# ---------------------------
# Example (uncomment to run in your environment):
# if __name__ == "__manin__":
#     parser_android = WhatsAppChatParser(dayfirst=True)
#     file_path = "sample_data/WhatsApp Chat - JOHN✊🏽/_chat.txt"

#     messages = parser_android.parse_file(file_path)
#     print(f"Parsed {len(messages)} messages from {file_path}")
#     cleaned: List[CleanedMessage] = clean_messages(messages)

#     summary = run_pidgin_vader_pass(cleaned)
#     print(summary)