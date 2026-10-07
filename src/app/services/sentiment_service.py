# src/app/services/sentiment_service.py

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import asyncio

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
from src.app.config import settings

log = logging.getLogger(__name__)

SENTIMENT_BATCH_SIZE = 32


class AfroXLMRMiniSentimentClassifier:
    """INT8 ONNX Sequence Classification for Nigerian Sentiment Analysis.

    Model: JohnAlbarkaIbrahim/afroxlmr-mini-nigerian-sentiment (v1/onnx_int8/)
    Inference: ONNX Runtime CPUExecutionProvider (Zero PyTorch runtime dependency).
    """

    def __init__(
        self,
        repo_id: Optional[str] = None,
        subfolder: Optional[str] = None,
        local_dir: Optional[str] = None,
    ):
        self.repo_id = repo_id or settings.SENTIMENT_MODEL_REPO
        self.subfolder = subfolder or settings.SENTIMENT_MODEL_SUBFOLDER
        self.local_dir = Path(local_dir or settings.SENTIMENT_MODEL_DIR)

        log.info("Loading tokenizer for sentiment model %s...", self.repo_id)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("Davlan/afro-xlmr-mini")
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(self.repo_id, use_fast=True)

        self._ensure_model_files()
        # Dynamically load id2label mapping from config.json
        self._load_label_mapping()

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        onnx_model_path = self._find_onnx_model_path()
        log.info("Initializing Sentiment ONNX Runtime session from %s...", onnx_model_path)
        self.session = ort.InferenceSession(
            str(onnx_model_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        log.info("AfroXLMR-Mini Nigerian Sentiment classifier ready. Labels: %s", self.id2label)

    def _ensure_model_files(self):
        """Download tokenizer and ONNX INT8 artifacts from HuggingFace Hub if missing."""
        self.local_dir.mkdir(parents=True, exist_ok=True)
        config_path = self.local_dir / "config.json"
        
        # Check if ONNX model exists in local_dir
        has_onnx = any(
            (self.local_dir / name).exists()
            for name in ["model_quantized.onnx", "model_optimized.onnx", "model.onnx"]
        )

        if has_onnx and config_path.exists():
            return

        log.info(
            "Fetching ONNX sentiment artifacts from %s (subfolder: %s)...",
            self.repo_id,
            self.subfolder,
        )

        # Download config and tokenizer files
        for filename in [
            "config.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "sentencepiece.bpe.model",
        ]:
            try:
                hf_hub_download(
                    repo_id=self.repo_id,
                    filename=filename,
                    local_dir=str(self.local_dir),
                )
            except Exception as e:
                log.debug("Optional file %s not found on hub: %s", filename, e)

        # Download ONNX model file from subfolder or root
        onnx_downloaded = False
        for onnx_name in ["model_quantized.onnx", "model_optimized.onnx", "model.onnx"]:
            try:
                # Try with subfolder
                downloaded = hf_hub_download(
                    repo_id=self.repo_id,
                    filename=f"{self.subfolder}/{onnx_name}" if self.subfolder else onnx_name,
                    local_dir=str(self.local_dir),
                )
                if downloaded:
                    onnx_downloaded = True
                    break
            except Exception:
                try:
                    # Try without subfolder
                    downloaded = hf_hub_download(
                        repo_id=self.repo_id,
                        filename=onnx_name,
                        local_dir=str(self.local_dir),
                    )
                    if downloaded:
                        onnx_downloaded = True
                        break
                except Exception:
                    continue

        if not onnx_downloaded and not has_onnx:
            # Check if models/onnx_model_optimized exists as a local fallback
            legacy_dir = Path("./models/onnx_model_optimized")
            if legacy_dir.exists() and (legacy_dir / "model.onnx").exists():
                log.info("Using local fallback model from %s", legacy_dir)
                for f in legacy_dir.glob("*"):
                    shutil_dest = self.local_dir / f.name
                    if not shutil_dest.exists() and f.is_file():
                        import shutil
                        shutil.copy2(str(f), str(shutil_dest))

    def _find_onnx_model_path(self) -> Path:
        for root, _, files in os.walk(self.local_dir):
            for file in files:
                if file.endswith(".onnx"):
                    return Path(root) / file
        raise FileNotFoundError(f"No .onnx model file found in {self.local_dir}")

    def _load_label_mapping(self):
        config_file = self.local_dir / "config.json"
        self.id2label = {0: "negative", 1: "neutral", 2: "positive"}
        self.label2id = {"negative": 0, "neutral": 1, "positive": 2}

        if config_file.exists():
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                if "id2label" in cfg and cfg["id2label"]:
                    # Normalize string keys to ints and lowercase labels
                    self.id2label = {
                        int(k): str(v).lower() for k, v in cfg["id2label"].items()
                    }
                if "label2id" in cfg and cfg["label2id"]:
                    self.label2id = {
                        str(k).lower(): int(v) for k, v in cfg["label2id"].items()
                    }
            except Exception as e:
                log.warning("Could not read id2label from config.json: %s. Using standard defaults.", e)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    def predict_sync(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Run batch inference on a list of texts and return classification results."""
        if not texts:
            return []

        results = []
        for i in range(0, len(texts), SENTIMENT_BATCH_SIZE):
            batch = texts[i : i + SENTIMENT_BATCH_SIZE]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="np",
            )
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            ort_inputs = {
                self.input_names[0]: input_ids,
                self.input_names[1]: attention_mask,
            }
            if len(self.input_names) > 2 and "token_type_ids" in inputs:
                ort_inputs[self.input_names[2]] = inputs["token_type_ids"]

            outputs = self.session.run(None, ort_inputs)
            logits = outputs[0]  # shape: (batch_size, num_classes)
            probs = self._softmax(logits)

            for item_probs in probs:
                best_class_idx = int(np.argmax(item_probs))
                best_label = self.id2label.get(best_class_idx, "neutral").lower()
                best_score = float(item_probs[best_class_idx])

                # Build dictionary with score breakdown
                score_dict: Dict[str, float] = {}
                for idx, p in enumerate(item_probs):
                    label_name = self.id2label.get(idx, f"label_{idx}").lower()
                    score_dict[label_name] = float(p)

                results.append({
                    "overall_label": best_label,
                    "overall_label_score": best_score,
                    "score_positive": score_dict.get("positive"),
                    "score_negative": score_dict.get("negative"),
                    "score_neutral": score_dict.get("neutral"),
                })

        return results


_SENTIMENT_CLASSIFIER_INSTANCE: Optional[AfroXLMRMiniSentimentClassifier] = None


def get_sentiment_classifier() -> AfroXLMRMiniSentimentClassifier:
    """Singleton getter for Sentiment Classifier."""
    global _SENTIMENT_CLASSIFIER_INSTANCE
    if _SENTIMENT_CLASSIFIER_INSTANCE is None:
        _SENTIMENT_CLASSIFIER_INSTANCE = AfroXLMRMiniSentimentClassifier()
    return _SENTIMENT_CLASSIFIER_INSTANCE


async def predict_sentiment(texts: List[str]) -> List[Dict[str, Any]]:
    """Asynchronous wrapper for sentiment classification."""
    if not texts:
        return []
    classifier = get_sentiment_classifier()
    return await asyncio.to_thread(classifier.predict_sync, texts)
