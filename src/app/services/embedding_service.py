# src/app/services/embedding_service.py

import os
import shutil
import logging
import subprocess
from pathlib import Path
from typing import List, Union, Optional
import asyncio

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from src.app.config import settings

log = logging.getLogger(__name__)

EXPECTED_EMBEDDING_DIM = 384
EMBED_BATCH_SIZE = 32


class AfroXLMRMiniEmbedder:
    """ONNX-based INT8 embedder for multilingual `Davlan/afro-xlmr-mini`.

    Produces 384-dimensional L2-normalized embeddings via mean pooling.
    """

    def __init__(self, model_dir: Optional[str] = None):
        self.model_dir = Path(model_dir or settings.EMBEDDING_MODEL_DIR)
        self.base_repo = settings.EMBEDDING_MODEL_REPO
        self._ensure_model_artifacts()

        log.info("Loading AfroXLMR-Mini tokenizer from %s...", self.model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_dir),
            fix_mistral_regex=True,
        )

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        quant_file = self._find_onnx_file()
        log.info("Initializing ONNX Runtime session with %s...", quant_file)
        self.session = ort.InferenceSession(
            str(quant_file),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        log.info("AfroXLMR-Mini INT8 Embedder successfully initialized.")

    def _find_onnx_file(self) -> Path:
        candidates = [
            self.model_dir / "model_quantized.onnx",
            self.model_dir / "model_optimized.onnx",
            self.model_dir / "model.onnx",
        ]
        for c in candidates:
            if c.exists():
                return c
        raise FileNotFoundError(
            f"No ONNX model file found in {self.model_dir}. Looked for: {[c.name for c in candidates]}"
        )

    def _ensure_model_artifacts(self):
        """Ensure ONNX INT8 bundle and tokenizer exist, building or downloading if needed."""
        quant_file = self.model_dir / "model_quantized.onnx"
        base_onnx_file = self.model_dir / "model.onnx"

        if (quant_file.exists() or base_onnx_file.exists()) and (self.model_dir / "tokenizer_config.json").exists():
            return

        log.info("Model artifacts not found at %s. Attempting download/export for %s...", self.model_dir, self.base_repo)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # 1. Try Hugging Face Hub download
        try:
            from huggingface_hub import hf_hub_download
            for filename in ["config.json", "tokenizer_config.json", "special_tokens_map.json", "tokenizer.json", "sentencepiece.bpe.model"]:
                try:
                    hf_hub_download(repo_id=self.base_repo, filename=filename, local_dir=str(self.model_dir))
                except Exception:
                    pass

            for onnx_candidate in ["model_quantized.onnx", "model_optimized.onnx", "model.onnx"]:
                try:
                    hf_hub_download(repo_id=self.base_repo, filename=onnx_candidate, local_dir=str(self.model_dir))
                    if (self.model_dir / onnx_candidate).exists():
                        log.info("Successfully fetched %s from HF Hub.", onnx_candidate)
                        return
                except Exception:
                    pass
        except Exception as hub_err:
            log.debug("HF hub direct download skipped or failed: %s", hub_err)

        # 2. Fallback to optimum-cli if available
        temp_export = self.model_dir.parent / "temp_afro_export"
        try:
            if temp_export.exists():
                shutil.rmtree(temp_export)

            log.info("Step 1/2: Exporting base model to ONNX...")
            subprocess.run(
                f"optimum-cli export onnx --model {self.base_repo} --task feature-extraction {temp_export}",
                shell=True,
                check=True,
            )

            log.info("Step 2/2: Quantizing ONNX model to INT8 (AVX2)...")
            subprocess.run(
                f"optimum-cli onnxruntime quantize --avx2 --onnx_model {temp_export} -o {self.model_dir}",
                shell=True,
                check=True,
            )

            # Copy tokenizer files to model_dir if not present
            tokenizer = AutoTokenizer.from_pretrained(self.base_repo)
            tokenizer.save_pretrained(str(self.model_dir))

        except Exception as e:
            log.warning("Optimum-cli export failed: %s. Attempting direct tokenizer load...", e)
            try:
                tokenizer = AutoTokenizer.from_pretrained(self.base_repo)
                tokenizer.save_pretrained(str(self.model_dir))
            except Exception:
                pass
        finally:
            if temp_export.exists():
                shutil.rmtree(temp_export, ignore_errors=True)

    def _clean_texts(self, texts: List[str]) -> List[str]:
        cleaned = []
        for text in texts:
            if not text:
                cleaned.append("")
                continue
            t = text.replace("passage: ", "").replace("query: ", "")
            cleaned.append(t.strip())
        return cleaned

    def embed_sync(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        cleaned_texts = self._clean_texts(texts)
        all_embeddings: List[List[float]] = []

        for i in range(0, len(cleaned_texts), EMBED_BATCH_SIZE):
            batch = cleaned_texts[i : i + EMBED_BATCH_SIZE]
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
            # If model expects token_type_ids
            if len(self.input_names) > 2 and "token_type_ids" in inputs:
                ort_inputs[self.input_names[2]] = inputs["token_type_ids"]

            outputs = self.session.run(None, ort_inputs)
            last_hidden_state = outputs[0]

            mask = attention_mask.astype(np.float32)
            mask_sum = np.clip(mask.sum(axis=1, keepdims=True), a_min=1e-9, a_max=None)
            pooled = (last_hidden_state * mask[:, :, None]).sum(axis=1) / mask_sum
            normalized = pooled / np.linalg.norm(pooled, axis=1, keepdims=True)
            all_embeddings.extend(normalized.tolist())

        return all_embeddings


_EMBEDDER_INSTANCE: Optional[AfroXLMRMiniEmbedder] = None


def get_embedder() -> AfroXLMRMiniEmbedder:
    """Singleton getter for AfroXLMR-Mini embedder."""
    global _EMBEDDER_INSTANCE
    if _EMBEDDER_INSTANCE is None:
        _EMBEDDER_INSTANCE = AfroXLMRMiniEmbedder()
    return _EMBEDDER_INSTANCE


async def embed_texts(texts: List[str]) -> List[List[float]]:
    """Asynchronous wrapper for embedding a batch of texts."""
    if not texts:
        return []
    embedder = get_embedder()
    return await asyncio.to_thread(embedder.embed_sync, texts)


async def embed_query(query: Union[str, List[str]]) -> List[List[float]]:
    """Asynchronous wrapper for embedding queries."""
    if isinstance(query, str):
        query_list = [query]
    else:
        query_list = query
    return await embed_texts(query_list)