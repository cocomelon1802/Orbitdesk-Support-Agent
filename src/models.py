"""
Local model wrappers.

Two models are used, both loaded through Hugging Face libraries:

1. Embedding model  -> sentence-transformers/all-MiniLM-L6-v2
   Used for retrieval over the knowledge base + resolved cases.

2. Generation model -> Qwen/Qwen2.5-0.5B-Instruct
   Used to draft the final answer from retrieved evidence.
   Swap MODEL_NAME below for a larger model if your hardware allows
   (e.g. Qwen/Qwen2.5-1.5B-Instruct, microsoft/Phi-3-mini-4k-instruct).

A TF-IDF fallback embedder and a template-based fallback generator are
included behind USE_LOCAL_HF_MODELS=0 (or automatically if the HF
libraries / model weights are not available, e.g. no network). This
keeps the graph runnable and testable in offline/sandboxed environments
and is what backs the automated routing test in tests/test_graph_routing.py.
For the actual submission, models should be loaded for real -- see README.
"""
from __future__ import annotations

import os
import time
import logging
from typing import List
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("orbitdesk.models")

# NOTE: these two constants are pre-pinned to the exact commit hashes
# Hugging Face resolved on a real run (see the "Local models used" table in
# README.md). If you re-download on a different machine and HF resolves a
# newer commit for "main", update these two lines only -- do not remove them.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

GENERATION_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
GENERATION_MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"

USE_LOCAL_HF_MODELS = os.environ.get("USE_LOCAL_HF_MODELS", "1") == "1"


# --------------------------------------------------------------------------- #
# Embedding model
# --------------------------------------------------------------------------- #
class EmbeddingModel:
    """Wraps sentence-transformers with a TF-IDF fallback for offline dev/tests."""

    def __init__(self):
        self.backend = None
        self.load_time_seconds = None
        t0 = time.time()

        if USE_LOCAL_HF_MODELS:
            try:
                
                self._st_model = SentenceTransformer(
                    EMBEDDING_MODEL_NAME, revision=EMBEDDING_MODEL_REVISION, local_files_only=True,
                )
                self.backend = "sentence-transformers"
            except Exception as e:  # noqa: BLE001 - broad on purpose, this is a fallback
                logger.warning(
                    "Falling back to TF-IDF embeddings (%s). "
                    "Install sentence-transformers and ensure model weights are "
                    "downloaded for the real assignment run.",
                    e,
                )

        if self.backend is None:
            from sklearn.feature_extraction.text import TfidfVectorizer

            self._vectorizer = TfidfVectorizer()
            self._fitted = False
            self.backend = "tfidf-fallback"

        self.load_time_seconds = round(time.time() - t0, 3)
        logger.info("Embedding backend=%s load_time=%.3fs", self.backend, self.load_time_seconds)

    def fit_corpus(self, texts: List[str]):
        """Only needed for the TF-IDF fallback; no-op for sentence-transformers."""
        if self.backend == "tfidf-fallback":
            self._corpus_matrix = self._vectorizer.fit_transform(texts)
            self._fitted = True

    def encode(self, texts: List[str]):
        if self.backend == "sentence-transformers":
            return self._st_model.encode(texts, normalize_embeddings=True)
        else:
            if not self._fitted:
                raise RuntimeError("Call fit_corpus() before encode() with the TF-IDF fallback.")
            return self._vectorizer.transform(texts)

    def encode_corpus(self, texts: List[str]):
        """Encode the fixed corpus (used at index-build time)."""
        if self.backend == "sentence-transformers":
            return self._st_model.encode(texts, normalize_embeddings=True)
        else:
            self.fit_corpus(texts)
            return self._corpus_matrix


# --------------------------------------------------------------------------- #
# Generation model
# --------------------------------------------------------------------------- #
class GenerationModel:
    """Wraps a local HF causal LM with a deterministic template fallback."""

    def __init__(self):
        self.backend = None
        self.load_time_seconds = None
        t0 = time.time()

        if USE_LOCAL_HF_MODELS:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                import torch

                self._tokenizer = AutoTokenizer.from_pretrained(
                    GENERATION_MODEL_NAME, revision=GENERATION_MODEL_REVISION, local_files_only=True,
                )
                self._model = AutoModelForCausalLM.from_pretrained(
                    GENERATION_MODEL_NAME,
                    revision=GENERATION_MODEL_REVISION,
                    local_files_only=True,
                )
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
                self._model.to(self._device)
                self.backend = "hf-transformers"
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "Falling back to template-based generation (%s). "
                    "Install transformers/torch and download the model for the "
                    "real assignment run.",
                    e,
                )

        if self.backend is None:
            self.backend = "template-fallback"

        self.load_time_seconds = round(time.time() - t0, 3)
        logger.info("Generation backend=%s load_time=%.3fs", self.backend, self.load_time_seconds)

    def generate(self, prompt: str, system: str = None, max_new_tokens: int = 400) -> str:
        t0 = time.time()
        if self.backend == "hf-transformers":
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            # return_dict=True gives a BatchEncoding (dict-like: input_ids,
            # attention_mask, ...) rather than a bare tensor. Newer
            # transformers versions return a BatchEncoding either way even
            # without return_dict=True, which breaks a plain `.shape` /
            # positional-arg call -- being explicit here and unpacking with
            # **inputs is the version-robust way to call generate().
            inputs = self._tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            ).to(self._device)
            input_len = inputs["input_ids"].shape[1]
            output = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self._tokenizer.eos_token_id,
            )
            text = self._tokenizer.decode(
                output[0][input_len:], skip_special_tokens=True
            )
        else:
            # Deterministic, template-based stand-in used only when no local HF
            # generation model is available (offline sandbox / unit tests).
            text = self._template_fallback(prompt)

        latency = round(time.time() - t0, 3)
        logger.info("Generation backend=%s latency=%.3fs", self.backend, latency)
        return text

    @staticmethod
    def _template_fallback(prompt: str) -> str:
        # Extremely defensive stand-in: never invents content, just says it
        # cannot draft evidence-grounded prose without a real local LLM.
        # Plain text (not JSON) -- matches what the real model is now asked
        # for; _extract_answer_text() in nodes.py handles plain text directly.
        return (
            "A local generation model is not available in this environment, "
            "so no evidence-grounded answer was drafted. Install "
            "transformers/torch and download a local model to enable generation."
        )
