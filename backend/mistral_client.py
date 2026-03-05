"""
MARKET LENS AI — ULTRA STABLE CHAT CLIENT (2026)

Features:
✅ HuggingFace Router compatible
✅ Automatic model fallback
✅ Smart retry system
✅ Timeout protection
✅ Free-tier optimized
✅ Zero downtime responses
✅ OpenAI-style wrapper compatible
"""

import os
import time
from typing import Dict, List

from huggingface_hub import InferenceClient


# ==============================
# CORE CLIENT
# ==============================

class MarketLensAIClient:

    # ⭐ 2026 FREE-TIER STABLE MODELS (ordered by reliability)
    MODELS = [
        "mistralai/Mistral-7B-Instruct-v0.3",
        "microsoft/Phi-3-mini-4k-instruct",
        "HuggingFaceH4/zephyr-7b-beta",
        "Qwen/Qwen2.5-7B-Instruct",
    ]

    MAX_RETRIES = 2
    MODEL_COOLDOWN = 60  # seconds before retrying failed model

    def __init__(self, api_token: str = None):

        self.api_token = api_token or os.getenv("HUGGINGFACE_API_TOKEN")

        if not self.api_token:
            raise ValueError(
                "Missing HUGGINGFACE_API_TOKEN in .env"
            )

        self.client = InferenceClient(
            token=self.api_token,
            timeout=60,
        )

        self.failed_models = {}
        self.working_model = None

    # ==============================
    # MODEL HEALTH CHECK
    # ==============================

    def _model_available(self, model: str) -> bool:
        """Skip models recently failed."""
        if model not in self.failed_models:
            return True

        last_fail = self.failed_models[model]
        return (time.time() - last_fail) > self.MODEL_COOLDOWN

    def _mark_failed(self, model: str):
        self.failed_models[model] = time.time()

    # ==============================
    # CHAT COMPLETION
    # ==============================

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.4,
        max_tokens: int = 900,
    ) -> str:

        models_to_try = []

        if self.working_model:
            models_to_try.append(self.working_model)

        models_to_try.extend(
            [m for m in self.MODELS if m not in models_to_try]
        )

        last_error = None

        for model in models_to_try:

            if not self._model_available(model):
                continue

            for attempt in range(self.MAX_RETRIES):

                try:
                    response = self.client.chat_completion(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )

                    text = response.choices[0].message.content

                    if text and text.strip():
                        self.working_model = model
                        print(f"✅ MarketLens AI using: {model}")
                        return text.strip()

                except Exception as e:

                    error = str(e).lower()
                    last_error = str(e)

                    # Model loading
                    if "loading" in error or "503" in error:
                        return "⏳ AI is waking up. Please retry in 20 seconds."

                    # Router removed model
                    if "410" in error or "gone" in error:
                        self._mark_failed(model)
                        break

                    # Rate limit
                    if "429" in error:
                        time.sleep(2)
                        continue

                    # Unauthorized
                    if "401" in error or "unauthorized" in error:
                        return self._token_error()

                    self._mark_failed(model)

        return self._fallback_response(last_error)

    # ==============================
    # ERROR HANDLING
    # ==============================

    def _token_error(self):
        return (
            "❌ Invalid Hugging Face API token.\n\n"
            "1. Go to https://huggingface.co/settings/tokens\n"
            "2. Create READ token\n"
            "3. Update .env\n"
            "4. Restart app"
        )

    def _fallback_response(self, error):
        return (
            "⚠️ AI temporarily busy.\n\n"
            "Free AI models are under heavy demand.\n"
            "Please retry in a few seconds.\n\n"
            f"Debug: {str(error)[:150]}"
        )


# ==============================
# OPENAI COMPATIBILITY LAYER
# ==============================

class ChatCompletion:

    def __init__(self, client: MarketLensAIClient):
        self.client = client

    def create(self, model, messages, temperature=0.4, **kwargs):

        text = self.client.chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=kwargs.get("max_tokens", 900),
        )

        return Response(text)


class Response:
    def __init__(self, text):
        self.choices = [Choice(text)]


class Choice:
    def __init__(self, text):
        self.message = Message(text)


class Message:
    def __init__(self, text):
        self.content = text


class MarketLensAIWrapper:
    """
    Drop-in replacement for OpenAI client.
    """

    def __init__(self, api_token=None):
        client = MarketLensAIClient(api_token)
        self.chat = type(
            "Chat",
            (),
            {"completions": ChatCompletion(client)},
        )()
        # compatibility alias
MistralClientWrapper = MarketLensAIWrapper