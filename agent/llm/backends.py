"""LLM backend interface and implementations (TODO).

This module defines the interface for model backends used by the LLM agent.
Each backend wraps a specific model provider and handles tokenization,
inference, and response parsing.

Planned implementations:

* ``HuggingFaceBackend(model_path)`` — Local model via transformers.
  Switch models by changing the path::

      backend = HuggingFaceBackend("Qwen/Qwen2.5-3B-Instruct")
      backend = HuggingFaceBackend("Qwen/Qwen2.5-7B-Instruct")
      backend = HuggingFaceBackend("/path/to/local/model")

* ``OpenAIBackend(model, api_key)`` — OpenAI-compatible API.
* ``AnthropicBackend(model, api_key)`` — Anthropic Claude API.
* ``VLLMBackend(model_path, ...)`` — High-throughput local inference via vLLM.

Example future usage::

    from agent.llm.backends import HuggingFaceBackend
    from agent.llm.agent import LLMAgent

    backend = HuggingFaceBackend("Qwen/Qwen2.5-3B-Instruct")
    agent = LLMAgent(backend=backend)

    env = balatro_gym.make("easy")
    obs, info = env.reset()
    action = agent.act(obs, info)
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ModelBackend(Protocol):
    """Protocol for LLM backends.

    A backend takes a prompt string and returns the model's text response.
    """

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response from the model.

        Args:
            prompt: The full prompt string (system + user content).
            **kwargs: Backend-specific parameters (temperature, max_tokens, etc.)

        Returns:
            The model's text response.
        """
        ...


# --------------------------------------------------------------------------
# TODO: Implement the following backends when ready to run experiments.
# --------------------------------------------------------------------------

# class HuggingFaceBackend:
#     """Local model inference via HuggingFace transformers.
#
#     Args:
#         model_path: HuggingFace model ID or local path.
#         device: "cuda", "cpu", or "auto".
#         max_new_tokens: Maximum tokens to generate.
#         temperature: Sampling temperature.
#     """
#
#     def __init__(
#         self,
#         model_path: str,
#         device: str = "auto",
#         max_new_tokens: int = 256,
#         temperature: float = 0.1,
#     ):
#         from transformers import AutoModelForCausalLM, AutoTokenizer
#         self.tokenizer = AutoTokenizer.from_pretrained(model_path)
#         self.model = AutoModelForCausalLM.from_pretrained(
#             model_path, device_map=device, torch_dtype="auto"
#         )
#         self.max_new_tokens = max_new_tokens
#         self.temperature = temperature
#
#     def generate(self, prompt: str, **kwargs) -> str:
#         inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
#         outputs = self.model.generate(
#             **inputs,
#             max_new_tokens=kwargs.get("max_new_tokens", self.max_new_tokens),
#             temperature=kwargs.get("temperature", self.temperature),
#             do_sample=True,
#         )
#         return self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)


# class OpenAIBackend:
#     """OpenAI-compatible API backend.
#
#     Works with OpenAI, Azure OpenAI, or any OpenAI-compatible endpoint.
#     """
#
#     def __init__(self, model: str = "gpt-4o", api_key: str | None = None, base_url: str | None = None):
#         from openai import OpenAI
#         self.client = OpenAI(api_key=api_key, base_url=base_url)
#         self.model = model
#
#     def generate(self, prompt: str, **kwargs) -> str:
#         response = self.client.chat.completions.create(
#             model=self.model,
#             messages=[{"role": "user", "content": prompt}],
#             temperature=kwargs.get("temperature", 0.1),
#             max_tokens=kwargs.get("max_tokens", 256),
#         )
#         return response.choices[0].message.content


# class AnthropicBackend:
#     """Anthropic Claude API backend."""
#
#     def __init__(self, model: str = "claude-sonnet-4-20250514", api_key: str | None = None):
#         import anthropic
#         self.client = anthropic.Anthropic(api_key=api_key)
#         self.model = model
#
#     def generate(self, prompt: str, **kwargs) -> str:
#         response = self.client.messages.create(
#             model=self.model,
#             messages=[{"role": "user", "content": prompt}],
#             max_tokens=kwargs.get("max_tokens", 256),
#             temperature=kwargs.get("temperature", 0.1),
#         )
#         return response.content[0].text
