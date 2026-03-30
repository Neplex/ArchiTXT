from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.rate_limiters import BaseRateLimiter  # vu


def _get_local_chat_model(
    model_provider: str,
    model_name: str,
    *,
    max_tokens: int,
    temperature: float,
    rate_limiter: BaseRateLimiter | None = None,
    openvino: bool = False,
) -> BaseChatModel:
    if model_provider != 'huggingface':
        msg = f'Unsupported model provider for local mode: {model_provider}. Should be huggingface'
        raise ValueError(msg)

    from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

    pipeline = HuggingFacePipeline.from_model_id(
        model_id=model_name,
        task='text-generation',
        device_map=None if openvino else 'auto',
        backend='openvino' if openvino else 'pt',
        model_kwargs={'export': True} if openvino else {'torch_dtype': 'auto'},
        pipeline_kwargs={
            'use_cache': True,
            'do_sample': True,
            'return_full_text': False,
            'repetition_penalty': 1.1,
            'num_return_sequences': 1,
            'pad_token_id': 0,
        },
    )
    return ChatHuggingFace(
        llm=pipeline,
        rate_limiter=rate_limiter,
        max_tokens=max_tokens,
        temperature=temperature,
    )


def get_chat_model(
    model_provider: str,
    model_name: str,
    *,
    max_tokens: int,
    temperature: float,
    rate_limiter: BaseRateLimiter | None = None,
    local: bool = False,
    openvino: bool = False,
) -> BaseChatModel:
    if local:
        return _get_local_chat_model(
            model_provider,
            model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            rate_limiter=rate_limiter,
            openvino=openvino,
        )

    from langchain.chat_models import init_chat_model

    return init_chat_model(
        model_provider=model_provider,
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        rate_limiter=rate_limiter,
    )
