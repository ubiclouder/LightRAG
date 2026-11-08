from collections.abc import AsyncIterator
import json
import os
import re

import pipmaster as pm

# install specific modules
if not pm.is_installed("ollama"):
    pm.install("ollama")

# Install httpx for direct HTTP requests with NaN handling
if not pm.is_installed("httpx"):
    pm.install("httpx")

import ollama
import httpx

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from lightrag.exceptions import (
    APIConnectionError,
    RateLimitError,
    APITimeoutError,
)
from lightrag.api import __api_version__

import numpy as np
from typing import Optional, Union
from lightrag.utils import (
    wrap_embedding_func_with_attrs,
    logger,
)


_OLLAMA_CLOUD_HOST = "https://ollama.com"
_CLOUD_MODEL_SUFFIX_PATTERN = re.compile(r"(?:-cloud|:cloud)$")

# Timeout configuration for Ollama LLM calls (in seconds)
_OLLAMA_DEFAULT_TIMEOUT = 300  # 5 minutes for entity extraction
_OLLAMA_EMBED_TIMEOUT = 120    # 2 minutes for embeddings


def _coerce_host_for_cloud_model(host: Optional[str], model: object) -> Optional[str]:
    if host:
        return host
    try:
        model_name_str = str(model) if model is not None else ""
    except (TypeError, ValueError, AttributeError) as e:
        logger.warning(f"Failed to convert model to string: {e}, using empty string")
        model_name_str = ""
    if _CLOUD_MODEL_SUFFIX_PATTERN.search(model_name_str):
        logger.debug(
            f"Detected cloud model '{model_name_str}', using Ollama Cloud host"
        )
        return _OLLAMA_CLOUD_HOST
    return host


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APITimeoutError, httpx.ReadTimeout)
    ),
)
async def _ollama_model_if_cache(
    model,
    prompt,
    system_prompt=None,
    history_messages=[],
    enable_cot: bool = False,
    **kwargs,
) -> Union[str, AsyncIterator[str]]:
    if enable_cot:
        logger.debug("enable_cot=True is not supported for ollama and will be ignored.")
    stream = True if kwargs.get("stream") else False

    kwargs.pop("max_tokens", None)
    # kwargs.pop("response_format", None) # allow json
    host = kwargs.pop("host", None)
    timeout = kwargs.pop("timeout", None)
    # Use longer timeout for entity extraction (5 minutes default)
    # Convert from milliseconds to seconds for Ollama client
    if timeout == 0:
        timeout = None
    elif timeout is None:
        timeout = _OLLAMA_DEFAULT_TIMEOUT * 1000  # Convert to milliseconds
    else:
        # If timeout is provided, ensure it's at least our minimum
        timeout = max(timeout, _OLLAMA_DEFAULT_TIMEOUT * 1000)

    kwargs.pop("hashing_kv", None)
    api_key = kwargs.pop("api_key", None)
    # fallback to environment variable when not provided explicitly
    if not api_key:
        api_key = os.getenv("OLLAMA_API_KEY")
    headers = {
        "Content-Type": "application/json",
        "User-Agent": f"LightRAG/{__api_version__}",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    host = _coerce_host_for_cloud_model(host, model)

    ollama_client = ollama.AsyncClient(host=host, timeout=timeout, headers=headers)

    try:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})

        response = await ollama_client.chat(model=model, messages=messages, **kwargs)
        if stream:
            """cannot cache stream response and process reasoning"""

            async def inner():
                try:
                    async for chunk in response:
                        yield chunk["message"]["content"]
                except Exception as e:
                    logger.error(f"Error in stream response: {str(e)}")
                    raise
                finally:
                    try:
                        await ollama_client._client.aclose()
                        logger.debug("Successfully closed Ollama client for streaming")
                    except Exception as close_error:
                        logger.warning(f"Failed to close Ollama client: {close_error}")

            return inner()
        else:
            model_response = response["message"]["content"]

            """
            If the model also wraps its thoughts in a specific tag,
            this information is not needed for the final
            response and can simply be trimmed.
            """

            return model_response
    except httpx.ReadTimeout as e:
        logger.warning(f"Ollama LLM read timeout for model {model} with timeout {timeout}ms. Retrying...")
        raise APITimeoutError(f"Ollama read timeout: {str(e)}") from e
    except Exception as e:
        try:
            await ollama_client._client.aclose()
            logger.debug("Successfully closed Ollama client after exception")
        except Exception as close_error:
            logger.warning(
                f"Failed to close Ollama client after exception: {close_error}"
            )
        raise e
    finally:
        if not stream:
            try:
                await ollama_client._client.aclose()
                logger.debug(
                    "Successfully closed Ollama client for non-streaming response"
                )
            except Exception as close_error:
                logger.warning(
                    f"Failed to close Ollama client in finally block: {close_error}"
                )


async def ollama_model_complete(
    prompt,
    system_prompt=None,
    history_messages=[],
    enable_cot: bool = False,
    keyword_extraction=False,
    **kwargs,
) -> Union[str, AsyncIterator[str]]:
    keyword_extraction = kwargs.pop("keyword_extraction", None)
    if keyword_extraction:
        kwargs["format"] = "json"
    model_name = kwargs["hashing_kv"].global_config["llm_model_name"]
    return await _ollama_model_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        enable_cot=enable_cot,
        **kwargs,
    )


@wrap_embedding_func_with_attrs(embedding_dim=768, max_token_size=8192)
async def ollama_embed(texts: list[str], embed_model, **kwargs) -> np.ndarray:
    api_key = kwargs.pop("api_key", None)
    if not api_key:
        api_key = os.getenv("OLLAMA_API_KEY")
    headers = {
        "Content-Type": "application/json",
        "User-Agent": f"LightRAG/{__api_version__}",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    host = kwargs.pop("host", None)
    timeout = kwargs.pop("timeout", None)
    # Use longer timeout for embeddings (2 minutes default)
    timeout_seconds = timeout / 1000 if timeout else _OLLAMA_EMBED_TIMEOUT

    host = _coerce_host_for_cloud_model(host, embed_model)

    # Build the API URL
    if host is None:
        host = "http://localhost:11434"
    api_url = f"{host}/api/embed" if not host.endswith("/") else f"{host}api/embed"
    if host == _OLLAMA_CLOUD_HOST:
        api_url = f"{host}/api/embed"

    # Build request payload
    options = kwargs.pop("options", {})
    payload = {
        "model": embed_model,
        "input": texts,
    }
    # Only add options if they have values (avoid sending empty/None values)
    if options:
        filtered_options = {k: v for k, v in options.items() if v is not None and v != ""}
        if filtered_options:
            payload["options"] = filtered_options

    logger.debug(f"Ollama embed payload: model={embed_model}, input texts count={len(texts)}")

    # Retry logic for transient Ollama server errors
    max_retries = 3
    retry_delay = 1.0  # seconds
    last_error = None

    for attempt in range(max_retries):
        try:
            # Make direct HTTP request with NaN-safe JSON decoding
            async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                response = await client.post(
                    api_url,
                    json=payload,
                    headers=headers,
                )

                # Handle 500 errors from Ollama (NaN in response)
                if response.status_code >= 400:
                    error_text = response.text
                    if "NaN" in error_text or "unsupported value" in error_text:
                        logger.warning(
                            f"Ollama returned NaN in embedding response (attempt {attempt + 1}/{max_retries})"
                        )
                        if attempt < max_retries - 1:
                            import asyncio
                            await asyncio.sleep(retry_delay * (2 ** attempt))
                            continue
                        else:
                            # All retries failed, return zero embeddings as fallback
                            logger.error(
                                f"Ollama consistently returned NaN embeddings after {max_retries} attempts. "
                                f"Returning zero embeddings for {len(texts)} texts."
                            )
                            return np.zeros((len(texts), 768), dtype=np.float32)

                    response.raise_for_status()

                # Parse JSON with NaN handling - decode manually to handle NaN values
                response_text = response.text
                # Replace NaN, Infinity, -Infinity with null for JSON parsing
                sanitized_text = response_text.replace('NaN', 'null')
                sanitized_text = sanitized_text.replace('Infinity', 'null')
                sanitized_text = sanitized_text.replace('-Infinity', 'null')

                data = json.loads(sanitized_text)

            embeddings = np.array(data["embeddings"])
            # Replace any remaining NaN values with 0.0
            nan_count = np.sum(np.isnan(embeddings))
            if nan_count > 0:
                logger.warning(
                    f"Detected {nan_count} NaN values in Ollama embeddings, replacing with 0.0"
                )
                embeddings = np.where(np.isnan(embeddings), 0.0, embeddings)
            return embeddings

        except httpx.HTTPStatusError as e:
            last_error = e
            if e.response.status_code >= 500:
                # Server error - retry with backoff
                if attempt < max_retries - 1:
                    import asyncio
                    await asyncio.sleep(retry_delay * (2 ** attempt))
                    logger.warning(
                        f"Ollama server error (attempt {attempt + 1}/{max_retries}): {str(e)}, retrying..."
                    )
                    continue
            logger.error(f"HTTP error in ollama_embed: {str(e)}")
            raise

        except httpx.HTTPError as e:
            logger.error(f"HTTP error in ollama_embed: {str(e)}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in ollama_embed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error in ollama_embed: {str(e)}")
            raise

    # If we get here, all retries failed
    if last_error:
        raise last_error
    raise Exception("Unexpected error in ollama_embed: no data returned")
