from __future__ import annotations

from typing import Optional, Tuple, List, Union, Any
import base64
import io
import math
import mimetypes

import numpy as np
from PIL import Image
from openai import OpenAI


def _to_uint8_rgb(arr: np.ndarray) -> np.ndarray:
    """Ensure np array is uint8 RGB."""
    a = np.asarray(arr)
    if a.dtype != np.uint8:
        a = np.clip(a, 0.0, 255.0)
        if a.max() <= 1.0:
            a = a * 255.0
        a = a.astype(np.uint8)

    if a.ndim == 2:
        a = np.stack([a, a, a], axis=-1)

    if a.ndim == 3 and a.shape[-1] == 4:
        a = a[..., :3]
    elif a.ndim == 3 and a.shape[-1] == 1:
        a = np.repeat(a, 3, axis=-1)

    if a.ndim != 3 or a.shape[-1] != 3:
        raise ValueError(f"Unsupported image shape: {a.shape}")
    return a


def _image_to_data_uri(img: Union[np.ndarray, Image.Image, str], jpeg_quality: int = 90) -> str:
    """Convert (np.ndarray | PIL.Image | path) -> data URI. Uses PNG for .png path, JPEG otherwise."""
    if isinstance(img, str):
        mime, _ = mimetypes.guess_type(img)
        mime = mime or "application/octet-stream"
        with open(img, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime};base64,{b64}"

    if isinstance(img, np.ndarray):
        rgb = _to_uint8_rgb(img)
        pil = Image.fromarray(rgb, mode="RGB")
    elif isinstance(img, Image.Image):
        pil = img.convert("RGB")
    else:
        raise ValueError(f"Unsupported image type: {type(img)}")

    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def _extract_first_token_topk_probs(choice_dict: dict, top_k: int) -> Optional[List[Tuple[str, float]]]:
    """
    Parse OpenAI-style chat logprobs:
    choices[0].logprobs.content[0].top_logprobs = [{token, logprob, ...}, ...]
    Return top_k as [(token, prob), ...] for the FIRST generated token.
    """
    lp = choice_dict.get("logprobs") or {}
    content_lp = lp.get("content") or []
    if not content_lp:
        return None

    first = content_lp[0]
    top = first.get("top_logprobs") or []
    out: List[Tuple[str, float]] = []
    for item in top[:top_k]:
        tok = item.get("token", "")
        logp = item.get("logprob", None)
        if logp is None:
            continue
        out.append((tok, float(math.exp(logp))))
    return out


class LLavaNextClient:

    def __init__(self, port: int = 12189):
        self.url =f"http://localhost:{port}/v1"
        self.client = OpenAI(base_url=self.url, api_key="")

        models = self.client.models.list()
        if not models.data:
            self.model_name = "qwen"
        else:
            self.model_name = models.data[0].id
        self.max_new_tokens = 2500

    def ask(self, image: np.ndarray, prompt: Optional[str] = None, return_token_likelihood=False) -> tuple:
        if prompt is None:
            raise ValueError("prompt must be provided.")

        images = image if isinstance(image, list) else [image]
        content = []
        for img in images:
            content.append({"type": "image_url", "image_url": {"url": _image_to_data_uri(img)}})
        content.append({"type": "text", "text": prompt})

        kwargs = dict(
            model=self.model_name,
            messages=[{"role": "user", "content": content}],
            temperature=0.0,
            top_p=1.0,
            max_tokens=self.max_new_tokens,
            frequency_penalty=0.6, 
            presence_penalty=0.3,
        )
        if return_token_likelihood:
            kwargs.update(dict(logprobs=True, top_logprobs=3))
        
        response = self.client.chat.completions.create(**kwargs)
        response = response.model_dump()
        out_text = response["choices"][0]["message"]["content"]

        if return_token_likelihood:
            likelihood = _extract_first_token_topk_probs(response["choices"][0], top_k=3)
            return out_text, likelihood
        else:
            return out_text, None

    def ask2(self, image: np.ndarray, prompt: Optional[str] = None, return_token_likelihood=False) -> tuple:
        """Yes/No focused API. Uses server-side ask2 via mode='v2'.

        Returns
        -------
        (str, list|None)
            lmm_output string and likelihood list when requested.
        """
        if prompt is None:
            raise ValueError("Prompt must be provided for ask2 method.")

        images = image if isinstance(image, list) else [image]
        content = []
        for img in images:
            content.append({"type": "image_url", "image_url": {"url": _image_to_data_uri(img)}})
        content.append({"type": "text", "text": prompt})

        kwargs = dict(
            model=self.model_name,
            messages=[{"role": "user", "content": content}],
            temperature=0.0,
            top_p=1.0,
            max_tokens=self.max_new_tokens,
            frequency_penalty=0.6, 
            presence_penalty=0.3,
        )

        if return_token_likelihood:
            kwargs.update(dict(logprobs=True, top_logprobs=3))

        response = self.client.chat.completions.create(**kwargs)
        response = response.model_dump()
        out_text = response["choices"][0]["message"]["content"]
        usage = response.get("usage") or {}
        completion_tokens = usage.get("completion_tokens")

        if return_token_likelihood:
            likelihood = _extract_first_token_topk_probs(response["choices"][0], top_k=3)
            return out_text, likelihood, completion_tokens
        else:
            return out_text, None, completion_tokens
