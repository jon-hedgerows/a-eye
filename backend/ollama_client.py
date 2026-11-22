from __future__ import annotations

import asyncio
import base64
import io
import logging
import re
from pathlib import Path
from typing import Any

import httpx
from PIL import Image

from backend.image_io import open_image

logger = logging.getLogger(__name__)

# Generous timeout for vision models — they can be slow on large images
_GENERATE_TIMEOUT = 300.0  # 5 minutes
_DEFAULT_TIMEOUT = 30.0

# Max dimension (longest side) for images sent to the vision model.
# Vision models resize internally to ~448-672px anyway, so sending anything
# larger than this wastes memory, bandwidth, and processing time.
_VISION_MAX_PX = 1280
_VISION_JPEG_QUALITY = 85

# Name substrings used to guess vision capability when /api/show fails.
_VISION_NAME_HINTS = (
    "llava",
    "minicpm-v",
    "moondream",
    "bakllava",
    "-vl",
    "vl:",
    "vision",
)


class _V1Unsupported(Exception):
    """Internal signal that the /v1 endpoint is not supported by the server."""


def _error_detail(resp: httpx.Response) -> str:
    """Extract a human-readable error detail from an error response."""
    try:
        return resp.json().get("error", "") or resp.text[:300]
    except Exception:
        return resp.text[:300]


class OllamaClient:
    """Async wrapper for the Ollama HTTP API.

    Talks to OpenAI-compatible /v1 endpoints (/v1/models,
    /v1/chat/completions) when the target is an OpenAI-compatible
    server, and to Ollama-native endpoints (/api/*) when there is no
    OpenAI-compatible alternative, failing cleanly to reasonable defaults.
    The base URL is inspected at construction time: if it already ends with
    "/v1" it is used as-is for /v1 endpoints and stripped of "/v1" for
    Ollama-native endpoints; otherwise "/v1" is appended for /v1 endpoints
    and the host is used as-is for Ollama-native endpoints.
    """

    def __init__(
        self,
        host: str,
        vision_model: str,
        llm_model: str = "",
        openai_api_key: str = "",
    ) -> None:
        host = host.rstrip("/")
        self.host = host
        self.vision_model = vision_model
        self.llm_model = llm_model
        self.openai_api_key = openai_api_key
        self._client = httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT)
        # Prompt templates — set via set_templates() after startup
        self._vision_template: str = ""
        self._context_template: str = ""
        # Resolve the base URLs used for /v1 (OpenAI-compatible) and
        # Ollama-native (/api) endpoints.
        if host.endswith("/v1"):
            self._v1_base = host
            self._ollama_base = host[: -len("/v1")]
        else:
            self._v1_base = host + "/v1"
            self._ollama_base = host

    def set_templates(self, vision_template: str, context_template: str) -> None:
        """Set the active prompt templates (called on startup and when prompts change)."""
        self._vision_template = vision_template
        self._context_template = context_template

    async def close(self) -> None:
        await self._client.aclose()

    # -- Helpers -------------------------------------------------------------

    def _auth_headers(self) -> dict[str, str]:
        """Return authorization headers if an API key is configured."""
        if self.openai_api_key:
            return {"Authorization": f"Bearer {self.openai_api_key}"}
        return {}

    # -- Connection ----------------------------------------------------------

    async def check_connection(self) -> bool:
        """Return True if the server is reachable.

        Tries the OpenAI-compatible /v1/models endpoint first, then falls
        back to the Ollama-native /api/tags endpoint.
        """
        # Try OpenAI-compatible endpoint first.
        try:
            resp = await self._client.get(
                f"{self._v1_base}/models",
                headers=self._auth_headers(),
            )
            if resp.status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        # Fall back to Ollama-native endpoint.
        try:
            resp = await self._client.get(f"{self._ollama_base}/api/tags")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def list_models(self) -> list[dict[str, Any]]:
        """Return the list of models available on the server.

        Prefers the OpenAI-compatible /v1/models endpoint (parsing
        ``data[].id``) and falls back to the Ollama-native /api/tags
        endpoint (parsing ``models[]``).
        """
        # Try OpenAI-compatible endpoint first.
        try:
            resp = await self._client.get(
                f"{self._v1_base}/models",
                headers=self._auth_headers(),
            )
            resp.raise_for_status()
            data = resp.json()
            models = [
                {"id": m.get("id", ""), "name": m.get("id", "")}
                for m in data.get("data", [])
                if m.get("id")
            ]
            if models:
                return models
        except httpx.HTTPError:
            logger.debug(
                "OpenAI /v1/models unavailable, trying Ollama /api/tags", exc_info=True
            )
        # Fall back to Ollama-native endpoint.
        try:
            resp = await self._client.get(f"{self._ollama_base}/api/tags")
            resp.raise_for_status()
            data = resp.json()
            return data.get("models", [])
        except httpx.HTTPError:
            logger.warning("Failed to list models", exc_info=True)
            return []

    async def get_model_capabilities(self, model_name: str) -> list[str] | None:
        """Fetch capabilities for a single model via /api/show. None = query failed.

        /api/show is Ollama-native; OpenAI-compatible servers do not expose it,
        so a failure here is expected and handled gracefully by callers.
        """
        try:
            resp = await self._client.post(
                f"{self._ollama_base}/api/show",
                json={"name": model_name},
            )
            resp.raise_for_status()
            caps = resp.json().get("capabilities", None)
            if not caps:
                logger.warning(
                    "Capability query return nothing for model %s",
                    model_name,
                    exc_info=True,
                )
            return caps
        except httpx.HTTPError:
            logger.warning(
                "Capability query failed for model %s", model_name, exc_info=True
            )
            return None

    async def list_models_by_capability(self) -> dict[str, list[str]]:
        """Return models grouped into vision-capable and text-only lists."""
        models = await self.list_models()
        names = [m.get("name", "") for m in models if m.get("name")]
        if not names:
            return {"vision": [], "text": [], "all": []}

        # Query capabilities for all models in parallel
        caps = await asyncio.gather(*(self.get_model_capabilities(n) for n in names))

        vision = []
        text = []
        for name, model_caps in zip(names, caps):
            # vision.append(name)
            # text.append(name)
            if model_caps is None or not caps:
                # Capabilities unknown (e.g. OpenAI-compatible server without
                # /api/show). Present the model in both lists so the user can
                # select it as a vision model even if we can't confirm it.
                logger.warning(
                    "Capabilities unknown for %s — presenting in both vision and text lists",
                    name,
                )
                vision.append(name)
                text.append(name)
            elif "vision" in model_caps:
                logger.debug(
                    "Vision and text capability for %s",
                    name,
                )
                vision.append(name)
                text.append(name)
            else:
                logger.debug(
                    "Text capability for %s",
                    name,
                )
                text.append(name)

        return {"vision": vision, "text": text, "all": names}

    # -- Model Pull ----------------------------------------------------------

    async def pull_model_stream(self, model_name: str):
        """Stream pull progress from Ollama's /api/pull endpoint.
        Yields NDJSON lines with enriched percentage field.

        /api/pull is Ollama-native; OpenAI-compatible servers do not expose it.
        If the endpoint is unavailable a single error line is yielded and the
        stream ends gracefully instead of raising.
        """
        import json as _json

        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST",
                    f"{self._ollama_base}/api/pull",
                    json={"name": model_name, "stream": True},
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = _json.loads(line)
                            if (
                                obj.get("total")
                                and obj.get("completed")
                                and "percentage" not in obj
                            ):
                                obj["percentage"] = round(
                                    (obj["completed"] / obj["total"]) * 100
                                )
                            yield _json.dumps(obj) + "\n"
                        except _json.JSONDecodeError:
                            yield line + "\n"
        except httpx.HTTPError as exc:
            yield _json.dumps(
                {
                    "status": "error",
                    "error": f"model pull not supported by this endpoint ({exc.__class__.__name__}: {exc})",
                }
            )
            return

    # -- Generation ----------------------------------------------------------

    def _chat_payload(
        self,
        model: str,
        prompt: str,
        images: list[str] | None = None,
        options: dict | None = None,
    ) -> dict[str, Any]:
        """Build an OpenAI-compatible chat.completions payload.

        Images (base64 JPEG data URLs) are embedded as ``image_url`` content
        parts so vision-capable models receive the image alongside the prompt.
        Ollama ``options`` are mapped to OpenAI parameters where sensible.
        """
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        if images:
            for b64 in images:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    }
                )

        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "stream": False,
        }

        if options:
            temperature = options.get("temperature")
            if temperature is not None:
                payload["temperature"] = temperature
            top_p = options.get("top_p")
            if top_p is not None:
                payload["top_p"] = top_p
            # Ollama uses num_predict for max tokens.
            num_predict = options.get("num_predict")
            if num_predict is not None:
                payload["max_tokens"] = num_predict

        return payload

    async def _generate(
        self,
        model: str,
        prompt: str,
        images: list[str] | None = None,
        options: dict | None = None,
    ) -> str:
        """Low-level generate call. Returns the response text.

        Prefers the OpenAI-compatible /v1/chat/completions endpoint and falls
        back to the Ollama-native /api/generate endpoint on connection-level
        failures.
        """
        payload = self._chat_payload(model, prompt, images, options)

        # --- Primary: OpenAI-compatible /v1/chat/completions ---------------
        try:
            resp = await self._client.post(
                f"{self._v1_base}/chat/completions",
                json=payload,
                headers=self._auth_headers(),
                timeout=_GENERATE_TIMEOUT,
            )
            if resp.status_code == 404:
                # /v1 not supported by this server; fall back to Ollama-native.
                raise _V1Unsupported()
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except _V1Unsupported:
            pass
        except httpx.TimeoutException:
            raise RuntimeError(
                f"Timed out after {int(_GENERATE_TIMEOUT)}s — "
                "the model may still be loading or the GPU may be busy"
            )
        except httpx.HTTPError as exc:
            # Only fall back on connection-level errors, not on 400-level
            # model errors (those are surfaced to the caller).
            if (
                isinstance(exc, httpx.HTTPStatusError)
                and exc.response is not None
                and exc.response.status_code >= 400
            ):
                raise RuntimeError(
                    f"OpenAI-compatible API error ({exc.response.status_code}): "
                    f"{_error_detail(exc.response)}"
                )
            raise RuntimeError(
                f"OpenAI-compatible connection error: {exc.__class__.__name__}: {exc}"
            )

        # --- Fallback: Ollama-native /api/generate -------------------------
        ollama_payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
        if images:
            ollama_payload["images"] = images
        if options:
            ollama_payload["options"] = options

        try:
            resp = await self._client.post(
                f"{self._ollama_base}/api/generate",
                json=ollama_payload,
                timeout=_GENERATE_TIMEOUT,
            )
        except httpx.TimeoutException:
            raise RuntimeError(
                f"Timed out after {int(_GENERATE_TIMEOUT)}s — "
                "the model may still be loading or the GPU may be busy"
            )
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"Ollama connection error: {exc.__class__.__name__}: {exc}"
            )
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("error", "") or resp.text[:300]
            except Exception:
                detail = resp.text[:300]
            raise RuntimeError(f"Ollama error ({resp.status_code}): {detail}")
        data = resp.json()
        return data.get("response", "").strip()

    # -- Vision (describe + name) -------------------------------------------

    async def describe_and_name_image(
        self,
        image_path: Path,
        metadata: dict[str, Any],
        include_tags: bool = False,
        processing_context: str | None = None,
        output_language: str = "",
    ) -> tuple[str, str, list[str], list[str]]:
        """Single-model mode: vision model produces description, filename, tags, and quality flags.

        Returns (description, suggested_filename, tags, quality_flags).
        """
        from backend.prompts import (
            render_vision_prompt,
            DEFAULT_VISION_PROMPT,
            DEFAULT_CONTEXT_TEMPLATE,
        )

        meta_lines = _format_metadata_for_prompt(metadata)

        vision_tmpl = self._vision_template or DEFAULT_VISION_PROMPT
        context_tmpl = self._context_template or DEFAULT_CONTEXT_TEMPLATE

        prompt = render_vision_prompt(
            vision_tmpl,
            context_tmpl,
            metadata_text=meta_lines,
            include_tags=include_tags,
            processing_context=processing_context,
            output_language=output_language,
        )

        image_b64 = _encode_image(image_path)
        raw = await self._generate(
            model=self.vision_model,
            prompt=prompt,
            images=[image_b64],
        )

        parsed = _parse_response(raw)
        return (
            parsed["description"],
            parsed["filename"],
            parsed["tags"],
            parsed["quality_flags"],
        )


# -- Helpers -----------------------------------------------------------------


def _encode_image(image_path: Path) -> str:
    """Downscale an image and return its base64-encoded JPEG content.

    Opens the original file via open_image() (supports JPEG, PNG, HEIC, RAW, etc.),
    resizes to fit within _VISION_MAX_PX on the longest side, and encodes as JPEG
    in memory. The original file is never modified.
    """
    img = open_image(image_path)

    # Only downscale — don't upscale small images
    if max(img.size) > _VISION_MAX_PX:
        img.thumbnail((_VISION_MAX_PX, _VISION_MAX_PX), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=_VISION_JPEG_QUALITY)

    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _format_metadata_for_prompt(metadata: dict[str, Any]) -> str:
    """Format metadata dict into readable lines for the LLM prompt."""
    lines = ["Metadata:"]
    if metadata.get("date"):
        lines.append(f"- Date: {metadata['date']}")
    if metadata.get("location"):
        lines.append(f"- Location: {metadata['location']}")
    if metadata.get("camera_model"):
        lines.append(f"- Camera: {metadata['camera_model']}")
    if not any(metadata.get(k) for k in ("date", "location", "camera_model")):
        lines.append("- No metadata available")
    return "\n".join(lines)


_QUALITY_OK_WORDS = {
    "ok",
    "okay",
    "good",
    "fine",
    "clear",
    "normal",
    "none",
    "n/a",
    "na",
}


def _quality_is_ok(raw: str) -> bool:
    """Return True if the quality response means 'no issues'."""
    raw = raw.strip().lower().rstrip(".")
    if raw in _QUALITY_OK_WORDS:
        return True
    if any(
        phrase in raw
        for phrase in (
            "no issue",
            "no flaw",
            "no visible",
            "no problem",
            "good quality",
            "looks good",
            "looks fine",
            "no defect",
            "well exposed",
            "properly exposed",
        )
    ):
        return True
    return False


def _parse_response(raw: str) -> dict[str, Any]:
    """Parse a structured response with DESCRIPTION, FILENAME, TAGS, and QUALITY lines.

    Expected format:
        DESCRIPTION: A woman walking through a cobblestone alley
        FILENAME: woman-cobblestone-alley
        TAGS: woman, cobblestone, alley, europe, walking
        QUALITY: ok

    Falls back gracefully if the model doesn't follow the format exactly.
    Returns dict with keys: description, filename, tags, quality_flags.
    """
    # Pre-process: ensure each marker starts on its own line.
    # Handles models that return everything on one line, or omit newlines
    # between some markers (e.g. "description textFILENAME: name").
    processed = re.sub(
        r"(?i)(?<!\n)(DESCRIPTION:|FILENAME:|TAGS:|QUALITY:)", r"\n\1", raw
    ).lstrip("\n")

    description = ""
    filename = ""
    tags: list[str] = []
    quality_flags: list[str] = []

    for line in processed.splitlines():
        line = line.strip()
        upper = line.upper()
        if upper.startswith("DESCRIPTION:"):
            description = line.split(":", 1)[1].strip()
        elif upper.startswith("FILENAME:"):
            filename = line.split(":", 1)[1].strip()
        elif upper.startswith("TAGS:"):
            tags_raw = line.split(":", 1)[1].strip()
            tags = [t.strip().lower() for t in tags_raw.split(",") if t.strip()]
        elif upper.startswith("QUALITY:"):
            quality_raw = line.split(":", 1)[1].strip().lower()
            if quality_raw and not _quality_is_ok(quality_raw):
                quality_flags = [
                    q.strip()
                    for q in quality_raw.split(",")
                    if q.strip() and not _quality_is_ok(q.strip())
                ]

    # Fallback: if parsing failed, use the whole response as description
    if not description and not filename:
        description = raw.strip()
        words = description.split()[:6]
        filename = "-".join(w.lower() for w in words)

    if not filename and description:
        words = description.split()[:6]
        filename = "-".join(w.lower() for w in words)

    return {
        "description": description,
        "filename": filename,
        "tags": tags,
        "quality_flags": quality_flags,
    }
