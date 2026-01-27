import asyncio
import json
import traceback
from itertools import product
from pathlib import Path
from typing import Any

import json_repair
import yaml
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from openai.types.images_response import ImagesResponse
from pydantic import BaseModel, Field, PrivateAttr, ValidationError

from deeppresenter.utils.constants import MIN_IMAGE_SIZE, PACKAGE_DIR, RETRY_TIMES
from deeppresenter.utils.log import debug, error, logging_openai_exceptions


def get_json_from_response(response: str) -> dict | list:
    """
    Extract JSON from a text response.

    Args:
        response (str): The response text.

    Returns:
        Dict|List: The extracted JSON.

    Raises:
        Exception: If JSON cannot be extracted from the response.
    """

    assert isinstance(response, str) and len(response) > 0, (
        "response must be a non-empty string"
    )
    response = response.strip()
    try:
        return json.loads(response)
    except:
        pass

    # Try to find JSON by looking for matching braces
    open_braces = []
    close_braces = []

    for i, char in enumerate(response):
        if char == "{" or char == "[":
            open_braces.append(i)
        elif char == "}" or char == "]":
            close_braces.append(i)

    for i, j in product(open_braces, reversed(close_braces)):
        if i > j:
            continue
        try:
            json_obj = json.loads(response[i : j + 1])
            if isinstance(json_obj, (dict, list)):
                return max(
                    json_obj, json_repair.loads(response), key=lambda x: len(str(x))
                )
        except Exception:
            pass

    return json_repair.loads(response)


class LLM(BaseModel):
    """LLM Client Manager"""

    base_url: str = Field(description="API base URL")
    model: str = Field(description="Model name")
    api_key: str = Field(description="API key")
    is_multimodal: bool = Field(
        default=False, description="Whether the model is multimodal"
    )
    max_concurrent: int | None = Field(
        default=None, description="Maximum concurrency limit"
    )
    client_kwargs: dict[str, Any] = Field(
        default_factory=dict, description="Client parameters"
    )
    sampling_parameters: dict[str, Any] = Field(
        default_factory=dict, description="Sampling parameters"
    )
    soft_response_parsing: bool = Field(
        default=False,
        description="Enable soft parsing: parse response content as JSON directly instead of using completion.parse",
    )

    # Fallback configuration
    fallback_base_url: str | None = Field(
        default=None, description="Fallback API base URL"
    )
    fallback_model: str | None = Field(default=None, description="Fallback model name")
    fallback_api_key: str | None = Field(default=None, description="Fallback API key")
    fallback_client_kwargs: dict[str, Any] = Field(
        default_factory=dict, description="Fallback client parameters"
    )
    fallback_sampling_parameters: dict[str, Any] = Field(
        default_factory=dict, description="Fallback sampling parameters"
    )

    _semaphore: asyncio.Semaphore = PrivateAttr()
    _client: AsyncOpenAI = PrivateAttr()
    _fallback_client: AsyncOpenAI | None = PrivateAttr(default=None)

    model_config = {"arbitrary_types_allowed": True}

    @property
    def model_name(self) -> str:
        if "/" in self.model:
            return self.model.split("/")[-1]
        return self.model

    @property
    def has_fallback(self) -> bool:
        """Check if fallback is configured"""
        return all(
            [
                self.fallback_base_url,
                self.fallback_model,
                self.fallback_api_key,
            ]
        )

    def model_post_init(self, _) -> None:
        """Initialize semaphore and clients"""
        self._semaphore = asyncio.Semaphore(self.max_concurrent or 10000)
        self._client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            **self.client_kwargs,
        )

        # 初始化 fallback client
        if self.has_fallback:
            self._fallback_client = AsyncOpenAI(
                api_key=self.fallback_api_key,
                base_url=self.fallback_base_url,
                **self.fallback_client_kwargs,
            )

        model_lower = self.model.lower()
        if not self.is_multimodal and any(
            word in model_lower for word in ("gpt", "claude", "gemini", "vl")
        ):
            self.is_multimodal = True
            debug(
                f"Model {self.model} is detected as multimodal model, setting `is_multimodal` to True"
            )

    async def _call(
        self,
        client: AsyncOpenAI,
        model: str,
        messages: list[dict[str, Any]],
        sampling_params: dict[str, Any],
        response_format: type[BaseModel] | None = None,
        tools: list[dict[str, Any]] | None = None,
        retry_times: int = RETRY_TIMES,
    ) -> ChatCompletion:
        """Execute chat or tool call using the specified client"""
        last_error: Exception | None = None
        for retry_idx in range(retry_times):
            await asyncio.sleep(2**retry_idx - 1)
            try:
                if tools is not None:
                    response = await client.chat.completions.create(
                        model=model,
                        messages=messages,
                        tools=tools,
                        tool_choice="required",  # Force model to call at least one tool
                        **sampling_params,
                    )
                elif not self.soft_response_parsing and response_format is not None:
                    response: ChatCompletion = await client.chat.completions.parse(
                        model=model,
                        messages=messages,
                        response_format=response_format,
                        **sampling_params,
                    )
                else:
                    response: ChatCompletion = await client.chat.completions.create(
                        model=model,
                        messages=messages,
                        **sampling_params,
                    )
                assert response.choices is not None and len(response.choices) > 0, (
                    "No choices returned from the model"
                )
                message = response.choices[0].message
                if response_format is not None:
                    message.content = response_format(
                        **get_json_from_response(message.content)
                    ).model_dump_json(indent=2)
                assert tools is None or message.tool_calls, (
                    "No tool call returned from the model"
                )
                assert message.tool_calls or message.content, (
                    "Empty content returned from the model"
                )
                return response
            except (AssertionError, ValidationError) as e:
                last_error = e
            except Exception as e:
                last_error = e
                logging_openai_exceptions(model, e)
        error_msg = str(last_error) if last_error else "Unknown error after retries"
        error(f"Model {model} failed for: {error_msg}")
        raise ValueError(f"{model} cannot get valid response from the model: {error_msg}")

    async def run(
        self,
        messages: list[dict[str, Any]] | str,
        response_format: type[BaseModel] | None = None,
        tools: list[dict[str, Any]] | None = None,
        retry_times: int = RETRY_TIMES,
    ) -> ChatCompletion:
        """Unified interface for chat and tool calls"""
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        async with self._semaphore:
            try:
                return await self._call(
                    self._client,
                    self.model,
                    messages,
                    self.sampling_parameters,
                    response_format,
                    tools,
                    retry_times,
                )
            except Exception as e:
                if self._fallback_client is not None:
                    debug(
                        f"Primary model {self.model} failed, trying fallback model {self.fallback_model}"
                    )
                    return await self._call(
                        self._fallback_client,
                        self.fallback_model,
                        messages,
                        self.fallback_sampling_parameters,
                        response_format,
                        tools,
                        retry_times,
                    )
                raise e

    async def generate_image(
        self,
        prompt: str,
        width: int,
        height: int,
        retry_times: int = RETRY_TIMES,
    ) -> ImagesResponse:
        """Unified interface for image generation
        
        Supports two modes:
        1. OpenAI images.generate API (for DALL-E, Volcengine, etc.)
        2. Chat completions API (for Gemini and other multimodal models)
        """
        if MIN_IMAGE_SIZE is not None and (width * height) < int(MIN_IMAGE_SIZE):
            ratio = (int(MIN_IMAGE_SIZE) / (width * height)) ** 0.5
            width = int(width * ratio)
            height = int(height * ratio)
        width = ((width + 15) // 16) * 16
        height = ((height + 15) // 16) * 16
        
        # Check if using chat completions mode (for Gemini-style models)
        use_chat_mode = self.sampling_parameters.get("use_chat_completions", False)
        
        async with self._semaphore:
            for retry_idx in range(retry_times):
                await asyncio.sleep(retry_idx)
                try:
                    if use_chat_mode:
                        return await self._generate_image_via_chat(prompt, width, height)
                    else:
                        return await self._generate_image_via_images_api(prompt, width, height)
                except Exception as e:
                    logging_openai_exceptions(self.model, e)
            raise ValueError("Cannot generate image")

    async def _generate_image_via_images_api(
        self,
        prompt: str,
        width: int,
        height: int,
    ) -> ImagesResponse:
        """Generate image using OpenAI images.generate API"""
        params = {k: v for k, v in self.sampling_parameters.items() if k != "use_chat_completions"}
        return await self._client.images.generate(
            prompt=prompt,
            model=self.model,
            size=f"{width}x{height}",
            **params,
        )

    def _get_closest_aspect_ratio(self, width: int, height: int) -> str:
        """Convert width/height to closest supported Gemini aspect ratio"""
        # Supported aspect ratios for Gemini
        supported_ratios = [
            ("1:1", 1.0),
            ("2:3", 2/3),
            ("3:2", 3/2),
            ("3:4", 3/4),
            ("4:3", 4/3),
            ("4:5", 4/5),
            ("5:4", 5/4),
            ("9:16", 9/16),
            ("16:9", 16/9),
            ("21:9", 21/9),
        ]
        
        target_ratio = width / height
        
        # Find closest ratio
        closest = min(supported_ratios, key=lambda x: abs(x[1] - target_ratio))
        return closest[0]

    def _get_image_size(self, width: int, height: int) -> str:
        """Determine Gemini image size based on dimensions"""
        # Max dimension determines the size tier
        max_dim = max(width, height)
        
        if max_dim <= 1024:
            return "1K"
        elif max_dim <= 2048:
            return "2K"
        else:
            return "4K"

    async def _generate_image_via_chat(
        self,
        prompt: str,
        width: int,
        height: int,
    ) -> ImagesResponse:
        """Generate image using chat completions API (for Gemini-style models)
        
        Note: Gemini API uses aspectRatio + imageSize instead of exact pixel dimensions.
        Supported aspectRatios: 1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9
        Supported imageSizes: 1K, 2K, 4K
        """
        import base64
        import math
        import re
        from openai.types.images_response import Image
        
        # Convert width/height to closest aspect ratio
        aspect_ratio = self._get_closest_aspect_ratio(width, height)
        
        # Determine image size based on dimensions
        image_size = self._get_image_size(width, height)
        
        # Build the image generation prompt with size instruction in text
        # This helps when API proxy doesn't support generationConfig
        generation_prompt = f"Generate a {aspect_ratio} aspect ratio image ({image_size} resolution). Description: {prompt}"
        
        messages = [
            {
                "role": "user",
                "content": generation_prompt,
            }
        ]
        
        # Build extra body with Gemini-specific parameters
        extra_body = {
            "generationConfig": {
                "responseModalities": ["TEXT", "IMAGE"],
                "imageConfig": {
                    "aspectRatio": aspect_ratio,
                    "imageSize": image_size,
                }
            }
        }
        
        # Log the request parameters for debugging
        debug(f"[T2I] Requesting aspect_ratio={aspect_ratio}, image_size={image_size} (from {width}x{height})")
        
        # Call chat completions with Gemini-specific config
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            extra_body=extra_body,
        )
        
        # Extract image from response
        content = response.choices[0].message.content
        
        # Try to find base64 image data in the response
        # Gemini returns images in various formats, try to extract them
        image_data = None
        image_url = None
        
        # Pattern 1: Look for inline base64 image data (data:image/...;base64,...)
        base64_pattern = r'data:image/[^;]+;base64,([A-Za-z0-9+/=]+)'
        match = re.search(base64_pattern, content or "")
        if match:
            image_data = match.group(1)
        
        # Pattern 2: Check if response contains raw base64 (no data URI prefix)
        if not image_data and content:
            # Check if content looks like base64
            clean_content = content.strip()
            if re.match(r'^[A-Za-z0-9+/=]+$', clean_content) and len(clean_content) > 1000:
                image_data = clean_content
        
        # Pattern 3: Look for image URL in response
        if not image_data:
            url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+\.(?:png|jpg|jpeg|gif|webp)'
            url_match = re.search(url_pattern, content or "", re.IGNORECASE)
            if url_match:
                image_url = url_match.group(0)
        
        # Pattern 4: Check if the model returned structured content with image parts
        # This handles cases where the API returns images in a structured format
        if not image_data and hasattr(response.choices[0].message, 'content'):
            # Some APIs return content as a list of parts
            msg = response.choices[0].message
            if hasattr(msg, 'parts'):
                for part in msg.parts:
                    if hasattr(part, 'inline_data'):
                        image_data = part.inline_data.data
                        break
        
        if not image_data and not image_url:
            raise ValueError(f"No image found in response. Content: {content[:500] if content else 'None'}...")
        
        # Build ImagesResponse compatible object
        return ImagesResponse(
            created=int(asyncio.get_event_loop().time()),
            data=[
                Image(
                    b64_json=image_data,
                    url=image_url,
                    revised_prompt=None,
                )
            ],
        )

    async def validate(self):
        models = await self._client.models.list()
        # ? This for compatibility with google generative ai
        if not any(model.id.endswith(self.model) for model in models.data):
            raise Exception(
                f"Model {self.model} is not available at {self.base_url}, please check your apikey or {PACKAGE_DIR / 'config.yaml'}\n"
            )


class DeepPresenterConfig(BaseModel):
    """DeepPresenter Global Configuration"""

    mcp_config_file: str = Field(
        description="MCP configuration file", default=PACKAGE_DIR / "mcp.json"
    )
    research_agent: LLM = Field(description="Research agent model configuration")
    design_agent: LLM = Field(description="Design agent model configuration")
    long_context_model: LLM = Field(description="Long context model configuration")
    vision_model: LLM = Field(description="Vision model configuration")
    t2i_model: LLM = Field(description="Text-to-image model configuration")

    @classmethod
    def load_from_file(cls, config_path: str | None = None) -> "DeepPresenterConfig":
        """Load configuration from file"""
        if config_path:
            config_file = Path(config_path)
        else:
            config_file = PACKAGE_DIR / "config.yaml"

        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file {config_file} does not exist")
        config_data = {}
        with open(config_file, encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}

        return cls(**config_data)

    async def validate_llms(self):
        # ? we do not valite t2i model since some providers like volcengine did not open this endpoint
        await asyncio.gather(
            self.research_agent.validate(),
            self.design_agent.validate(),
            self.long_context_model.validate(),
            self.vision_model.validate(),
        )

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


GLOBAL_CONFIG = DeepPresenterConfig.load_from_file()
