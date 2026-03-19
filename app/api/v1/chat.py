"""
Chat Completions API 路由。
"""

from __future__ import annotations

from typing import Any, AsyncGenerator, AsyncIterable, Dict, List, Optional, Union

import orjson
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from app.core.auth import verify_api_key
from app.core.exceptions import AppException, ValidationException
from app.services.grok.chat import ChatService
from app.services.grok.model import ModelService
from app.services.quota import enforce_daily_quota


router = APIRouter(tags=["Chat"])


VALID_ROLES = ("developer", "system", "user", "assistant")
USER_CONTENT_TYPES = ("text", "image_url", "input_audio", "file")


class MessageItem(BaseModel):
    """聊天消息。"""

    role: str
    content: Optional[Union[str, Dict[str, Any], List[Dict[str, Any]]]]

    @field_validator("role")
    @classmethod
    def validate_role(cls, value: str) -> str:
        if value not in VALID_ROLES:
            raise ValueError(f"role must be one of {list(VALID_ROLES)}")
        return value


class VideoConfig(BaseModel):
    """视频生成配置。"""

    aspect_ratio: Optional[str] = Field("3:2", description="视频比例: 3:2, 16:9, 1:1 等")
    video_length: Optional[int] = Field(6, description="视频时长(秒): 5-15")
    resolution: Optional[str] = Field("SD", description="视频分辨率: SD, HD")
    preset: Optional[str] = Field("custom", description="风格预设: fun, normal, spicy, custom")

    @field_validator("aspect_ratio")
    @classmethod
    def validate_aspect_ratio(cls, value: Optional[str]) -> Optional[str]:
        allowed = ["2:3", "3:2", "1:1", "9:16", "16:9"]
        if value and value not in allowed:
            raise ValidationException(
                message=f"aspect_ratio must be one of {allowed}",
                param="video_config.aspect_ratio",
                code="invalid_aspect_ratio",
            )
        return value

    @field_validator("video_length")
    @classmethod
    def validate_video_length(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and (value < 5 or value > 15):
            raise ValidationException(
                message="video_length must be between 5 and 15 seconds",
                param="video_config.video_length",
                code="invalid_video_length",
            )
        return value

    @field_validator("resolution")
    @classmethod
    def validate_resolution(cls, value: Optional[str]) -> Optional[str]:
        allowed = ["SD", "HD"]
        if value and value not in allowed:
            raise ValidationException(
                message=f"resolution must be one of {allowed}",
                param="video_config.resolution",
                code="invalid_resolution",
            )
        return value

    @field_validator("preset")
    @classmethod
    def validate_preset(cls, value: Optional[str]) -> str:
        if not value:
            return "custom"
        allowed = ["fun", "normal", "spicy", "custom"]
        if value not in allowed:
            raise ValidationException(
                message=f"preset must be one of {allowed}",
                param="video_config.preset",
                code="invalid_preset",
            )
        return value


class ChatCompletionRequest(BaseModel):
    """Chat Completions 请求。"""

    model: str = Field(..., description="模型名称")
    messages: List[MessageItem] = Field(..., description="消息数组")
    stream: Optional[bool] = Field(None, description="是否流式输出")
    thinking: Optional[str] = Field(None, description="思考模式: enabled / disabled / None")
    video_config: Optional[VideoConfig] = Field(None, description="视频生成参数")

    model_config = {"extra": "ignore"}


def _validate_text_block(block: Dict[str, Any], idx: int, block_idx: int) -> None:
    if not isinstance(block, dict):
        raise ValidationException(
            message="Content block must be an object",
            param=f"messages.{idx}.content.{block_idx}",
            code="invalid_block",
        )

    block_type = block.get("type")
    if block_type != "text":
        raise ValidationException(
            message="When content is an object, type must be 'text'",
            param=f"messages.{idx}.content.{block_idx}.type",
            code="invalid_content_type",
        )

    text = block.get("text", "")
    if not isinstance(text, str) or not text.strip():
        raise ValidationException(
            message="Text content cannot be empty",
            param=f"messages.{idx}.content.{block_idx}.text",
            code="empty_text",
        )


def validate_request(request: ChatCompletionRequest) -> None:
    """验证请求参数。"""

    if not ModelService.valid(request.model):
        raise ValidationException(
            message=f"The model `{request.model}` does not exist or you do not have access to it.",
            param="model",
            code="model_not_found",
        )

    for idx, msg in enumerate(request.messages):
        content = msg.content

        # 兼容 assistant 中间态空内容。
        if content is None:
            if msg.role == "assistant":
                continue
            raise ValidationException(
                message="Message content cannot be null",
                param=f"messages.{idx}.content",
                code="empty_content",
            )

        if isinstance(content, str):
            if not content.strip():
                raise ValidationException(
                    message="Message content cannot be empty",
                    param=f"messages.{idx}.content",
                    code="empty_content",
                )
            continue

        if isinstance(content, dict):
            _validate_text_block(content, idx, 0)
            continue

        if not isinstance(content, list):
            raise ValidationException(
                message="Message content must be a string, object, or array",
                param=f"messages.{idx}.content",
                code="invalid_content",
            )

        if not content:
            raise ValidationException(
                message="Message content cannot be an empty array",
                param=f"messages.{idx}.content",
                code="empty_content",
            )

        for block_idx, block in enumerate(content):
            if not isinstance(block, dict):
                raise ValidationException(
                    message="Content block must be an object",
                    param=f"messages.{idx}.content.{block_idx}",
                    code="invalid_block",
                )
            if not block:
                raise ValidationException(
                    message="Content block cannot be empty",
                    param=f"messages.{idx}.content.{block_idx}",
                    code="empty_block",
                )

            if "type" not in block:
                raise ValidationException(
                    message="Content block must have a 'type' field",
                    param=f"messages.{idx}.content.{block_idx}",
                    code="missing_type",
                )

            block_type = block.get("type")
            if not isinstance(block_type, str) or not block_type.strip():
                raise ValidationException(
                    message="Content block 'type' cannot be empty",
                    param=f"messages.{idx}.content.{block_idx}.type",
                    code="empty_type",
                )

            if msg.role == "user":
                if block_type not in USER_CONTENT_TYPES:
                    raise ValidationException(
                        message=f"Invalid content block type: '{block_type}'",
                        param=f"messages.{idx}.content.{block_idx}.type",
                        code="invalid_type",
                    )
            elif block_type != "text":
                raise ValidationException(
                    message=f"The `{msg.role}` role only supports 'text' type, got '{block_type}'",
                    param=f"messages.{idx}.content.{block_idx}.type",
                    code="invalid_type",
                )

            if block_type == "text":
                text = block.get("text", "")
                if not isinstance(text, str) or not text.strip():
                    raise ValidationException(
                        message="Text content cannot be empty",
                        param=f"messages.{idx}.content.{block_idx}.text",
                        code="empty_text",
                    )
            elif block_type == "image_url":
                image_url = block.get("image_url")
                if not image_url or not (isinstance(image_url, dict) and image_url.get("url")):
                    raise ValidationException(
                        message="image_url must have a 'url' field",
                        param=f"messages.{idx}.content.{block_idx}.image_url",
                        code="missing_url",
                    )


def _stream_error_payload(exc: Exception) -> Dict[str, Any]:
    if isinstance(exc, AppException):
        return {
            "error": {
                "message": exc.message,
                "type": exc.error_type,
                "code": exc.code,
            }
        }

    return {
        "error": {
            "message": str(exc) or "stream_error",
            "type": "server_error",
            "code": "stream_error",
        }
    }


async def _safe_sse_stream(stream: AsyncIterable[str]) -> AsyncGenerator[str, None]:
    """将流式异常转换成 SSE 错误事件。"""

    try:
        async for chunk in stream:
            yield chunk
    except Exception as exc:
        payload = _stream_error_payload(exc)
        yield f"event: error\ndata: {orjson.dumps(payload).decode()}\n\n"
        yield "data: [DONE]\n\n"


def _streaming_error_response(exc: Exception) -> StreamingResponse:
    payload = _stream_error_payload(exc)

    async def _one_shot_error() -> AsyncGenerator[str, None]:
        yield f"event: error\ndata: {orjson.dumps(payload).decode()}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        _one_shot_error(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@router.post("/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    api_key: Optional[str] = Depends(verify_api_key),
):
    """Chat Completions API - OpenAI 兼容入口。"""

    validate_request(request)
    await enforce_daily_quota(api_key, request.model)

    model_info = ModelService.get(request.model)
    if model_info and model_info.is_video:
        from app.services.grok.media import VideoService

        video_conf = request.video_config or VideoConfig()
        try:
            result = await VideoService.completions(
                model=request.model,
                messages=[msg.model_dump() for msg in request.messages],
                stream=request.stream,
                thinking=request.thinking,
                aspect_ratio=video_conf.aspect_ratio,
                video_length=video_conf.video_length,
                resolution=video_conf.resolution,
                preset=video_conf.preset,
            )
        except Exception as exc:
            if request.stream is not False:
                return _streaming_error_response(exc)
            raise
    else:
        try:
            result = await ChatService.completions(
                model=request.model,
                messages=[msg.model_dump() for msg in request.messages],
                stream=request.stream,
                thinking=request.thinking,
            )
        except Exception as exc:
            if request.stream is not False:
                return _streaming_error_response(exc)
            raise

    if isinstance(result, dict):
        return JSONResponse(content=result)

    return StreamingResponse(
        _safe_sse_stream(result),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


__all__ = ["router", "validate_request", "chat_completions", "_safe_sse_stream"]
