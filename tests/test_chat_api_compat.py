import asyncio
from types import SimpleNamespace

import pytest
from fastapi.responses import StreamingResponse

from app.api.v1 import chat as chat_api
from app.core.exceptions import AppException
from app.services.grok.chat import MessageExtractor


async def _collect_chunks(stream):
    chunks = []
    async for chunk in stream:
        if isinstance(chunk, bytes):
            chunks.append(chunk.decode())
        else:
            chunks.append(chunk)
    return chunks


def test_validate_request_accepts_assistant_null_content(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(chat_api.ModelService, "valid", lambda _model_id: True)
    request = chat_api.ChatCompletionRequest(
        model="grok-4",
        messages=[chat_api.MessageItem(role="assistant", content=None)],
    )

    chat_api.validate_request(request)


def test_validate_request_accepts_openai_like_text_object(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(chat_api.ModelService, "valid", lambda _model_id: True)
    request = chat_api.ChatCompletionRequest(
        model="grok-4",
        messages=[
            chat_api.MessageItem(
                role="user",
                content={"type": "text", "text": "hello"},
            )
        ],
    )

    chat_api.validate_request(request)


def test_validate_request_rejects_null_user_content(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(chat_api.ModelService, "valid", lambda _model_id: True)
    request = chat_api.ChatCompletionRequest(
        model="grok-4",
        messages=[chat_api.MessageItem(role="user", content=None)],
    )

    with pytest.raises(chat_api.ValidationException) as exc:
        chat_api.validate_request(request)

    assert exc.value.code == "empty_content"


def test_message_extractor_handles_dict_content_and_assistant_null():
    message, attachments = MessageExtractor.extract(
        [
            {"role": "assistant", "content": None},
            {"role": "user", "content": {"type": "text", "text": "describe this"}},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
                ],
            },
        ]
    )

    assert message == "describe this"
    assert attachments == [("image", "https://example.com/cat.png")]


def test_safe_sse_stream_returns_error_event_for_app_exception():
    async def _broken_stream():
        raise AppException(
            message="upstream broken",
            error_type="server_error",
            code="stream_error",
        )
        yield "unreachable"

    chunks = asyncio.run(_collect_chunks(chat_api._safe_sse_stream(_broken_stream())))

    assert len(chunks) == 2
    assert chunks[0].startswith("event: error")
    assert '"message":"upstream broken"' in chunks[0]
    assert chunks[1] == "data: [DONE]\n\n"


def test_chat_completions_returns_sse_error_when_stream_setup_fails(monkeypatch: pytest.MonkeyPatch):
    async def _noop_quota(_api_key, _model):
        return None

    async def _fail_chat(**_kwargs):
        raise AppException(
            message="token unavailable",
            error_type="rate_limit_error",
            code="rate_limit_exceeded",
            status_code=429,
        )

    monkeypatch.setattr(chat_api, "enforce_daily_quota", _noop_quota)
    monkeypatch.setattr(chat_api.ModelService, "valid", lambda _model_id: True)
    monkeypatch.setattr(
        chat_api.ModelService,
        "get",
        lambda _model_id: SimpleNamespace(is_video=False),
    )
    monkeypatch.setattr(chat_api.ChatService, "completions", _fail_chat)

    request = chat_api.ChatCompletionRequest(
        model="grok-4",
        messages=[chat_api.MessageItem(role="user", content="hello")],
        stream=True,
    )

    response = asyncio.run(chat_api.chat_completions(request, api_key="test-key"))

    assert isinstance(response, StreamingResponse)
    chunks = asyncio.run(_collect_chunks(response.body_iterator))
    assert len(chunks) == 2
    assert chunks[0].startswith("event: error")
    assert '"code":"rate_limit_exceeded"' in chunks[0]
    assert chunks[1] == "data: [DONE]\n\n"
