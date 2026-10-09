import json
from typing import Final

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from integration._support.client import Gateway, object_value
from integration._support.wire import Reply, Request, wire_server

_BACKEND: Final = "gemini-3.7-flash"
_PROJECT: Final = "scripted-project"
_LOCATION: Final = "us-central1"
_MODEL_PATH: Final = f"/v1/projects/{_PROJECT}/locations/{_LOCATION}/publishers/google/models/{_BACKEND}"
_STREAM_TARGET: Final = f"{_MODEL_PATH}:streamGenerateContent?alt=sse"
_EVENT_PAYLOAD: Final = '{"candidates":[{"content":{"role":"model","parts":[{"text":"pong"}]}}]}'
_EVENT: Final = f"data: {_EVENT_PAYLOAD}\n\n".encode()


def _service_account_json(token_url: str) -> str:
    private_key: Final = (
        rsa.generate_private_key(public_exponent=65537, key_size=2048)
        .private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
        .decode()
    )
    return json.dumps(
        {
            "type": "service_account",
            "project_id": _PROJECT,
            "private_key_id": "scripted",
            "private_key": private_key,
            "client_email": f"scripted@{_PROJECT}.iam.gserviceaccount.com",
            "client_id": "0",
            "auth_uri": f"{token_url}/_oauth/authorize",
            "token_uri": f"{token_url}/_oauth/token",
        }
    )


def test_vertex_generate_content_stream_dropped_before_first_chunk_is_retried(gateway: Gateway) -> None:
    """A 200 whose connection closes before the first SSE event must be retried per num_retries.

    Regression for https://github.com/BerriAI/litellm/issues/45457: the router's
    retry loop used to never see this failure because agenerate_content_stream
    returned as soon as the upstream response headers arrived, and the first
    body read happened later in proxy_server.async_data_generator, which folds
    the error into an SSE error frame with no retry.
    """
    attempts: Final[list[Request]] = []

    def respond(request: Request) -> Reply:
        assert request.method == "POST"
        assert request.target == _STREAM_TARGET
        attempts.append(request)
        if len(attempts) == 1:
            return Reply(content_type="text/event-stream", chunks=(_EVENT,), abort_after=0)
        return Reply(content_type="text/event-stream", chunks=(_EVENT,))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"vertex_ai/{_BACKEND}",
            api_base=f"{wire.url}{_MODEL_PATH}",
            api_key=None,
            vertex_project=_PROJECT,
            vertex_location=_LOCATION,
            vertex_credentials=_service_account_json(gateway.upstream_url.rstrip("/")),
        )
        original_retries: Final = object_value(gateway.get("/router/settings")["current_values"])["num_retries"]
        gateway.post("/config/update", {"router_settings": {"num_retries": 2}})
        try:
            with gateway.client.stream(
                "POST",
                f"/v1beta/models/{model}:streamGenerateContent?alt=sse",
                json={"contents": [{"role": "user", "parts": [{"text": "Reply with exactly: pong"}]}]},
                headers={"Authorization": f"Bearer {gateway.key}"},
                timeout=30,
            ) as response:
                assert response.status_code == 200, response.read()
                lines: Final = tuple(line for line in response.iter_lines() if line.startswith("data: "))
        finally:
            gateway.post("/config/update", {"router_settings": {"num_retries": original_retries}})
        assert lines == (f"data: {_EVENT_PAYLOAD}",), lines
        assert [(request.method, request.target) for request in wire.drain()] == [
            ("POST", _STREAM_TARGET),
            ("POST", _STREAM_TARGET),
        ]
