"""What a client is told about a provider failure, and what the usage log records.

This module turns a read upstream exception into what the caller is answered
with: the status, the detail, the headers and the stream error event. The split
in the detail text is the rule it exists to hold. A rejection of the caller's own request carries the
provider's own message, redacted, because the provider is the only party that
knows what it objected to. A failure of the deployment's own (a rejected
credential, an exhausted provider account, a 5xx) carries a fixed wording, since
those carry no remedy the caller could apply and are where a raw message is most
likely to name the operator's credentials or topology.
"""

import re
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Any, NamedTuple, cast

from any_llm.exceptions import AnyLLMError, ContextLengthExceededError, InvalidRequestError, UnsupportedParameterError
from any_llm.types.completion import CompletionParams
from any_llm.types.messages import MessagesParams
from any_llm.types.responses import ResponsesParams
from fastapi import HTTPException, status

from gateway.core.error_codes import (
    CONTEXT_LENGTH_EXCEEDED,
    PROVIDER_CREDENTIAL_REJECTED,
    PROVIDER_ERROR,
    PROVIDER_STREAM_INTERRUPTED,
    REQUEST_TOO_LARGE,
    STRUCTURED_OUTPUT_FAILED,
    UNSUPPORTED_CONTENT,
    UNSUPPORTED_PARAMETER,
    UPSTREAM_RATE_LIMITED,
    error_code_of,
    error_headers,
)
from gateway.core.error_details import ProviderDetail
from gateway.core.provider_params import SENSITIVE_PARAM_FIELDS
from gateway.exceptions.guardrails_exceptions import AnswerRefused
from gateway.services._tool_loop import MaxToolIterationsExceeded
from gateway.services.inference._upstream_errors import (
    EMPTY_COMPLETION_DETAIL,
    is_empty_completion,
    is_provider_billing_error,
    is_provider_credential_rejection,
    is_upstream_connect_failure,
    is_upstream_request_fault,
    is_upstream_stream_interrupted,
    upstream_body_messages,
    upstream_error_message,
    upstream_error_tag,
    upstream_exception_chain,
    upstream_exception_shape,
    upstream_retry_after,
)
from gateway.services.provider_kwargs import missing_credential
from gateway.services.upstream_redaction import redacted_upstream_detail
from gateway.streaming import IncompleteStreamError


class ProviderErrorMapping(NamedTuple):
    """A safe, client-facing (status, detail) for a classified provider failure."""

    status_code: int
    detail: str


_UNEXPECTED_KWARG = re.compile(r"unexpected keyword argument '([^']+)'")

# The params a caller can actually put in a request body, across the three
# formats that share this classifier. Derived from any-llm's ``*Params`` for the
# same reason the request schemas are: a param any-llm grows is picked up here
# without an edit, and nothing else can pass the gate.
#
# Derived the same way means derived *minus* ``SENSITIVE_PARAM_FIELDS``, not
# merely from the same models. ``derive_request_base`` skips those names because
# a future any-llm version could add a credential or provider-selection field to
# a typed ``*Params``, and exposing one as caller-settable would let a request
# override the operator's value. A name the schema refuses to accept is by
# definition a name no caller sent, so it must not pass the caller-fault gate
# either: the two definitions of "settable by a caller" are one definition, and
# spelling it twice is how they drift.
_FORWARDED_PARAMS: frozenset[str] = frozenset(
    (set(CompletionParams.model_fields) | set(MessagesParams.model_fields) | set(ResponsesParams.model_fields))
    - SENSITIVE_PARAM_FIELDS
)

# Lowercase substrings of a rejection that says the model takes no image input.
# Each is a provider's wording seen in a real response; add one only on the same footing.
# Narrow for the same reason as the billing probes: each names images and says
# they are refused, so a malformed request of any other kind cannot match.
_IMAGE_REJECTION_PROBES: tuple[str, ...] = (
    "image content is not supported",  # cohere
    "unsupported chatmessagecontent type: image_url",  # fireworks
)

# Groq's code for a generation that did not match the requested schema.
_STRUCTURED_OUTPUT_FAILURE_TAGS = frozenset({"json_validate_failed"})

# Groq's message points at ``failed_generation``, a body field Otari does not relay.
_FAILED_GENERATION_POINTER = re.compile(r"\s*See 'failed_generation'[^.]*\.?")


def _error_response(exc: BaseException) -> Mapping[str, object]:
    """The ``Error`` part of an AWS SDK error response, or an empty mapping."""
    response = getattr(exc, "response", None)
    error = response.get("Error") if isinstance(response, Mapping) else None
    return error if isinstance(error, Mapping) else {}


def _carried_message(exc: BaseException) -> str | None:
    """The message the provider put on one exception, without text the SDK adds, or ``None``."""
    message = getattr(exc, "message", None)
    if not isinstance(message, str):
        message = _error_response(exc).get("Message")
    return message if isinstance(message, str) else None


def _upstream_message_detail(exc: BaseException, fallback: str) -> str:
    """The detail for a failure the caller can act on.

    Returns the provider's own message, redacted and length-capped, because the
    provider is the only party that knows what it objected to.

    An error body an SDK parsed onto the exception is read first, and when one
    is there nothing else is: the message inside it is what the provider wrote,
    while the same exception's ``message`` and string form wrap that body in a
    status prefix and a Python dict repr (OpenAI, Anthropic) or prepend the
    response headers (Cohere).

    Otherwise a message the provider put on an exception wins over the joined chain.
    An SDK's string form can repeat that message, add its own labels, or append the response body.
    A wrapper's message is its original's string form, so a message deeper in the chain wins over it.

    Falls back to ``fallback`` when what is left says nothing. Some SDKs
    stringify a failure as bare punctuation or the status code itself, and
    "404" is not an explanation the caller did not already have from the
    status. Requiring one letter is a low bar deliberately: it rejects the
    empty cases without second-guessing a provider that wrote a real sentence.
    A message made entirely of redaction placeholders is empty for this
    purpose, too.
    """
    body_messages = upstream_body_messages(exc)
    if body_messages is not None:
        for body_message in body_messages:
            if detail := redacted_upstream_detail(body_message, ""):
                return detail
        return fallback
    wrapper_message: str | None = None
    for candidate in upstream_exception_chain(exc):
        message = _carried_message(candidate)
        if message is None:
            continue
        if getattr(candidate, "original_exception", None) is not None:
            wrapper_message = wrapper_message or message
        elif detail := redacted_upstream_detail(message, ""):
            return detail
    if wrapper_message is not None and (detail := redacted_upstream_detail(wrapper_message, "")):
        return detail
    return redacted_upstream_detail(upstream_error_message(exc), fallback)


def _is_unsupported_feature_error(exc: BaseException) -> bool:
    """True when any-llm refused a request feature its backend cannot express.

    Newer any-llm releases use ``UnsupportedParameterError`` for typed provider
    capability checks, while older feature checks still raise
    ``NotImplementedError``. Unwrap ``original_exception`` as well so either
    signal survives the unified-exception wrapper.
    """
    unsupported_types = (NotImplementedError, UnsupportedParameterError)
    return any(isinstance(candidate, unsupported_types) for candidate in upstream_exception_chain(exc))


def _unsupported_feature_detail(exc: BaseException) -> str:
    """Return one actionable reason for a locally rejected request feature."""
    for candidate in upstream_exception_chain(exc):
        if isinstance(candidate, UnsupportedParameterError):
            # AnyLLMError.__str__ prefixes the provider to ``message``. Feeding
            # both through upstream_error_message would repeat the same reason.
            return redacted_upstream_detail(candidate.message, ProviderDetail.BAD_REQUEST)
    return _upstream_message_detail(exc, ProviderDetail.BAD_REQUEST)


def _rejected_param(exc: BaseException, request_params: Collection[str] = _FORWARDED_PARAMS) -> str | None:
    """The request param a provider SDK refused as an unknown keyword, if any.

    any-llm hands a provider's SDK the params its ``*Params`` model declares, so
    a param that model carries but the SDK's method does not take arrives as
    ``TypeError: ...create() got an unexpected keyword argument 'x'``. That is
    every OpenAI-only chat param against a provider that never grew one
    (``seed``, ``n``, the penalties, the logprobs family, against Anthropic).

    It carries no HTTP status, so it would otherwise fall through to the generic
    502 and report a permanent, caller-fixable mismatch between a model and a
    param as an upstream outage. Only the param name is returned: the SDK's
    method name is an internal detail and never reaches the client.

    Two gates keep that from blaming a caller for someone else's fault, because
    Python raises this same wording for any bad keyword and this classifier is
    reached from ``except Exception`` arms that wrap more than the provider call:

    * The name has to be one a request body can carry: ``request_params``, which
      defaults to the completion formats' :data:`_FORWARDED_PARAMS`.
      An operator's ``client_args`` typo (``timeoutt``) and a gateway-internal
      signature drift after an SDK bump (a tool backend's ``image``) are neither,
      so they keep the generic 502 and stay on the error-rate panel as the
      upstream-or-gateway failures they are.
    * The failing callable must not be a constructor. ``client_args`` is
      operator-owned and reaches a provider *client's* ``__init__``, so a key
      there that happens to collide with a real param name (``client_args={"seed":
      1}``) would otherwise pass the first gate and read as the caller's ``seed``.

    Both gates key on the same interpreter wording the match already depends on,
    so neither adds a new assumption about how CPython phrases the error.
    """
    for candidate in upstream_exception_chain(exc):
        if not isinstance(candidate, TypeError):
            continue
        message = str(candidate)
        if "__init__" in message:
            continue
        match = _UNEXPECTED_KWARG.search(message)
        if match and match.group(1) in request_params:
            return match.group(1)
    return None


def _provider_rejected_param_detail(param: str) -> str:
    """Detail for a request param the provider serving this model cannot express."""
    return f"The provider serving this model does not accept the '{param}' parameter"


def _is_image_input_rejection(exc: BaseException) -> bool:
    """Whether the provider refused the request because the model does not accept images.

    Only a request fault can say that: a 400 or 422, or a rejection the provider
    raised before assigning a status.
    """
    _kind, status_code = upstream_exception_shape(exc)
    if status_code not in (None, 400, 422):
        return False
    message = upstream_error_message(exc).lower()
    return any(probe in message for probe in _IMAGE_REJECTION_PROBES)


def _unsupported_image_detail(model: str | None = None) -> str:
    """Detail for a request carrying an image to a model that does not accept images."""
    subject = f"The model '{model}'" if model else "The model serving this request"
    return (
        f"{subject} does not accept image content. "
        "Remove the image parts, or send the request to a model that accepts images."
    )


def _is_account_quota_error(exc: BaseException) -> bool:
    """Whether the provider refused because the operator's account is over a service quota."""
    return any(
        _error_response(current).get("Code") == "ServiceQuotaExceededException"
        for current in upstream_exception_chain(exc)
    )


def _is_context_length_error(exc: BaseException) -> bool:
    """Whether any-llm classified the failure as a prompt too long for the model."""
    return any(isinstance(current, ContextLengthExceededError) for current in upstream_exception_chain(exc))


def _is_structured_output_failure(exc: BaseException) -> bool:
    """Whether the provider's own code says the model's output did not match the requested schema."""
    return upstream_error_tag(exc) in _STRUCTURED_OUTPUT_FAILURE_TAGS


def _structured_output_failed_detail(exc: BaseException) -> str:
    """The provider's sentence, without its pointer to a field the caller never receives."""
    detail = _upstream_message_detail(exc, ProviderDetail.BAD_REQUEST)
    return _FAILED_GENERATION_POINTER.sub("", detail).strip() or ProviderDetail.BAD_REQUEST


def _is_upstream_failure(exc: BaseException) -> bool:
    """Whether ``exc`` came from the provider rather than from the gateway."""
    if isinstance(exc, IncompleteStreamError) or upstream_exception_shape(exc) != (None, None):
        return True
    return any(isinstance(current, AnyLLMError) for current in upstream_exception_chain(exc))


def _provider_failure_cause(exc: BaseException) -> str | None:
    if isinstance(exc, IncompleteStreamError):
        return ProviderDetail.STREAM_INTERRUPTED
    kind, status_code = upstream_exception_shape(exc)
    if status_code is not None:
        return f"The provider failed the request with HTTP {status_code}"
    if kind == "conn_err":
        if is_upstream_connect_failure(exc):
            return "The gateway could not connect to the provider"
        return "The provider closed the connection before answering"
    if kind == "timeout":
        return "The provider did not answer in time"
    return None


def _unsupported_parameter_name(exc: BaseException) -> str | None:
    """The parameter any-llm said the provider cannot express, or ``None`` when it named none."""
    for candidate in upstream_exception_chain(exc):
        if isinstance(candidate, UnsupportedParameterError):
            return candidate.parameter_name
    return None


def _classify_provider_error(
    exc: BaseException, *, request_params: Collection[str] = _FORWARDED_PARAMS, model: str | None = None
) -> ProviderErrorMapping | None:
    """Map an upstream provider exception to a safe, specific (status, detail).

    Returns ``None`` when the failure carries no signal we can safely act on, so
    the caller falls back to its existing generic provider-error response. The
    mapping is intentionally conservative, classifying only the cases a caller
    can act on and leaving everything else (including provider 5xx and
    connection errors) to the generic 502.

    Detail text splits on whether the caller can act on the failure. A rejection
    of the caller's request (400/422/404) and a rate limit (429) pass the
    provider's own message through, redacted and length-capped, because it is
    the only description of what was actually wrong. When the failure is the
    gateway's own (a rejected credential, an exhausted provider account, a 5xx),
    the detail stays a fixed string: those carry no remedy the caller could
    apply, and are where a raw message is most likely to name the operator's
    credentials or topology.

    Timeout detection (including the OpenAI/Anthropic SDKs' own
    ``APITimeoutError``, and a duck-typed fallback for other provider SDKs)
    comes from :func:`upstream_exception_shape`, so every reader of a failure's
    shape agrees on it.

    ``request_params`` is the set of names a caller could have sent, which is
    what lets an SDK's unknown-keyword ``TypeError`` read as the caller's 400
    (see :func:`_rejected_param`). ``model``, the name the caller knows the
    model by, is named in a detail that is about the model.
    """
    missing = missing_credential(exc)
    if missing is not None:
        return ProviderErrorMapping(missing.status_code, missing.detail)
    if is_empty_completion(exc):
        return ProviderErrorMapping(status.HTTP_502_BAD_GATEWAY, EMPTY_COMPLETION_DETAIL)
    kind, status_code = upstream_exception_shape(exc)
    if kind == "timeout":
        return ProviderErrorMapping(status.HTTP_504_GATEWAY_TIMEOUT, ProviderDetail.TIMEOUT)
    # any-llm raises UnsupportedParameterError for typed provider-capability
    # checks and still uses NotImplementedError for older feature checks such as
    # context_management/betas. Neither carries an HTTP status, so either
    # would otherwise fall through to a generic 502/500 and tell the caller a
    # guaranteed-permanent failure was transient. The exception type is the
    # whole signal here, and its message names the unsupported feature.
    if _is_unsupported_feature_error(exc):
        return ProviderErrorMapping(status.HTTP_400_BAD_REQUEST, _unsupported_feature_detail(exc))
    # The same shape one layer down: a param any-llm forwards that the resolved
    # provider's SDK has no parameter for.
    if (param := _rejected_param(exc, request_params)) is not None:
        return ProviderErrorMapping(status.HTTP_400_BAD_REQUEST, _provider_rejected_param_detail(param))
    # Each provider words this its own way, so it gets one wording and one code.
    if _is_image_input_rejection(exc):
        return ProviderErrorMapping(status.HTTP_400_BAD_REQUEST, _unsupported_image_detail(model))
    if status_code is None:
        # A status-less InvalidRequestError means the provider (gemini, bedrock,
        # anthropic) rejected the request shape before assigning an HTTP status.
        # Mirror the UnsupportedParameterError / NotImplementedError branches: the
        # exception type is the whole signal, and the provider's own message names
        # what was wrong. Gated on the failure's own status rather than each
        # link's, so a chain that does carry one (401, 429, 5xx) keeps its fixed
        # detail instead of echoing the upstream text as a client 400.
        if any(isinstance(candidate, InvalidRequestError) for candidate in upstream_exception_chain(exc)):
            return ProviderErrorMapping(
                status.HTTP_400_BAD_REQUEST, _upstream_message_detail(exc, ProviderDetail.BAD_REQUEST)
            )
        return None
    # Account billing exhaustion, which several providers report as a 400/422
    # rather than the 402 the condition deserves (and DeepSeek does report as a
    # 402). This is a gateway-side account fault rather than anything wrong with
    # the caller's request, so it surfaces as a 502 and never as a client-facing
    # 400: telling the caller to "check the model name and parameters" for an
    # empty wallet sends operators debugging the wrong thing. Checked ahead of the
    # status branches so it wins over both the generic bad-request detail and the
    # 402 fall-through to a bare 502.
    if is_provider_billing_error(exc):
        return ProviderErrorMapping(status.HTTP_502_BAD_GATEWAY, ProviderDetail.BILLING)
    # An account quota is the operator's limit, like billing, even where the provider answers 400.
    if _is_account_quota_error(exc):
        return ProviderErrorMapping(status.HTTP_502_BAD_GATEWAY, ProviderDetail.ACCOUNT_QUOTA)
    if status_code == 400 and _is_structured_output_failure(exc):
        return ProviderErrorMapping(status.HTTP_400_BAD_REQUEST, _structured_output_failed_detail(exc))
    # OpenAI answers some request faults with a 403 (logprobs on a model without
    # them), which is the caller's to fix and not the gateway's key.
    if status_code in (400, 422) or is_upstream_request_fault(exc):
        return ProviderErrorMapping(
            status.HTTP_400_BAD_REQUEST, _upstream_message_detail(exc, ProviderDetail.BAD_REQUEST)
        )
    if status_code == 404:
        return ProviderErrorMapping(
            status.HTTP_404_NOT_FOUND, _upstream_message_detail(exc, ProviderDetail.MODEL_NOT_FOUND)
        )
    # A provider rejecting the gateway's credentials is a gateway-config fault,
    # not the caller's, so never a client-facing 401/403. Nor a 5xx: no retry
    # makes the key valid, so it is refused like a missing credential is.
    if is_provider_credential_rejection(exc):
        return ProviderErrorMapping(status.HTTP_424_FAILED_DEPENDENCY, ProviderDetail.CREDENTIALS)
    # A 429 is not the caller's request to fix but is theirs to act on, and only
    # the provider's text names the exhausted quota and the retry window. Still
    # drops the upstream Retry-After: the (status, detail) pair cannot carry it,
    # and on a streaming failure the headers are already flushed.
    if status_code == 429:
        return ProviderErrorMapping(
            status.HTTP_429_TOO_MANY_REQUESTS, _upstream_message_detail(exc, ProviderDetail.RATE_LIMITED)
        )
    # A 413 refuses the request for its size, so it fails the same way on every
    # retry and the provider's sentence (the limit, the size requested) is what
    # tells the caller how far to shrink it.
    if status_code == 413:
        return ProviderErrorMapping(
            status.HTTP_413_CONTENT_TOO_LARGE, _upstream_message_detail(exc, ProviderDetail.REQUEST_TOO_LARGE)
        )
    return None


def _provider_error_headers(exc: BaseException, status_code: int) -> dict[str, str] | None:
    """Response headers for a classified provider failure, or ``None``.

    On a 429, ``Otari-Error-Code: upstream_rate_limited`` (so a caller can tell
    the provider's limit from the gateway's own) and the upstream ``Retry-After``,
    the one piece of a provider's rate-limit response that its message body cannot
    always carry. Restricted to the 429: on the statuses that surface as a
    fixed-detail 502 the header would describe the gateway's own upstream
    account, which is not the caller's to read. A 424 names which of the
    gateway's credentials failed it, missing or rejected, and nothing more. A
    completion with no choices is the one 502 with a code, ``provider_error``.
    """
    if status_code == status.HTTP_400_BAD_REQUEST and _is_context_length_error(exc):
        return error_headers(CONTEXT_LENGTH_EXCEEDED)
    if status_code == status.HTTP_400_BAD_REQUEST and _is_image_input_rejection(exc):
        return error_headers(UNSUPPORTED_CONTENT)
    if status_code == status.HTTP_400_BAD_REQUEST and _is_structured_output_failure(exc):
        return error_headers(STRUCTURED_OUTPUT_FAILED)
    if status_code == status.HTTP_400_BAD_REQUEST and _unsupported_parameter_name(exc) is not None:
        return error_headers(UNSUPPORTED_PARAMETER)
    if status_code == status.HTTP_413_CONTENT_TOO_LARGE and upstream_exception_shape(exc)[1] == 413:
        return error_headers(REQUEST_TOO_LARGE)
    missing = missing_credential(exc)
    if missing is not None and status_code == missing.status_code:
        return error_headers(missing.code)
    if status_code == status.HTTP_424_FAILED_DEPENDENCY and is_provider_credential_rejection(exc):
        return error_headers(PROVIDER_CREDENTIAL_REJECTED)
    if status_code == status.HTTP_502_BAD_GATEWAY and is_empty_completion(exc):
        return error_headers(PROVIDER_ERROR)
    if status_code != status.HTTP_429_TOO_MANY_REQUESTS:
        return None
    headers = error_headers(UPSTREAM_RATE_LIMITED)
    retry_after = upstream_retry_after(exc)
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return headers


def _refusal_code(
    exc: BaseException, *, request_params: Collection[str] = _FORWARDED_PARAMS, model: str | None = None
) -> str | None:
    """The ``Otari-Error-Code`` an exception ending a request stands for, or None.

    Read by the stream error events, which go out after the headers that would
    otherwise carry it.
    """
    if isinstance(exc, HTTPException):
        return error_code_of(exc.headers)
    mapping = _classify_provider_error(exc, request_params=request_params, model=model)
    if mapping is None:
        return None
    return error_code_of(_provider_error_headers(exc, mapping.status_code))


def _stream_error_code(
    exc: BaseException, *, request_params: Collection[str] = _FORWARDED_PARAMS, model: str | None = None
) -> str | None:
    """The ``error.code`` of the event that ends a stream on ``exc``, or None.

    The refusal's own code where it has one, otherwise ``provider_error`` for any
    failure the provider caused, so a client can tell a provider failure from the
    gateway's own without reading the message. A failure
    :func:`_classify_provider_error` recognizes counts as the provider's even
    when it carries no status, such as a parameter the provider's SDK refused.
    """
    code = _refusal_code(exc, request_params=request_params, model=model)
    if code is not None or isinstance(exc, HTTPException):
        return code
    if is_upstream_stream_interrupted(exc):
        return PROVIDER_STREAM_INTERRUPTED
    if (
        _is_upstream_failure(exc)
        or _classify_provider_error(exc, request_params=request_params, model=model) is not None
    ):
        return PROVIDER_ERROR
    return None


@dataclass(frozen=True, slots=True)
class UpstreamFailure:
    """One provider failure, as the caller and the usage log see it.

    ``request_params`` is the set of names the caller could have sent, which is
    what lets an SDK's unknown-keyword ``TypeError`` read as the caller's 400.
    It defaults to the completion formats' own set, and a request set of another
    shape is named instead. ``model`` is the name the caller knows the model by,
    named in a detail that is about the model.

    The members are ordered by what reads them: the HTTP response, then the
    usage row, then the stream event.

    Every member that classifies the exception reads both fields, so the HTTP
    error, the usage row and the stream event cannot describe one failure
    differently.
    """

    exc: BaseException
    request_params: Collection[str] = _FORWARDED_PARAMS
    model: str | None = None

    @property
    def mapping(self) -> ProviderErrorMapping | None:
        """The status and detail this failure maps to, or None when it carries no signal to act on."""
        return _classify_provider_error(self.exc, request_params=self.request_params, model=self.model)

    @property
    def headers(self) -> dict[str, str] | None:
        """The response headers that go with :attr:`mapping`, or None where it needs none.

        Every header here is one only the status it accompanies may carry, so the
        status is :attr:`mapping`'s own rather than a caller's to choose.
        """
        mapping = self.mapping
        if mapping is None:
            return None
        return _provider_error_headers(self.exc, mapping.status_code)

    def unclassified_detail(self, fallback: str = ProviderDetail.ERROR) -> str:
        """What a failure :attr:`mapping` leaves unclassified says went wrong upstream.

        It names the provider's error status, a connection it closed or never
        opened, or a stream it cut short, and nothing else: no host, message or
        body, since those are the deployment's own topology. ``fallback`` where
        the failure names none of that, such as a fault in Otari itself.
        """
        return _provider_failure_cause(self.exc) or fallback

    @property
    def logged_status(self) -> int:
        """The HTTP status a usage row records for this failure.

        Prefers the status the provider actually returned, which is deliberately
        not always the status the caller saw: an upstream 401/403 surfaces to the
        caller as a 424 (a provider rejecting the deployment's credentials is a
        configuration fault, not the caller's key), but the log keeps the 401 so
        "how much of my error rate is my own misconfiguration" stays answerable.
        Where the provider returned no status at all (timeout, unreachable),
        records Otari's own classification instead, so an error row still carries
        a code to group on.
        """
        if isinstance(self.exc, MaxToolIterationsExceeded):
            # Otari's own limit carries no upstream status, so it would otherwise
            # fall through to the generic 502 and read as a provider outage.
            return status.HTTP_422_UNPROCESSABLE_CONTENT
        if isinstance(self.exc, AnswerRefused):
            return self.exc.status_code
        # Before the wrapper's own status: a wrapper can carry a 500 around a
        # credential any-llm never found, and the row records what the caller saw.
        missing = missing_credential(self.exc)
        if missing is not None:
            return missing.status_code
        _kind, status_code = upstream_exception_shape(self.exc)
        if status_code is not None:
            return status_code
        mapping = self.mapping
        return mapping.status_code if mapping is not None else status.HTTP_502_BAD_GATEWAY

    @property
    def logged_message(self) -> str:
        """The error text of this failure, as the upstream gave it.

        Otari's own wording where the failure is one it names, since an SDK's
        string form can redact to nothing (Cohere's leads with the response
        headers).

        NOTE: This is the raw upstream text. A caller that persists or returns it
        must redact it first, since it may name the deployment's own topology.
        """
        if _is_image_input_rejection(self.exc):
            return _unsupported_image_detail()
        return str(self.exc)

    @property
    def error_code(self) -> str | None:
        """The ``Otari-Error-Code`` this failure stands for, or None."""
        return _refusal_code(self.exc, request_params=self.request_params, model=self.model)

    @property
    def stream_code(self) -> str | None:
        """The ``error.code`` of the event that ends a stream on this failure, or None."""
        return _stream_error_code(self.exc, request_params=self.request_params, model=self.model)

    @property
    def stream_message(self) -> str | None:
        """The message of that event, or None for the format's fixed one.

        The detail the same failure gets as an HTTP error, where that names a
        cause, since that detail is already safe for the caller to read.
        """
        if isinstance(self.exc, AnswerRefused):
            return str(cast("dict[str, Any]", self.exc.detail)["message"])
        if isinstance(self.exc, HTTPException):
            return None
        mapping = self.mapping
        if mapping is not None:
            return mapping.detail
        if is_upstream_stream_interrupted(self.exc):
            return ProviderDetail.STREAM_INTERRUPTED
        return _provider_failure_cause(self.exc)

    @property
    def unsupported_parameter_name(self) -> str | None:
        """The parameter the provider cannot express, or None when it named none."""
        return _unsupported_parameter_name(self.exc)
