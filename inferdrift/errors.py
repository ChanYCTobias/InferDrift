# Copyright 2026 The InferDrift Authors
# SPDX-License-Identifier: Apache-2.0
"""Exception hierarchy.  Every message that leaves the package passes the redactor."""

from __future__ import annotations


class FidelityError(Exception):
    """Base class.  ``code`` is a stable machine-readable reason code."""

    code = "FIDELITY_ERROR"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        from .redact import scrub_text

        super().__init__(scrub_text(message))
        if code is not None:
            self.code = code


class RefusedError(FidelityError):
    """A job was refused before it started (allowance, unsafe repository, bad spec)."""

    code = "REFUSED"


class UnsafeRepositoryError(RefusedError):
    code = "UNSAFE_REPOSITORY"


class AllowanceExceededError(RefusedError):
    code = "TRANSIENT_ALLOWANCE_EXCEEDED"


class CapReached(FidelityError):
    """A hard cap (dollars, tokens or wall time) stops the run."""

    code = "CAP_REACHED"

    def __init__(self, message: str, *, cap: str) -> None:
        super().__init__(message, code=f"CAP_{cap.upper()}")
        self.cap = cap


class AdapterError(FidelityError):
    code = "ADAPTER_ERROR"


class EndpointError(AdapterError):
    code = "ENDPOINT_ERROR"

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class UnsupportedError(AdapterError):
    code = "UNSUPPORTED"


class SpecError(RefusedError):
    code = "BAD_SPEC"


_NOT_SHOWN = object()


class BadInputError(SpecError):
    """A value outside its declared domain: NaN, infinity, a negative or zero
    cap, a boolean where a count is expected, an out-of-range probability, a malformed JSON
    input.  Raised before any request is sent or any output is written; the CLI exits 3.

    ``field`` names the input, ``problem`` says which part of its domain it violates."""

    code = "BAD_INPUT"

    def __init__(self, field: str, problem: str, value=_NOT_SHOWN) -> None:
        self.field = field
        self.problem = problem
        shown = ""
        if value is not _NOT_SHOWN:
            text = repr(value)
            shown = f" (got {text[:80]}{'...' if len(text) > 80 else ''})"
        super().__init__(f"{field}: {problem}{shown}")


class MeterError(FidelityError):
    """The cost meter refused an operand it cannot account for (non-finite cost, negative or
    non-integer token count).  The run stops; a refused settlement is charged at its
    reserved worst case so the metered total never falls below what may have been spent."""

    code = "METER_REFUSED"


class PartialCapReached(CapReached):
    """A cap stopped a batch part-way; ``done`` holds the results completed before it."""

    def __init__(self, done: list, cause: CapReached) -> None:
        super().__init__(str(cause), cap=getattr(cause, "cap", "usd"))
        self.done = done
