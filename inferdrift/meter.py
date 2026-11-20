# Copyright 2026 The InferDrift Authors
# SPDX-License-Identifier: Apache-2.0
"""Cost meter with hard dollar, token and wall-time caps.

Every request is *reserved* before it is sent: its worst-case cost is
``prompt_tokens_upper * price_in + max_new_tokens * price_out``.  A request whose
reservation would take spent + reserved past a cap is never sent; the run stops
with ``CapReached``.  After the response the reservation is settled at the usage
the provider reported (or, when the provider reports none, at an estimate,
flagged ``usage_source = 'estimated'``).  As long as the provider bills no more
than the prompt it was sent and the ``max_tokens`` it was given, the metered
total can therefore never exceed the cap.  A provider that reports more than the
reservation is recorded as an overrun and stops the run at once.

Wall time: ``check_wall`` raises once the elapsed time exceeds the cap, and
``remaining_wall`` bounds every request timeout and every local generation's
deadline.

Closed domains.  ``Caps`` and ``PriceCard`` accept only finite
values in their intervals (a cap is absent or > 0; a price is >= 0 and at most
``MAX_PRICE_USD_PER_MTOK``; a token cap is a positive integer, never a boolean)
and raise ``BadInputError`` otherwise.  The meter re-checks both when it is
built, and refuses (``MeterError``) a reservation whose token bounds are not
non-negative integers or whose cost is not finite, and a settlement whose token
counts are not non-negative integers in the exact range.  A refused settlement is
charged at its reserved worst case and stops the run, so NaN, infinity, negative
balances and string or boolean coercions can never satisfy a limit.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import asdict, dataclass

from . import domains
from .errors import BadInputError, CapReached, MeterError

MAX_PRICE_USD_PER_MTOK = 1e6
USAGE_SOURCES = ("provider", "local", "estimated", "estimated_worst_case", "invalid_provider_usage")


@dataclass(frozen=True)
class PriceCard:
    """USD per million input and output tokens: finite, >= 0, at most MAX_PRICE_USD_PER_MTOK."""

    usd_per_mtok_in: float = 0.0
    usd_per_mtok_out: float = 0.0

    def __post_init__(self) -> None:
        for name in ("usd_per_mtok_in", "usd_per_mtok_out"):
            v = domains.number(getattr(self, name), f"price {name}", ge=0.0, le=MAX_PRICE_USD_PER_MTOK)
            object.__setattr__(self, name, v)

    def cost(self, tokens_in: int, tokens_out: int) -> float:
        return (tokens_in * self.usd_per_mtok_in + tokens_out * self.usd_per_mtok_out) / 1e6

    @staticmethod
    def parse(text: str | None, name: str = "price") -> "PriceCard | None":
        """'IN,OUT' (USD per million tokens) -> PriceCard; None -> None (no price given, which
        is not the same as a price of zero).  Anything else, the empty string included, must
        be two plain decimal numbers."""
        if text is None:
            return None
        if not isinstance(text, str):
            raise BadInputError(name, "must be 'USD_PER_MTOK_IN,USD_PER_MTOK_OUT'", text)
        parts = text.split(",")
        if len(parts) != 2:
            raise BadInputError(name, "must be 'USD_PER_MTOK_IN,USD_PER_MTOK_OUT'", text)
        a = domains.parse_number(parts[0], f"{name} (input)", ge=0.0, le=MAX_PRICE_USD_PER_MTOK)
        b = domains.parse_number(parts[1], f"{name} (output)", ge=0.0, le=MAX_PRICE_USD_PER_MTOK)
        return PriceCard(a, b)


@dataclass
class Reservation:
    rid: int
    stack: str
    kind: str
    tokens_in_upper: int
    tokens_out_upper: int
    usd: float
    settled: bool = False


@dataclass(frozen=True)
class Caps:
    """Hard caps.  Each is absent (None: no cap) or inside its domain: dollars and seconds
    finite and > 0, tokens a positive integer in the exact range (never a boolean)."""

    max_usd: float | None = None
    max_tokens: int | None = None
    max_wall_s: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_usd", domains.optional_number(self.max_usd, "max_usd", gt=0.0))
        object.__setattr__(self, "max_tokens", domains.optional_integer(self.max_tokens, "max_tokens", ge=1))
        object.__setattr__(self, "max_wall_s", domains.optional_number(self.max_wall_s, "max_wall_s", gt=0.0))

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class _Totals:
    usd: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    requests: int = 0
    estimated_requests: int = 0
    invalid_usage_requests: int = 0


class CostMeter:
    def __init__(self, caps: Caps, prices: dict[str, PriceCard] | None = None,
                 clock=time.monotonic) -> None:
        if not isinstance(caps, Caps):
            raise BadInputError("caps", "must be a Caps", caps)
        # re-check the domains here too: a frozen dataclass can still be forced from outside
        self.caps = Caps(caps.max_usd, caps.max_tokens, caps.max_wall_s)
        self.prices: dict[str, PriceCard] = {}
        for stack, card in (prices or {}).items():
            if card is None:
                continue  # no price given for this stack
            if not isinstance(card, PriceCard):
                raise BadInputError(f"price of {stack}", "must be a PriceCard", card)
            self.prices[stack] = PriceCard(card.usd_per_mtok_in, card.usd_per_mtok_out)
        self._clock = clock
        self._t0 = clock()
        self._lock = threading.Lock()
        self._next = 0
        self._open: dict[int, Reservation] = {}
        self.totals: dict[str, _Totals] = {}
        self.ledger: list[dict] = []
        self.stopped_by: str | None = None
        self.overruns: list[dict] = []
        self.refusals: list[dict] = []

    # -- time --------------------------------------------------------------------------
    def elapsed(self) -> float:
        return self._clock() - self._t0

    def remaining_wall(self) -> float | None:
        if self.caps.max_wall_s is None:
            return None
        return self.caps.max_wall_s - self.elapsed()

    def check_wall(self) -> None:
        rem = self.remaining_wall()
        if rem is not None and rem <= 0:
            self._stop("wall")
            raise CapReached(f"wall-time cap of {self.caps.max_wall_s} s reached", cap="wall")

    # -- money and tokens -------------------------------------------------------------
    def price(self, stack: str) -> PriceCard:
        return self.prices.get(stack, PriceCard())

    def _spent_usd(self) -> float:
        return sum(t.usd for t in self.totals.values())

    def _spent_tokens(self) -> int:
        return sum(t.tokens_in + t.tokens_out for t in self.totals.values())

    def _reserved(self) -> tuple[float, int]:
        usd = sum(r.usd for r in self._open.values())
        tok = sum(r.tokens_in_upper + r.tokens_out_upper for r in self._open.values())
        return usd, tok

    def _stop(self, cap: str) -> None:
        if self.stopped_by is None:
            self.stopped_by = cap

    @staticmethod
    def _token_count(value) -> int | None:
        """A non-negative integer token count in the exact range, else None."""
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= domains.MAX_EXACT_INT:
            return None
        return int(value)

    def _refuse(self, what: str, detail: dict) -> MeterError:
        self.refusals.append({"what": what, **detail})
        self._stop("meter_refusal")
        return MeterError(f"cost meter refused {what}: {detail}")

    def reserve(self, stack: str, tokens_in_upper: int, tokens_out_upper: int, *, kind: str = "generate") -> Reservation:
        tin, tout = self._token_count(tokens_in_upper), self._token_count(tokens_out_upper)
        if tin is None or tout is None:
            with self._lock:
                raise self._refuse("a reservation with invalid token bounds",
                                   {"stack": stack, "tokens_in_upper": repr(tokens_in_upper)[:40],
                                    "tokens_out_upper": repr(tokens_out_upper)[:40]})
        self.check_wall()
        cost = self.price(stack).cost(tin, tout)
        with self._lock:
            if self.refusals:
                raise MeterError("cost meter stopped after a refused operand; run stopped")
            if not math.isfinite(cost) or cost < 0:
                raise self._refuse("a reservation with a non-finite or negative cost",
                                   {"stack": stack, "tokens_in_upper": tin, "tokens_out_upper": tout,
                                    "cost": repr(cost)})
            if self.overruns:
                self._stop("usd")
                raise CapReached("a provider billed more than its reservation; run stopped", cap="usd")
            r_usd, r_tok = self._reserved()
            if self.caps.max_usd is not None and self._spent_usd() + r_usd + cost > self.caps.max_usd + 1e-12:
                self._stop("usd")
                raise CapReached(
                    f"next request (worst case ${cost:.6f}) would pass the ${self.caps.max_usd} cap", cap="usd")
            need = tin + tout
            if self.caps.max_tokens is not None and self._spent_tokens() + r_tok + need > self.caps.max_tokens:
                self._stop("tokens")
                raise CapReached(f"next request ({need} tokens worst case) would pass the token cap", cap="tokens")
            self._next += 1
            res = Reservation(self._next, stack, kind, tin, tout, cost)
            self._open[res.rid] = res
            return res

    def settle(self, res: Reservation, tokens_in: int, tokens_out: int, *, usage_source: str) -> float:
        with self._lock:
            if res.settled:
                raise RuntimeError("reservation already settled")
            tin, tout = self._token_count(tokens_in), self._token_count(tokens_out)
            cost = self.price(res.stack).cost(tin, tout) if tin is not None and tout is not None else None
            problem = None
            if tin is None or tout is None:
                problem = "a settlement with invalid token counts"
            elif cost is None or not math.isfinite(cost) or cost < 0:
                problem = "a settlement with a non-finite or negative cost"
            elif usage_source not in USAGE_SOURCES:
                problem = "a settlement with an unknown usage source"
            if problem is not None:
                # charge the reserved worst case: the request may have been billed
                detail = {"stack": res.stack, "rid": res.rid, "tokens_in": repr(tokens_in)[:40],
                          "tokens_out": repr(tokens_out)[:40], "usage_source": repr(usage_source)[:40]}
                self._book(res, res.tokens_in_upper, res.tokens_out_upper, res.usd, "refused_operand_worst_case")
                raise self._refuse(problem, detail)
            self._book(res, tin, tout, cost, usage_source)
            if cost > res.usd + 1e-12 or tin + tout > res.tokens_in_upper + res.tokens_out_upper:
                self.overruns.append(self.ledger[-1])
                self._stop("usd")
            return cost

    def _book(self, res: Reservation, tin: int, tout: int, cost: float, usage_source: str) -> None:
        res.settled = True
        self._open.pop(res.rid, None)
        t = self.totals.setdefault(res.stack, _Totals())
        t.usd += cost
        t.tokens_in += tin
        t.tokens_out += tout
        t.requests += 1
        if usage_source not in ("provider", "local"):
            t.estimated_requests += 1
        if usage_source == "invalid_provider_usage":
            t.invalid_usage_requests += 1
        self.ledger.append({
            "rid": res.rid, "stack": res.stack, "kind": res.kind, "tokens_in": tin,
            "tokens_out": tout, "usd": cost, "reserved_usd": res.usd,
            "usage_source": usage_source, "t": round(self.elapsed(), 3),
        })

    def cancel(self, res: Reservation) -> None:
        with self._lock:
            if not res.settled:
                res.settled = True
                self._open.pop(res.rid, None)

    def snapshot(self) -> dict:
        with self._lock:
            per = {k: asdict(v) for k, v in self.totals.items()}
            return {
                "usd_total": self._spent_usd(),
                "tokens_total": self._spent_tokens(),
                "requests_total": sum(v.requests for v in self.totals.values()),
                "per_stack": per,
                "prices_usd_per_mtok": {k: asdict(v) for k, v in self.prices.items()},
                "caps": self.caps.to_dict(),
                "wall_s": round(self.elapsed(), 3),
                "stopped_by": self.stopped_by,
                "overruns": list(self.overruns),
                "refusals": list(self.refusals),
                "open_reservations": len(self._open),
            }
