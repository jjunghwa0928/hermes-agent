"""Tests for agent.rate_limit_tracker — header parsing and formatting."""

import time
from datetime import datetime, timedelta, timezone

import pytest
from agent.rate_limit_tracker import (
    RateLimitBucket,
    format_rate_limit_compact,
    format_rate_limit_display,
    parse_rate_limit_headers,
)

# ── Sample headers from Nous inference API ──────────────────────────────

NOUS_HEADERS = {
    "x-ratelimit-limit-requests": "800",
    "x-ratelimit-limit-requests-1h": "33600",
    "x-ratelimit-limit-tokens": "8000000",
    "x-ratelimit-limit-tokens-1h": "336000000",
    "x-ratelimit-remaining-requests": "795",
    "x-ratelimit-remaining-requests-1h": "33590",
    "x-ratelimit-remaining-tokens": "7999500",
    "x-ratelimit-remaining-tokens-1h": "335999000",
    "x-ratelimit-reset-requests": "45.5",
    "x-ratelimit-reset-requests-1h": "3500.0",
    "x-ratelimit-reset-tokens": "42.3",
    "x-ratelimit-reset-tokens-1h": "3490.0",
}

class TestParseHeaders:
    def test_basic_parsing(self):
        state = parse_rate_limit_headers(NOUS_HEADERS, provider="nous")
        assert state is not None
        assert state.provider == "nous"
        assert state.has_data

        assert state.requests_min.limit == 800
        assert state.requests_min.remaining == 795
        assert state.requests_min.reset_seconds == 45.5

        assert state.requests_hour.limit == 33600
        assert state.requests_hour.remaining == 33590

        assert state.tokens_min.limit == 8000000
        assert state.tokens_min.remaining == 7999500

        assert state.tokens_hour.limit == 336000000
        assert state.tokens_hour.remaining == 335999000
        assert state.tokens_hour.reset_seconds == 3490.0

    def test_no_headers(self):
        state = parse_rate_limit_headers({})
        assert state is None

class TestBucket:

    def test_usage_pct(self):
        b = RateLimitBucket(limit=100, remaining=20, reset_seconds=30.0, captured_at=time.time())
        assert b.usage_pct == pytest.approx(80.0)

    def test_remaining_seconds_now(self):
        now = time.time()
        b = RateLimitBucket(limit=800, remaining=795, reset_seconds=60.0, captured_at=now - 10)
        # ~50 seconds should remain
        assert 49 <= b.remaining_seconds_now <= 51


# ── Anthropic unified (subscription) headers ────────────────────────────

# The 5-hour window is anchored to the session's first request, so its reset lands on an
# arbitrary instant — deliberately off every clock grid here (odd minutes AND odd seconds).
_5H_OFFSET = 5 * 3600 - 1543
_7D_OFFSET = 7 * 86400 - 907


def _unified_headers(skew: int = 0):
    """Unified headers as Anthropic sends them: ISO-8601 for one window, unix epoch for the
    other, mixed header casing, plus a family member that is not a window. ``skew`` shifts the
    5-hour anchor, standing in for a session that started at a different moment."""
    now = datetime.now(timezone.utc)
    five_h = now + timedelta(seconds=_5H_OFFSET + skew)
    seven_d = now + timedelta(seconds=_7D_OFFSET)
    return {
        "anthropic-ratelimit-unified-status": "allowed_warning",
        "anthropic-ratelimit-unified-5h-limit": "7000",
        "Anthropic-RateLimit-Unified-5h-Remaining": "1234",
        "anthropic-ratelimit-unified-5h-reset": five_h.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "anthropic-ratelimit-unified-5h-status": "allowed",
        "anthropic-ratelimit-unified-7d-limit": "90000",
        "anthropic-ratelimit-unified-7d-remaining": "40000",
        "anthropic-ratelimit-unified-7d-reset": str(int(seven_d.timestamp())),
        "anthropic-ratelimit-unified-fallback-percentage": "0.5",
    }


class TestUnifiedHeaders:
    def test_unified_only_response_is_captured_with_an_absolute_reset(self):
        """A subscription response carries no x-ratelimit-* at all; it must still parse, and the
        reset must survive as the exact instant sent rather than being floored into a bucket."""
        headers = _unified_headers()
        state = parse_rate_limit_headers(headers, provider="anthropic")
        assert state is not None and state.has_data

        five_h = state.unified["5h"]
        assert (five_h.limit, five_h.remaining) == (7000, 1234)
        assert five_h.status == "allowed"
        assert state.unified["overall"].status == "allowed_warning"

        # Stored as the exact instant the header named, not derived from a window width.
        sent = datetime.strptime(
            headers["anthropic-ratelimit-unified-5h-reset"], "%Y-%m-%dT%H:%M:%SZ"
        ).replace(tzinfo=timezone.utc).timestamp()
        assert five_h.reset_at == pytest.approx(sent, abs=1)
        assert five_h.remaining_seconds_now == pytest.approx(_5H_OFFSET, abs=5)

        # Unix-epoch resets parse to the same absolute scale as the ISO-8601 ones.
        assert state.unified["7d"].remaining_seconds_now == pytest.approx(_7D_OFFSET, abs=5)
        # ``-fallback-percentage`` has no window/field shape and is not mistaken for one.
        assert "fallback" not in state.unified

        # Parsed state reaches both /usage renderers.
        assert "5h" in format_rate_limit_compact(state)
        assert "allowed_warning" in format_rate_limit_display(state)

    def test_reset_instants_are_not_snapped_to_a_fixed_window_grid(self):
        """The 5-hour window is anchored to a session's first request, so two sessions whose
        resets differ by 137s must stay 137s apart — flooring onto a 4h/5h grid would collapse
        them onto the same instant."""
        skew = 137
        early = parse_rate_limit_headers(_unified_headers(), provider="anthropic")
        late = parse_rate_limit_headers(_unified_headers(skew=skew), provider="anthropic")
        assert late.unified["5h"].reset_at - early.unified["5h"].reset_at == pytest.approx(skew, abs=1)

    def test_classic_and_unified_families_coexist_without_changing_classic_semantics(self):
        """x-ratelimit-* resets stay seconds-from-capture; unified resets stay absolute."""
        classic_only = parse_rate_limit_headers(NOUS_HEADERS, provider="nous")
        assert classic_only.unified == {}
        assert classic_only.requests_min.reset_at == 0.0
        assert classic_only.requests_min.remaining_seconds_now == pytest.approx(45.5, abs=2)

        both = parse_rate_limit_headers({**NOUS_HEADERS, **_unified_headers()}, provider="anthropic")
        assert both.requests_min.limit == classic_only.requests_min.limit
        assert both.requests_min.reset_seconds == classic_only.requests_min.reset_seconds
        assert set(both.unified) == {"overall", "5h", "7d"}


# Subscription responses as received (two Pro/Max OAuth accounts, 2026-11-04 20:48:55Z): no
# ``-limit``/``-remaining`` at all — usage arrives only as ``-utilization``, a fraction — and
# overage switched off at org level reports ``rejected`` on the healthy account as well.
_PROBED_AT = 1793825335.0  # 2026-11-04 20:48:55Z

_SUBSCRIPTION_HEALTHY = {  # HTTP 200
    "anthropic-ratelimit-unified-status": "allowed",
    "anthropic-ratelimit-unified-reset": "1793841000",
    "anthropic-ratelimit-unified-5h-status": "allowed",
    "anthropic-ratelimit-unified-5h-utilization": "0.1",
    "anthropic-ratelimit-unified-5h-reset": "1793841000",
    "anthropic-ratelimit-unified-7d-status": "allowed",
    "anthropic-ratelimit-unified-7d-utilization": "0.51",
    "anthropic-ratelimit-unified-7d-reset": "1794297600",
    "anthropic-ratelimit-unified-overage-status": "rejected",
    "anthropic-ratelimit-unified-overage-disabled-reason": "org_level_disabled",
    "anthropic-ratelimit-unified-representative-claim": "five_hour",
    "anthropic-ratelimit-unified-fallback-percentage": "0.5",
}

_SUBSCRIPTION_WEEK_SPENT = {  # HTTP 429
    "anthropic-ratelimit-unified-status": "rejected",
    "anthropic-ratelimit-unified-reset": "1794164400",
    "anthropic-ratelimit-unified-5h-status": "allowed",
    "anthropic-ratelimit-unified-5h-utilization": "0.0",
    "anthropic-ratelimit-unified-5h-reset": "1793842800",
    "anthropic-ratelimit-unified-7d-status": "rejected",
    "anthropic-ratelimit-unified-7d-utilization": "1.0",
    "anthropic-ratelimit-unified-7d-surpassed-threshold": "1.0",
    "anthropic-ratelimit-unified-7d-reset": "1794164400",
    "anthropic-ratelimit-unified-overage-status": "rejected",
    "anthropic-ratelimit-unified-overage-disabled-reason": "org_level_disabled",
    "anthropic-ratelimit-unified-representative-claim": "seven_day",
    "anthropic-ratelimit-unified-fallback-percentage": "0.5",
    "anthropic-ratelimit-unified-upgrade-paths": "upgrade_plan,overage",
}


def _subscription_state(headers, **overrides):
    state = parse_rate_limit_headers({**headers, **overrides}, provider="anthropic")
    assert state is not None
    return state


@pytest.fixture
def at_probe_time(monkeypatch):
    monkeypatch.setattr("agent.rate_limit_tracker.time.time", lambda: _PROBED_AT)


class TestUnifiedUtilization:
    def test_utilization_only_windows_report_usage(self, at_probe_time):
        """With no counters sent, the window's usage is the utilization fraction Anthropic sent."""
        state = _subscription_state(_SUBSCRIPTION_HEALTHY)
        week = state.unified["7d"]
        assert (week.limit, week.remaining) == (0, 0)
        sent = float(_SUBSCRIPTION_HEALTHY["anthropic-ratelimit-unified-7d-utilization"])
        assert week.usage_pct == pytest.approx(sent * 100)
        assert week.remaining_seconds_now == pytest.approx(
            float(_SUBSCRIPTION_HEALTHY["anthropic-ratelimit-unified-7d-reset"]) - _PROBED_AT
        )
        # Header family members that are not windows stay out (claim, threshold, fallback).
        assert set(state.unified) == {"overall", "5h", "7d", "overage"}

        display = format_rate_limit_display(state)
        week_line = next(line for line in display.splitlines() if "Weekly/7d" in line)
        assert f"{week.usage_pct:.1f}%" in week_line
        assert "no limit reported" not in week_line
        assert f"Weekly/7d: {week.usage_pct:.0f}% used" in format_rate_limit_compact(state)

    def test_zero_and_over_one_utilization_are_readings_not_absence(self, at_probe_time):
        spent = _subscription_state(_SUBSCRIPTION_WEEK_SPENT)
        session_line = next(
            line for line in format_rate_limit_display(spent).splitlines() if "Session/5h" in line
        )
        assert "0.0%" in session_line
        # A spent window can read past 1.0; the percentage follows it rather than capping.
        over = _subscription_state(
            _SUBSCRIPTION_WEEK_SPENT, **{"anthropic-ratelimit-unified-5h-utilization": "1.01"}
        )
        assert over.unified["5h"].usage_pct == pytest.approx(101.0)
        # No utilization header at all is still "not reported".
        assert spent.unified["overall"].utilization is None

    def test_disabled_overage_is_not_shown_as_a_rejection(self, at_probe_time):
        """``overage-status: rejected`` with a disabled reason is a switched-off feature: the
        healthy account must not show a rejection, while a real rejection (spent week) still does."""
        for headers in (_SUBSCRIPTION_HEALTHY, _SUBSCRIPTION_WEEK_SPENT):
            state = _subscription_state(headers)
            reason = headers["anthropic-ratelimit-unified-overage-disabled-reason"]
            display = format_rate_limit_display(state)
            overage_line = next(line for line in display.splitlines() if "overage" in line)
            assert "[rejected]" not in overage_line
            assert reason in overage_line
            assert "overage" not in format_rate_limit_compact(state)
            # The status is still recorded as sent; only its display changes.
            assert state.unified["overage"].status == headers["anthropic-ratelimit-unified-overage-status"]

        healthy = _subscription_state(_SUBSCRIPTION_HEALTHY)
        assert "[rejected]" not in format_rate_limit_display(healthy)
        spent = _subscription_state(_SUBSCRIPTION_WEEK_SPENT)
        week_line = next(line for line in format_rate_limit_display(spent).splitlines() if "Weekly/7d" in line)
        assert "[rejected]" in week_line
