"""Spending limits for the live server.

These matter more than their size suggests: the thing they protect is a URL that
spends real money per click. A limit that silently does not apply is worse than no
limit, because you stop watching the bill.
"""

from __future__ import annotations

import pytest

from roadsense.web.guards import Budget, Denied, Limits


def test_a_server_with_no_flags_caps_nothing_but_upload_size():
    """The laptop default stays out of the way -- except where AWS already says no."""
    budget = Budget(Limits())
    assert budget.limits.enabled is False
    for _ in range(50):
        budget.reserve("1.2.3.4")
    budget.check_token("")
    budget.check_upload(4 * 1024 * 1024, is_sample=False)
    assert budget.snapshot()["runs_remaining_today"] is None


def test_the_upload_bound_holds_even_unguarded():
    """Rekognition refuses over 5 MB, so forwarding it would only buy an error."""
    with pytest.raises(Denied, match="larger than"):
        Budget(Limits()).check_upload(6 * 1024 * 1024, is_sample=False)


def test_the_daily_cap_refuses_once_it_is_reached():
    budget = Budget(Limits(max_runs_per_day=3))
    for _ in range(3):
        budget.reserve("1.2.3.4")
    with pytest.raises(Denied, match="3 runs for today"):
        budget.reserve("1.2.3.4")


def test_the_daily_cap_counts_every_client_together():
    """Otherwise one cap per address is no cap at all."""
    budget = Budget(Limits(max_runs_per_day=2))
    budget.reserve("1.1.1.1")
    budget.reserve("2.2.2.2")
    with pytest.raises(Denied):
        budget.reserve("3.3.3.3")


def test_the_per_client_cap_is_per_client():
    budget = Budget(Limits(max_runs_per_ip_hour=2))
    budget.reserve("1.1.1.1")
    budget.reserve("1.1.1.1")
    with pytest.raises(Denied, match="runs in an hour"):
        budget.reserve("1.1.1.1")
    budget.reserve("2.2.2.2")  # a different client is unaffected


def test_a_refused_run_is_not_counted_against_the_day():
    """A rejected request costs nothing, so it must not consume budget."""
    budget = Budget(Limits(max_runs_per_day=10, max_runs_per_ip_hour=1))
    budget.reserve("1.1.1.1")
    with pytest.raises(Denied):
        budget.reserve("1.1.1.1")
    assert budget.snapshot()["runs_today"] == 1


def test_the_daily_counter_rolls_over():
    budget = Budget(Limits(max_runs_per_day=1))
    budget.reserve("1.1.1.1")
    with pytest.raises(Denied):
        budget.reserve("1.1.1.1")
    budget._day = "1999-01-01"  # pretend the clock moved
    budget.reserve("1.1.1.1")


def test_samples_only_refuses_uploads_but_allows_samples():
    budget = Budget(Limits(samples_only=True))
    budget.check_upload(1234, is_sample=True)
    with pytest.raises(Denied, match="bundled sample images"):
        budget.check_upload(1234, is_sample=False)


def test_oversized_uploads_are_refused_with_the_limit_named():
    budget = Budget(Limits(max_upload_bytes=1024))
    budget.check_upload(1024, is_sample=False)
    with pytest.raises(Denied, match="larger than"):
        budget.check_upload(1025, is_sample=False)


def test_a_sample_is_never_measured_against_the_upload_limit():
    budget = Budget(Limits(max_upload_bytes=1))
    budget.check_upload(10_000_000, is_sample=True)


@pytest.mark.parametrize("presented", ["", "wrong", "sekret ", " secret"])
def test_a_wrong_token_is_refused(presented):
    budget = Budget(Limits(token="secret"))
    with pytest.raises(Denied, match="access token"):
        budget.check_token(presented)


def test_the_right_token_passes_and_no_token_configured_passes_anything():
    Budget(Limits(token="secret")).check_token("secret")
    Budget(Limits()).check_token("anything at all")


def test_snapshot_reports_what_is_enforced():
    snap = Budget(Limits(max_runs_per_day=5, samples_only=True, token="x")).snapshot()
    assert snap == {
        "guarded": True,
        "runs_today": 0,
        "runs_remaining_today": 5,
        "samples_only": True,
        "token_required": True,
    }


def test_public_shorthand_actually_limits_all_three_ways():
    """The flag people will actually use must not be a no-op."""
    limits = Limits(max_runs_per_day=200, max_runs_per_ip_hour=10, samples_only=True)
    assert limits.enabled
    budget = Budget(limits)
    for _ in range(10):
        budget.reserve("9.9.9.9")
    with pytest.raises(Denied):
        budget.reserve("9.9.9.9")
    with pytest.raises(Denied):
        budget.check_upload(10, is_sample=False)
