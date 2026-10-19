"""Spending limits for a server that calls paid APIs.

Every run of this agent costs money: two Rekognition calls and a Bedrock completion,
on a GPU instance billed by the hour. None of that is expensive once; all of it is
expensive if a public URL lets strangers loop it.

The guards here are deliberately crude, because the failure they prevent is crude. A
counter that resets daily and a per-client window stop the realistic problem — someone
holding down a button, or a crawler finding the endpoint — without pretending to be
authentication. For anything facing the open internet, put a real front door in front
of this: the point of `require_token` is to make the URL useless on its own, not to be
a credential system.

Defaults are off, with one exception: the upload size bound is always enforced,
because Rekognition will not accept more than 5 MB of image bytes and a server that
forwards a larger file is only paying to receive an error. Everything else -- run
caps, sample-only mode, the shared token -- is opt-in, so a server started with no
flags stays out of the way on a laptop.
"""

from __future__ import annotations

import hmac
import threading
import time
from dataclasses import dataclass, field


@dataclass
class Limits:
    """What a run is allowed to cost, and how often one may be asked for."""

    max_runs_per_day: int = 0
    """0 disables the cap. Counts successful starts, not completions."""
    max_runs_per_ip_hour: int = 0
    max_upload_bytes: int = 5 * 1024 * 1024
    """Always enforced, flags or not. Rekognition refuses image bytes over 5 MB, so a
    larger upload cannot succeed anyway -- rejecting it here turns a confusing API
    error into a sentence the user can act on, and avoids paying to find out."""
    samples_only: bool = False
    """Refuse uploads entirely. Bounds what Rekognition can be asked to look at."""
    token: str = ""
    """When set, every run must present it. Compared in constant time."""

    @property
    def enabled(self) -> bool:
        """Whether anything beyond the always-on upload bound is being enforced."""
        return bool(
            self.max_runs_per_day
            or self.max_runs_per_ip_hour
            or self.samples_only
            or self.token
        )


class Denied(Exception):
    """A run was refused. The message is shown to the caller verbatim."""


@dataclass
class Budget:
    """Tracks what has been spent today and by whom.

    Everything is in memory: a restart forgives the day, which is the right trade for
    a demo. Persisting it would mean a store to operate and would not stop anyone
    determined enough to matter.
    """

    limits: Limits
    _day: str = ""
    _today: int = 0
    _by_ip: dict[str, list[float]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def _roll_day(self, now: float) -> None:
        day = time.strftime("%Y-%m-%d", time.gmtime(now))
        if day != self._day:
            self._day, self._today, self._by_ip = day, 0, {}

    def check_token(self, presented: str) -> None:
        if not self.limits.token:
            return
        if not hmac.compare_digest(presented or "", self.limits.token):
            raise Denied("This demo needs an access token. Ask whoever shared the link.")

    def check_upload(self, size: int, is_sample: bool) -> None:
        if is_sample:
            return
        if self.limits.samples_only:
            raise Denied(
                "This instance only runs the bundled sample images. Clone the repo to "
                "run it on your own photographs."
            )
        if size > self.limits.max_upload_bytes:
            mb = self.limits.max_upload_bytes / 1024 / 1024
            raise Denied(f"That image is larger than the {mb:.0f} MB limit.")

    def reserve(self, client: str) -> None:
        """Count a run against the caps, or refuse it."""
        now = time.time()
        with self._lock:
            self._roll_day(now)

            if self.limits.max_runs_per_day and self._today >= self.limits.max_runs_per_day:
                raise Denied(
                    f"This demo has used its {self.limits.max_runs_per_day} runs for today. "
                    "It resets at midnight UTC, and the repo runs without any limit."
                )

            if self.limits.max_runs_per_ip_hour:
                window = [t for t in self._by_ip.get(client, []) if now - t < 3600]
                if len(window) >= self.limits.max_runs_per_ip_hour:
                    wait = int((3600 - (now - window[0])) / 60) + 1
                    raise Denied(
                        f"That is {self.limits.max_runs_per_ip_hour} runs in an hour from "
                        f"here. Try again in about {wait} minutes."
                    )
                window.append(now)
                self._by_ip[client] = window

            self._today += 1

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            self._roll_day(time.time())
            remaining = (
                self.limits.max_runs_per_day - self._today
                if self.limits.max_runs_per_day
                else None
            )
            return {
                "guarded": self.limits.enabled,
                "runs_today": self._today,
                "runs_remaining_today": remaining,
                "samples_only": self.limits.samples_only,
                "token_required": bool(self.limits.token),
            }
