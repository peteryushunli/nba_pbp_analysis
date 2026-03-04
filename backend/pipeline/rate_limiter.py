"""Rate limiter for NBA API requests."""

import time
from typing import TypeVar, Type

from backend.config import settings

T = TypeVar("T")

_last_request_time: float = 0.0


def throttled_request(endpoint_class: Type[T], **kwargs) -> T:
    """
    Instantiate an nba_api endpoint class with rate limiting.

    Enforces settings.NBA_API_DELAY between consecutive calls.
    Retries up to NBA_API_MAX_RETRIES times with exponential backoff on errors.
    """
    global _last_request_time

    elapsed = time.time() - _last_request_time
    if elapsed < settings.NBA_API_DELAY:
        time.sleep(settings.NBA_API_DELAY - elapsed)

    for attempt in range(settings.NBA_API_MAX_RETRIES):
        try:
            _last_request_time = time.time()
            return endpoint_class(**kwargs)
        except Exception:
            if attempt == settings.NBA_API_MAX_RETRIES - 1:
                raise
            wait = 2 ** (attempt + 1)
            time.sleep(wait)

    raise RuntimeError("Unreachable")
