"""Live data ingestion from Google Finance."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Iterable, List

import httpx

from ..utils import Quote, ensure_list

LOGGER = logging.getLogger(__name__)

BASE_URL = "https://www.google.com/finance/quote/{symbol}"
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0 Safari/537.36"
PRICE_REGEX = re.compile(r'data-last-price="([0-9.,+-]+)"')
PRICE_TEXT_REGEX = re.compile(r'<div class="YMlKec fxKbKc">([^<]+)</div>')
CURRENCY_CLEANUP = re.compile(r"[0-9.,\s\u202f\xa0]")


@dataclass(slots=True)
class GoogleFinanceClient:
    """HTTP client responsible for downloading quotes."""

    timeout: float = 10.0
    _client: httpx.Client = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._client = httpx.Client(
            headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"},
            timeout=self.timeout,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "GoogleFinanceClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def _extract_price(self, html: str) -> float:
        price_match = PRICE_REGEX.search(html)
        if price_match:
            # Google often reports the unformatted value via data-last-price.
            return float(price_match.group(1).replace(",", ""))
        # Fall back to the visible price string which may include currency glyphs.
        text_match = PRICE_TEXT_REGEX.search(html)
        if not text_match:
            raise ValueError("Unable to parse price from Google Finance response")
        cleaned = text_match.group(1).strip()
        cleaned = cleaned.replace(" ", "").replace(" ", "")
        cleaned = cleaned.replace(",", "").replace("−", "-").replace("–", "-")
        cleaned = re.sub(r"[^0-9.\-]", "", cleaned)
        if not cleaned or cleaned == "-":
            raise ValueError("Unable to parse price from Google Finance response")
        return float(cleaned)

    def _extract_currency(self, html: str) -> str:
        text_match = PRICE_TEXT_REGEX.search(html)
        if not text_match:
            return "INR"
        raw = text_match.group(1)
        symbol_only = CURRENCY_CLEANUP.sub("", raw).strip()
        if not symbol_only or symbol_only in {"₹", "−"}:
            return "INR"
        return symbol_only

    def fetch_quote(self, symbol: str) -> Quote:
        url = BASE_URL.format(symbol=symbol)
        LOGGER.debug("Fetching quote", extra={"symbol": symbol, "url": url})
        response = self._client.get(url)
        response.raise_for_status()
        price = self._extract_price(response.text)
        currency = self._extract_currency(response.text)
        return Quote.create(symbol=symbol, price=price, currency=currency)

    def fetch_quotes(self, symbols: Iterable[str]) -> List[Quote]:
        quotes: List[Quote] = []
        for symbol in ensure_list(symbols):
            try:
                quotes.append(self.fetch_quote(symbol))
            except Exception as exc:  # pragma: no cover - network errors
                LOGGER.error("Failed to fetch %s: %s", symbol, exc)
        return quotes


def parse_quote_from_html(symbol: str, html: str) -> Quote:
    """Parse a :class:`Quote` from a HTML payload (useful for testing)."""

    dummy_client = GoogleFinanceClient()
    try:
        price = dummy_client._extract_price(html)
        currency = dummy_client._extract_currency(html)
    finally:
        dummy_client.close()
    return Quote.create(symbol=symbol, price=price, currency=currency)
