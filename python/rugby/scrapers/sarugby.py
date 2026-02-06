"""SA Rugby scraper stub (not yet implemented)."""

import logging
from typing import List

from .wikipedia import RawMatch

log = logging.getLogger(__name__)


class SaRugbyScraper:
    """Scraper for South African match data from sarugby.co.za.

    Stub implementation - returns empty lists.
    """

    def fetch_season(self, year: int) -> List[RawMatch]:
        log.debug("SaRugby scraper not implemented, returning empty list")
        return []

    def fetch_all(self) -> List[RawMatch]:
        return []
