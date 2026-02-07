"""Lassen.co.nz scraper stub (not yet implemented)."""

import logging
from typing import List

from .wikipedia import RawMatch

log = logging.getLogger(__name__)


class LassenScraper:
    """Scraper for match round numbers and times from lassen.co.nz.

    Stub implementation - returns empty lists.
    """

    def fetch_season(self, year: int) -> List[RawMatch]:
        log.debug("Lassen scraper not implemented, returning empty list")
        return []

    def fetch_all(self) -> List[RawMatch]:
        return []
