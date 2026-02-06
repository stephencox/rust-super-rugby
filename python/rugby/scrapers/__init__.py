"""Web scrapers for rugby match data."""

from .cache import HttpCache
from .wikipedia import (
    WikipediaScraper,
    RawMatch,
    RawFixture,
    TeamInfo,
    SUPER_RUGBY_ALIASES,
    normalize_team_name,
    extract_date,
)
from .sixnations import (
    SixNationsScraper,
    SIX_NATIONS_ALIASES,
    VENUE_MAP,
    normalize_six_nations_team,
)
from .sarugby import SaRugbyScraper
from .lassen import LassenScraper

__all__ = [
    "HttpCache",
    "WikipediaScraper",
    "SixNationsScraper",
    "SaRugbyScraper",
    "LassenScraper",
    "RawMatch",
    "RawFixture",
    "TeamInfo",
    "SUPER_RUGBY_ALIASES",
    "SIX_NATIONS_ALIASES",
    "VENUE_MAP",
    "normalize_team_name",
    "normalize_six_nations_team",
    "extract_date",
]
