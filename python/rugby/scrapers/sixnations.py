"""Wikipedia scraper for Six Nations Championship match results and fixtures."""

import logging
import re
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from bs4 import BeautifulSoup

from .cache import HttpCache
from .wikipedia import (
    RawFixture,
    RawMatch,
    TeamInfo,
    extract_date,
    _infer_rounds,
)

log = logging.getLogger(__name__)

# Six Nations alias map: lowercase alias -> (canonical name, country)
SIX_NATIONS_ALIASES: Dict[str, Tuple[str, str]] = {
    # England
    "england": ("England", "England"),
    "england xv": ("England", "England"),
    "eng": ("England", "England"),
    # France
    "france": ("France", "France"),
    "france xv": ("France", "France"),
    "fra": ("France", "France"),
    # Ireland
    "ireland": ("Ireland", "Ireland"),
    "ireland xv": ("Ireland", "Ireland"),
    "ire": ("Ireland", "Ireland"),
    # Wales
    "wales": ("Wales", "Wales"),
    "wales xv": ("Wales", "Wales"),
    "wal": ("Wales", "Wales"),
    # Scotland
    "scotland": ("Scotland", "Scotland"),
    "scotland xv": ("Scotland", "Scotland"),
    "sco": ("Scotland", "Scotland"),
    # Italy
    "italy": ("Italy", "Italy"),
    "italy xv": ("Italy", "Italy"),
    "ita": ("Italy", "Italy"),
}

# Home venue for each team
VENUE_MAP: Dict[str, str] = {
    "England": "Twickenham",
    "France": "Stade de France",
    "Ireland": "Aviva Stadium",
    "Wales": "Principality Stadium",
    "Scotland": "Murrayfield",
    "Italy": "Stadio Olimpico",
}


def normalize_six_nations_team(name: str) -> Optional[TeamInfo]:
    """Resolve a scraped team name to canonical Six Nations name."""
    cleaned = re.sub(r"[\[\]*\u2020\u2021]", "", name).strip().lower()
    if not cleaned:
        return None

    # Direct match
    if cleaned in SIX_NATIONS_ALIASES:
        canon, country = SIX_NATIONS_ALIASES[cleaned]
        return TeamInfo(name=canon, country=country)

    # Substring match
    for alias, (canon, country) in SIX_NATIONS_ALIASES.items():
        if cleaned in alias or alias in cleaned:
            return TeamInfo(name=canon, country=country)

    return None


class SixNationsScraper:
    """Scraper for Six Nations Championship data from Wikipedia."""

    BASE_URL = "https://en.wikipedia.org/wiki/"

    def __init__(self, cache: HttpCache):
        self.cache = cache

    def get_season_url(self, year: int) -> str:
        return f"{self.BASE_URL}{year}_Six_Nations_Championship"

    def fetch_season(self, year: int) -> List[RawMatch]:
        """Fetch and parse all matches for a Six Nations season."""
        url = self.get_season_url(year)
        html = self.cache.fetch(url)
        if html is None:
            log.warning("Could not fetch %s", url)
            return []

        matches = self.parse_page(html)
        log.info("Six Nations %d: %d matches", year, len(matches))
        _infer_rounds(matches, day_gap=5)
        return matches

    def fetch_all(self, start_year: int = 2000, end_year: Optional[int] = None) -> List[RawMatch]:
        """Fetch all Six Nations seasons."""
        if end_year is None:
            end_year = datetime.now().year

        all_matches: List[RawMatch] = []
        for year in range(start_year, end_year + 1):
            try:
                matches = self.fetch_season(year)
                all_matches.extend(matches)
            except Exception as e:
                log.warning("Failed to fetch Six Nations %d: %s", year, e)

        return all_matches

    def fetch_fixtures(self, year: int) -> List[RawFixture]:
        """Fetch upcoming Six Nations fixtures."""
        url = self.get_season_url(year)
        html = self.cache.fetch(url)
        if html is None:
            return []

        fixtures: List[RawFixture] = []
        score_re = re.compile(r"\d{1,3}\s*[–\-]\s*\d{1,3}")
        soup = BeautifulSoup(html, "html.parser")

        for event in soup.find_all("div", itemtype="http://schema.org/SportsEvent"):
            event_text = event.get_text()
            match_date = extract_date(event_text)

            teams = event.find_all("span", class_="fn")
            if len(teams) < 2:
                continue

            home_info = normalize_six_nations_team(teams[0].get_text())
            away_info = normalize_six_nations_team(teams[1].get_text())

            has_score = any(score_re.search(td.get_text()) for td in event.find_all("td"))
            if has_score:
                continue

            # Infer venue from home team
            venue = None
            if home_info:
                venue = VENUE_MAP.get(home_info.name)
            if venue is None:
                loc = event.find("span", class_="location")
                venue = loc.get_text().strip() if loc else None

            if match_date and home_info and away_info:
                fixtures.append(RawFixture(
                    date=match_date,
                    home_team=home_info,
                    away_team=away_info,
                    venue=venue,
                ))

        # Deduplicate
        seen = set()
        unique = []
        for f in fixtures:
            key = (f.date, f.home_team.name, f.away_team.name)
            if key not in seen:
                seen.add(key)
                unique.append(f)

        unique.sort(key=lambda f: (f.date, f.home_team.name))
        _infer_rounds(unique, day_gap=5)
        return unique

    def parse_page(self, html: str) -> List[RawMatch]:
        """Parse Six Nations page using same strategies as Wikipedia scraper."""
        soup = BeautifulSoup(html, "html.parser")
        matches: List[RawMatch] = []

        matches.extend(self._parse_sports_events(soup))
        matches.extend(self._parse_tables(soup))

        # Deduplicate
        seen = set()
        unique = []
        for m in matches:
            key = (m.date, m.home_team.name, m.away_team.name)
            if key not in seen:
                seen.add(key)
                unique.append(m)

        unique.sort(key=lambda m: (m.date, m.home_team.name))
        return unique

    def _parse_sports_events(self, soup: BeautifulSoup) -> List[RawMatch]:
        """Parse schema.org/SportsEvent divs."""
        matches = []
        score_re = re.compile(r"(\d{1,3})\s*[–\-]\s*(\d{1,3})")

        for event in soup.find_all("div", itemtype="http://schema.org/SportsEvent"):
            event_text = event.get_text()
            match_date = extract_date(event_text)

            teams = event.find_all("span", class_="fn")
            if len(teams) < 2:
                continue

            home_info = normalize_six_nations_team(teams[0].get_text())
            away_info = normalize_six_nations_team(teams[1].get_text())

            score_match = None
            for td in event.find_all("td"):
                score_match = score_re.search(td.get_text())
                if score_match:
                    break

            if not score_match:
                continue

            if match_date and home_info and away_info:
                venue = VENUE_MAP.get(home_info.name)
                if venue is None:
                    loc = event.find("span", class_="location")
                    venue = loc.get_text().strip() if loc else None

                matches.append(RawMatch(
                    date=match_date,
                    home_team=home_info,
                    away_team=away_info,
                    home_score=int(score_match.group(1)),
                    away_score=int(score_match.group(2)),
                    venue=venue,
                ))

        return matches

    def _parse_tables(self, soup: BeautifulSoup) -> List[RawMatch]:
        """Parse wikitable format."""
        matches = []
        score_re = re.compile(r"(\d{1,3})\s*[–\-]\s*(\d{1,3})")

        for table in soup.find_all("table", class_="wikitable"):
            for row in table.find_all("tr"):
                cells = row.find_all("td")
                if len(cells) < 3:
                    continue

                row_text = row.get_text()
                match_date = extract_date(row_text)

                score_match = None
                score_cell_idx = None
                for i, cell in enumerate(cells):
                    score_match = score_re.search(cell.get_text())
                    if score_match:
                        score_cell_idx = i
                        break

                if not score_match or score_cell_idx is None:
                    continue

                home_text = cells[score_cell_idx - 1].get_text() if score_cell_idx > 0 else ""
                away_text = cells[score_cell_idx + 1].get_text() if score_cell_idx + 1 < len(cells) else ""

                home_info = normalize_six_nations_team(home_text)
                away_info = normalize_six_nations_team(away_text)

                if match_date and home_info and away_info:
                    venue = VENUE_MAP.get(home_info.name)
                    matches.append(RawMatch(
                        date=match_date,
                        home_team=home_info,
                        away_team=away_info,
                        home_score=int(score_match.group(1)),
                        away_score=int(score_match.group(2)),
                        venue=venue,
                    ))

        return matches
