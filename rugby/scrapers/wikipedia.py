"""Wikipedia scraper for Super Rugby match results and fixtures."""

import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from bs4 import BeautifulSoup, NavigableString

from .cache import HttpCache

log = logging.getLogger(__name__)


def _count_tries_in_cell(td) -> int:
    """Count tries from a match detail <td> element.

    Handles two formats:
    - Old: <b>Try:</b> <a>Name</a><br/><a>Name2</a> (2)<br/><b>Con:</b>...
    - New: <b>Try:</b> <a>Name</a> 16'<a>Name2</a> (2) 44', 68'<b>Con:</b>...
    """
    try_tag = None
    for b in td.find_all('b'):
        if b.get_text().strip().rstrip(':').lower() == 'try':
            try_tag = b
            break
    if not try_tag:
        return 0

    total = 0
    prev_was_link = False
    for sibling in try_tag.next_siblings:
        if sibling.name == 'b':
            break
        if sibling.name == 'a':
            total += 1
            prev_was_link = True
        elif isinstance(sibling, NavigableString):
            text = str(sibling).strip()
            if not text:
                prev_was_link = False
                continue
            multi = re.match(r'^\((\d+)\)', text)
            if multi and prev_was_link:
                # Previous link was counted as 1, adjust to N
                total += int(multi.group(1)) - 1
            prev_was_link = False
        else:
            prev_was_link = False
    return total


@dataclass
class TeamInfo:
    """Team identity from scraped data."""
    name: str
    country: str


@dataclass
class RawMatch:
    """A scraped match result."""
    date: date
    home_team: TeamInfo
    away_team: TeamInfo
    home_score: int
    away_score: int
    home_tries: Optional[int] = None
    away_tries: Optional[int] = None
    round: Optional[int] = None
    venue: Optional[str] = None

    def __eq__(self, other):
        if not isinstance(other, RawMatch):
            return NotImplemented
        return (self.date, self.home_team.name, self.away_team.name) == (
            other.date, other.home_team.name, other.away_team.name
        )

    def __hash__(self):
        return hash((self.date, self.home_team.name, self.away_team.name))


@dataclass
class RawFixture:
    """A scraped upcoming fixture (no score)."""
    date: date
    home_team: TeamInfo
    away_team: TeamInfo
    venue: Optional[str] = None
    round: Optional[int] = None


# Complete alias map from Rust implementation: lowercase alias -> (canonical name, country)
SUPER_RUGBY_ALIASES: Dict[str, Tuple[str, str]] = {
    # New Zealand teams
    "blues": ("Blues", "NewZealand"),
    "auckland blues": ("Blues", "NewZealand"),
    "chiefs": ("Chiefs", "NewZealand"),
    "waikato chiefs": ("Chiefs", "NewZealand"),
    "crusaders": ("Crusaders", "NewZealand"),
    "canterbury crusaders": ("Crusaders", "NewZealand"),
    "highlanders": ("Highlanders", "NewZealand"),
    "otago highlanders": ("Highlanders", "NewZealand"),
    "hurricanes": ("Hurricanes", "NewZealand"),
    "wellington hurricanes": ("Hurricanes", "NewZealand"),
    "moana pasifika": ("Moana Pasifika", "NewZealand"),
    # Australian teams
    "brumbies": ("Brumbies", "Australia"),
    "act brumbies": ("Brumbies", "Australia"),
    "reds": ("Reds", "Australia"),
    "queensland reds": ("Reds", "Australia"),
    "waratahs": ("Waratahs", "Australia"),
    "nsw waratahs": ("Waratahs", "Australia"),
    "new south wales waratahs": ("Waratahs", "Australia"),
    "force": ("Force", "Australia"),
    "western force": ("Force", "Australia"),
    "rebels": ("Rebels", "Australia"),
    "melbourne rebels": ("Rebels", "Australia"),
    # South African teams
    "bulls": ("Bulls", "SouthAfrica"),
    "blue bulls": ("Bulls", "SouthAfrica"),
    "northern bulls": ("Bulls", "SouthAfrica"),
    "vodacom bulls": ("Bulls", "SouthAfrica"),
    "northern transvaal": ("Bulls", "SouthAfrica"),
    "lions": ("Lions", "SouthAfrica"),
    "golden lions": ("Lions", "SouthAfrica"),
    "johannesburg lions": ("Lions", "SouthAfrica"),
    "emirates lions": ("Lions", "SouthAfrica"),
    "cats": ("Lions", "SouthAfrica"),
    "golden cats": ("Lions", "SouthAfrica"),
    "gauteng lions": ("Lions", "SouthAfrica"),
    "transvaal": ("Lions", "SouthAfrica"),
    "auto & general lions": ("Lions", "SouthAfrica"),
    "mtn lions": ("Lions", "SouthAfrica"),
    "sharks": ("Sharks", "SouthAfrica"),
    "natal sharks": ("Sharks", "SouthAfrica"),
    "durban sharks": ("Sharks", "SouthAfrica"),
    "cell c sharks": ("Sharks", "SouthAfrica"),
    "natal": ("Sharks", "SouthAfrica"),
    "coastal sharks": ("Sharks", "SouthAfrica"),
    "the sharks": ("Sharks", "SouthAfrica"),
    "stormers": ("Stormers", "SouthAfrica"),
    "western province stormers": ("Stormers", "SouthAfrica"),
    "dhl stormers": ("Stormers", "SouthAfrica"),
    "western province": ("Stormers", "SouthAfrica"),
    "vodacom stormers": ("Stormers", "SouthAfrica"),
    "western stormers": ("Stormers", "SouthAfrica"),
    "cheetahs": ("Cheetahs", "SouthAfrica"),
    "free state cheetahs": ("Cheetahs", "SouthAfrica"),
    "central cheetahs": ("Cheetahs", "SouthAfrica"),
    "free state": ("Cheetahs", "SouthAfrica"),
    "toyota cheetahs": ("Cheetahs", "SouthAfrica"),
    "vodacom cheetahs": ("Cheetahs", "SouthAfrica"),
    "kings": ("Kings", "SouthAfrica"),
    "southern kings": ("Kings", "SouthAfrica"),
    # Other teams
    "sunwolves": ("Sunwolves", "Japan"),
    "hito-communications sunwolves": ("Sunwolves", "Japan"),
    "jaguares": ("Jaguares", "Argentina"),
    "fijian drua": ("Fijian Drua", "Fiji"),
    "drua": ("Fijian Drua", "Fiji"),
}

# Month name lookups
_MONTHS_FULL = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}
_MONTHS_ABBREV = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "may": 5, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def normalize_team_name(name: str) -> Optional[TeamInfo]:
    """Resolve a scraped team name to canonical name via alias lookup.

    Tries direct match first, then substring match.
    """
    cleaned = re.sub(r"[\[\]*\u2020\u2021]", "", name).strip().lower()
    if not cleaned:
        return None

    # Direct match
    if cleaned in SUPER_RUGBY_ALIASES:
        canon, country = SUPER_RUGBY_ALIASES[cleaned]
        return TeamInfo(name=canon, country=country)

    # Substring match
    for alias, (canon, country) in SUPER_RUGBY_ALIASES.items():
        if cleaned in alias or alias in cleaned:
            return TeamInfo(name=canon, country=country)

    return None


def extract_date(text: str) -> Optional[date]:
    """Extract a date from text, trying 4 patterns in order.

    1. "1 January 2024"
    2. "January 1, 2024"
    3. "1 jan 2024"
    4. "2024-01-01"
    """
    # Pattern 1: "31 January 2020"
    m = re.search(
        r"(\d{1,2})\s+(January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+(\d{4})",
        text, re.IGNORECASE,
    )
    if m:
        day, month_str, year = int(m.group(1)), m.group(2).lower(), int(m.group(3))
        return date(year, _MONTHS_FULL[month_str], day)

    # Pattern 2: "January 31, 2020"
    m = re.search(
        r"(January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})",
        text, re.IGNORECASE,
    )
    if m:
        month_str, day, year = m.group(1).lower(), int(m.group(2)), int(m.group(3))
        return date(year, _MONTHS_FULL[month_str], day)

    # Pattern 3: "1 Jan 2024" (abbreviated months)
    m = re.search(
        r"(\d{1,2})\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*(\d{4})",
        text, re.IGNORECASE,
    )
    if m:
        day, month_str, year = int(m.group(1)), m.group(2).lower(), int(m.group(3))
        return date(year, _MONTHS_ABBREV[month_str], day)

    # Pattern 4: ISO "2024-03-15"
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", text)
    if m:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))

    return None


def _infer_rounds(fixtures: List, day_gap: int = 3) -> None:
    """Group fixtures into rounds based on date proximity.

    Dates within day_gap days of the round start are assigned the same round.
    Modifies fixtures in-place, setting the .round attribute.
    """
    if not fixtures:
        return

    dates = sorted(set(f.date for f in fixtures))
    date_to_round: Dict[date, int] = {}
    current_round = 1
    round_start = dates[0]

    for d in dates:
        if (d - round_start).days > day_gap:
            current_round += 1
            round_start = d
        date_to_round[d] = current_round

    for f in fixtures:
        f.round = date_to_round[f.date]


class WikipediaScraper:
    """Scraper for Super Rugby match data from Wikipedia."""

    BASE_URL = "https://en.wikipedia.org/wiki/"

    def __init__(self, cache: HttpCache):
        self.cache = cache

    def get_season_urls(self, year: int) -> List[str]:
        """Get Wikipedia URLs for a season, based on competition era.

        Older seasons (1996-2015) have all match data on the season page.
        From 2016+, match results are on separate List_of_ pages.
        """
        base = self.BASE_URL
        if 1996 <= year <= 2005:
            return [f"{base}{year}_Super_12_season"]
        elif 2006 <= year <= 2010:
            return [f"{base}{year}_Super_14_season"]
        elif 2011 <= year <= 2015:
            return [f"{base}{year}_Super_Rugby_season"]
        elif 2016 <= year <= 2021:
            return [
                f"{base}{year}_Super_Rugby_season",
                f"{base}List_of_{year}_Super_Rugby_matches",
            ]
        else:
            # 2022+ Super Rugby Pacific
            return [
                f"{base}{year}_Super_Rugby_Pacific_season",
                f"{base}List_of_{year}_Super_Rugby_Pacific_matches",
            ]

    def fetch_season(self, year: int) -> List[RawMatch]:
        """Fetch and parse all matches for a season."""
        all_matches: List[RawMatch] = []

        for url in self.get_season_urls(year):
            html = self.cache.fetch(url)
            if html is None:
                log.warning("Could not fetch %s", url)
                continue
            matches = self.parse_page(html)
            all_matches.extend(matches)
            log.info("Parsed %d matches from %s", len(matches), url)

        # Deduplicate
        seen = set()
        unique = []
        for m in all_matches:
            key = (m.date, m.home_team.name, m.away_team.name)
            if key not in seen:
                seen.add(key)
                unique.append(m)

        unique.sort(key=lambda m: (m.date, m.home_team.name))
        _infer_rounds(unique)
        return unique

    def fetch_all(self, start_year: int = 1996, end_year: Optional[int] = None) -> List[RawMatch]:
        """Fetch all seasons from start_year to end_year."""
        if end_year is None:
            end_year = datetime.now().year

        all_matches: List[RawMatch] = []
        for year in range(start_year, end_year + 1):
            try:
                matches = self.fetch_season(year)
                all_matches.extend(matches)
                log.info("Season %d: %d matches", year, len(matches))
            except Exception as e:
                log.warning("Failed to fetch season %d: %s", year, e)

        return all_matches

    def fetch_fixtures(self, year: int) -> List[RawFixture]:
        """Fetch upcoming fixtures (matches without scores) for a season."""
        fixtures: List[RawFixture] = []
        score_re = re.compile(r"\d{1,3}\s*[–\-]\s*\d{1,3}")

        for url in self.get_season_urls(year):
            html = self.cache.fetch(url)
            if html is None:
                continue

            soup = BeautifulSoup(html, "html.parser")

            for event in soup.find_all("div", itemtype="http://schema.org/SportsEvent"):
                event_text = event.get_text()
                match_date = extract_date(event_text)

                teams = event.find_all("span", class_="fn")
                if len(teams) < 2:
                    continue

                home_info = normalize_team_name(teams[0].get_text())
                away_info = normalize_team_name(teams[1].get_text())

                # Skip if match already has a score
                has_score = any(score_re.search(td.get_text()) for td in event.find_all("td"))
                if has_score:
                    continue

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
        _infer_rounds(unique)
        return unique

    def parse_page(self, html: str) -> List[RawMatch]:
        """Parse HTML page using 4 strategies in order, deduplicating results."""
        soup = BeautifulSoup(html, "html.parser")
        matches: List[RawMatch] = []

        # Strategy 1: schema.org/SportsEvent
        matches.extend(self._parse_sports_events(soup))

        # Strategy 2: collapsible tables
        matches.extend(self._parse_collapsible_tables(soup))

        # Strategy 3: wikitable format
        matches.extend(self._parse_tables(soup))

        # Strategy 4: text-based regex fallback
        matches.extend(self._parse_text_content(soup))

        # Deduplicate by (date, home, away)
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
        """Strategy 1: Parse div[itemtype='http://schema.org/SportsEvent']."""
        matches = []
        score_re = re.compile(r"(\d{1,3})\s*[–\-]\s*(\d{1,3})")

        for event in soup.find_all("div", itemtype="http://schema.org/SportsEvent"):
            event_text = event.get_text()
            match_date = extract_date(event_text)

            teams = event.find_all("span", class_="fn")
            if len(teams) < 2:
                continue

            home_info = normalize_team_name(teams[0].get_text())
            away_info = normalize_team_name(teams[1].get_text())

            # Extract score
            score_match = None
            for td in event.find_all("td"):
                score_match = score_re.search(td.get_text())
                if score_match:
                    break

            if not score_match:
                continue

            if match_date and home_info and away_info:
                loc = event.find("span", class_="location")
                venue = loc.get_text().strip() if loc else None

                # Extract try counts from detail cells
                # In SportsEvent format: home details are before score, away after
                tds = event.find_all("td")
                score_idx = None
                for i, td in enumerate(tds):
                    if score_re.search(td.get_text()):
                        score_idx = i
                        break
                home_tries = None
                away_tries = None
                if score_idx is not None:
                    # Home detail cell: first cell with Try: before score
                    for td in tds[score_idx + 1:]:
                        t = _count_tries_in_cell(td)
                        if 'Try' in td.get_text() or t > 0:
                            if home_tries is None:
                                home_tries = t
                            elif away_tries is None:
                                away_tries = t
                                break

                matches.append(RawMatch(
                    date=match_date,
                    home_team=home_info,
                    away_team=away_info,
                    home_score=int(score_match.group(1)),
                    away_score=int(score_match.group(2)),
                    venue=venue,
                    home_tries=home_tries,
                    away_tries=away_tries,
                ))

        return matches

    def _parse_collapsible_tables(self, soup: BeautifulSoup) -> List[RawMatch]:
        """Strategy 2: Parse table.mw-collapsible (older seasons)."""
        matches = []
        score_re = re.compile(r"(\d{1,3})\s*[–\-]\s*(\d{1,3})")

        for table in soup.find_all("table", class_="mw-collapsible"):
            rows = table.find_all("tr")
            for row_idx, row in enumerate(rows):
                cells = row.find_all("td")
                if len(cells) < 4:
                    continue

                row_text = row.get_text()
                match_date = extract_date(row_text)

                # Look for score pattern
                score_match = None
                score_cell_idx = None
                for i, cell in enumerate(cells):
                    score_match = score_re.search(cell.get_text())
                    if score_match:
                        score_cell_idx = i
                        break

                if not score_match or score_cell_idx is None:
                    continue

                # Home team is before score, away team is after
                home_text = cells[score_cell_idx - 1].get_text() if score_cell_idx > 0 else ""
                away_text = cells[score_cell_idx + 1].get_text() if score_cell_idx + 1 < len(cells) else ""

                home_info = normalize_team_name(home_text)
                away_info = normalize_team_name(away_text)

                if match_date and home_info and away_info:
                    venue_text = cells[-1].get_text().strip() if len(cells) > score_cell_idx + 2 else None

                    # Extract try counts from the next row (detail row)
                    home_tries = None
                    away_tries = None
                    if row_idx + 1 < len(rows):
                        detail_cells = rows[row_idx + 1].find_all("td")
                        for dc in detail_cells:
                            if 'Try' in dc.get_text():
                                t = _count_tries_in_cell(dc)
                                if home_tries is None:
                                    home_tries = t
                                elif away_tries is None:
                                    away_tries = t
                                    break

                    matches.append(RawMatch(
                        date=match_date,
                        home_team=home_info,
                        away_team=away_info,
                        home_score=int(score_match.group(1)),
                        away_score=int(score_match.group(2)),
                        venue=venue_text,
                        home_tries=home_tries,
                        away_tries=away_tries,
                    ))

        return matches

    def _parse_tables(self, soup: BeautifulSoup) -> List[RawMatch]:
        """Strategy 3: Parse table.wikitable rows."""
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

                home_info = normalize_team_name(home_text)
                away_info = normalize_team_name(away_text)

                if match_date and home_info and away_info:
                    matches.append(RawMatch(
                        date=match_date,
                        home_team=home_info,
                        away_team=away_info,
                        home_score=int(score_match.group(1)),
                        away_score=int(score_match.group(2)),
                    ))

        return matches

    def _parse_text_content(self, soup: BeautifulSoup) -> List[RawMatch]:
        """Strategy 4: Regex-based text content parsing (fallback for very old pages)."""
        matches = []
        text = soup.get_text()

        pattern = re.compile(
            r"(\d{1,2}\s*(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*\d{4})"
            r"[^\|]*\|?\s*([a-z][a-z\s]*?)\s*\|?\s*"
            r"(\d{1,3})\s*[-–]\s*(\d{1,3})"
            r"\s*\|?\s*([a-z][a-z\s]*?)(?:\s*\||$)",
            re.IGNORECASE | re.MULTILINE,
        )

        for m in pattern.finditer(text):
            match_date = extract_date(m.group(1))
            home_info = normalize_team_name(m.group(2).strip())
            away_info = normalize_team_name(m.group(5).strip())

            if match_date and home_info and away_info:
                matches.append(RawMatch(
                    date=match_date,
                    home_team=home_info,
                    away_team=away_info,
                    home_score=int(m.group(3)),
                    away_score=int(m.group(4)),
                ))

        return matches

    def parse_file(self, path: Path) -> List[RawMatch]:
        """Parse a cached HTML file."""
        html = path.read_text()
        matches = self.parse_page(html)
        _infer_rounds(matches)
        return matches

    def parse_directory(self, directory: Path) -> List[RawMatch]:
        """Parse all cached HTML files in a directory."""
        all_matches: List[RawMatch] = []

        for html_file in sorted(directory.glob("*.html")):
            if html_file.name.endswith(".meta.json"):
                continue
            try:
                matches = self.parse_file(html_file)
                all_matches.extend(matches)
                log.info("Parsed %d matches from %s", len(matches), html_file.name)
            except Exception as e:
                log.warning("Failed to parse %s: %s", html_file.name, e)

        # Deduplicate
        seen = set()
        unique = []
        for m in all_matches:
            key = (m.date, m.home_team.name, m.away_team.name)
            if key not in seen:
                seen.add(key)
                unique.append(m)

        unique.sort(key=lambda m: (m.date, m.home_team.name))
        return unique
