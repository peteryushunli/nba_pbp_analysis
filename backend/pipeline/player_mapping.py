"""Build player name + position mapping from Basketball Reference player index.

Fetches the 26 alphabetical player index pages from BR to get:
- BR player ID → full name
- BR player ID → position (PG, SG, SF, PF, C or combinations)
Then maps combination positions to one of 5 primary positions.
"""

import re
import time
import pandas as pd
import requests
from pathlib import Path

from backend.config import settings


# Map BR raw positions to 5-position system
POSITION_MAP = {
    "PG": "PG",
    "SG": "SG",
    "SF": "SF",
    "PF": "PF",
    "C": "C",
    # Single-letter positions (older data)
    "G": "SG",   # Default guard to SG
    "F": "SF",    # Default forward to SF
    # Common combinations → primary position
    "G-F": "SG",
    "F-G": "SF",
    "F-C": "PF",
    "C-F": "C",
    "C-G": "C",
    "G-C": "SG",
    "PG-SG": "PG",
    "SG-PG": "SG",
    "SG-SF": "SG",
    "SF-SG": "SF",
    "SF-PF": "SF",
    "PF-SF": "PF",
    "PF-C": "PF",
    "C-PF": "C",
}


def map_position(raw_pos: str) -> str:
    """Map a BR position string to one of 5 positions."""
    if not raw_pos or pd.isna(raw_pos):
        return "SF"  # default
    raw = raw_pos.strip()
    if raw in POSITION_MAP:
        return POSITION_MAP[raw]
    # Try first part
    first = raw.split("-")[0].strip()
    if first in POSITION_MAP:
        return POSITION_MAP[first]
    return "SF"  # fallback


def build_player_reference(
    output_path: Path | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Build player reference from BR player index pages.

    Fetches 26 pages (a-z) from basketball-reference.com/players/{letter}/
    and extracts player name, BR ID, and position.

    Returns DataFrame with columns: br_id, name, position, position_raw
    """
    output_path = output_path or settings.DATA_DIR / "external" / "player_reference.parquet"

    if output_path.exists() and not force:
        print(f"Player reference already exists: {output_path}")
        return pd.read_parquet(output_path)

    # First try the GitHub mapping for names
    github_mapping = _load_github_mapping()

    # Then scrape BR for position data
    print("Scraping Basketball Reference player index...")
    all_players = []

    for letter in "abcdefghijklmnopqrstuvwxyz":
        url = f"https://www.basketball-reference.com/players/{letter}/"
        try:
            resp = requests.get(url, headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) "
                              "Chrome/120.0.0.0 Safari/537.36"
            }, timeout=15)
            resp.raise_for_status()
        except Exception as e:
            print(f"  Failed to fetch letter {letter}: {e}")
            continue

        # Parse the HTML table
        players = _parse_player_index_page(resp.text)
        all_players.extend(players)
        print(f"  Letter {letter}: {len(players)} players")
        time.sleep(3.5)  # Respect rate limits

    if not all_players:
        print("WARNING: Could not scrape BR. Using GitHub mapping only.")
        df = github_mapping[['BBRefID', 'BBRefName']].rename(
            columns={'BBRefID': 'br_id', 'BBRefName': 'name'}
        )
        df['position_raw'] = 'Unknown'
        df['position'] = 'SF'
        df.to_parquet(str(output_path), index=False)
        return df

    br_df = pd.DataFrame(all_players)
    print(f"Total players scraped: {len(br_df)}")

    # Merge with GitHub mapping for any name improvements
    if github_mapping is not None:
        # GitHub names are often better formatted
        github_names = github_mapping.set_index('BBRefID')['BBRefName'].to_dict()
        br_df['name'] = br_df['br_id'].map(github_names).fillna(br_df['name'])

    # Map positions
    br_df['position'] = br_df['position_raw'].apply(map_position)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    br_df.to_parquet(str(output_path), engine='pyarrow', index=False)
    print(f"Saved {len(br_df)} players to {output_path}")

    return br_df


def _load_github_mapping() -> pd.DataFrame | None:
    """Load the GitHub NBA player IDs mapping if available."""
    try:
        url = "https://raw.githubusercontent.com/djblechn-su/nba-player-team-ids/master/NBA_Player_IDs.csv"
        df = pd.read_csv(url, encoding='latin-1')
        return df[['BBRefID', 'BBRefName']].dropna(subset=['BBRefID'])
    except Exception as e:
        print(f"Could not load GitHub mapping: {e}")
        return None


def _parse_player_index_page(html: str) -> list[dict]:
    """Parse a BR player index HTML page to extract player info."""
    players = []

    # Find all player rows in the table
    # Pattern: <a href="/players/x/playerid01.html">Player Name</a>
    # followed by position in a <td> tag
    row_pattern = re.compile(
        r'<tr[^>]*>.*?'
        r'<a href="/players/\w/(\w+)\.html">([^<]+)</a>'
        r'.*?</th>'
        r'\s*<td[^>]*>(\d{4})</td>'   # from year
        r'\s*<td[^>]*>(\d{4})</td>'   # to year
        r'\s*<td[^>]*>([^<]*)</td>',  # position
        re.DOTALL
    )

    for match in row_pattern.finditer(html):
        br_id = match.group(1)
        name = match.group(2).strip()
        from_year = int(match.group(3))
        to_year = int(match.group(4))
        pos = match.group(5).strip()

        # Clean up HTML entities
        name = name.replace('&amp;', '&').replace('&#39;', "'")

        players.append({
            'br_id': br_id,
            'name': name,
            'position_raw': pos,
            'from_year': from_year,
            'to_year': to_year,
        })

    return players


def apply_player_names_and_positions(
    on_off_dir: Path | None = None,
    player_ref: pd.DataFrame | None = None,
    force: bool = False,
):
    """Update on/off Parquet files with player names and positions.

    Reads each on_off_{season}.parquet, joins with player reference,
    and overwrites with updated names and positions.
    """
    on_off_dir = on_off_dir or settings.DATA_DIR / "processed" / "on_off"

    if player_ref is None:
        ref_path = settings.DATA_DIR / "external" / "player_reference.parquet"
        if not ref_path.exists():
            print("No player reference found. Run build_player_reference() first.")
            return
        player_ref = pd.read_parquet(ref_path)

    # Build lookup dicts
    name_map = player_ref.set_index('br_id')['name'].to_dict()
    pos_map = player_ref.set_index('br_id')['position'].to_dict()

    for parquet_path in sorted(on_off_dir.glob("on_off_*.parquet")):
        df = pd.read_parquet(parquet_path)

        # Update names
        df['player_name'] = df['player_id'].map(name_map).fillna(df['player_id'])
        # Update positions
        df['position'] = df['player_id'].map(pos_map).fillna('SF')

        df.to_parquet(str(parquet_path), engine='pyarrow', index=False)
        season = parquet_path.stem.split('_')[-1]
        mapped = (df['player_name'] != df['player_id']).sum()
        print(f"  Season {season}: {mapped}/{len(df)} names mapped, "
              f"positions: {df['position'].value_counts().to_dict()}")
