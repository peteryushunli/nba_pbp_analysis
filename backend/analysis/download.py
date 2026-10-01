"""Cache public regular-season archives; extract only the expected CSV member."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import shutil
import tarfile
from urllib.request import urlopen

SOURCE = 'https://raw.githubusercontent.com/shufinskiy/nba_data/main/datasets'

def source_kinds(season: int) -> list[str]:
    return ['cdnnba'] if season >= 2026 else ['pbpstats', 'nbastats']

def download_archive(kind: str, season: int, data_dir: Path, postseason: bool = False) -> Path:
    if kind not in ['pbpstats', 'nbastats', 'cdnnba'] or not 2001 <= season <= 2100:
        raise ValueError('Unsupported provider or ending-year season')
    base = data_dir / 'external' / kind
    base.mkdir(parents=True, exist_ok=True)
    stem = f'{kind}_{"po_" if postseason else ""}{season-1}'
    csv_path = base / f'{stem}.csv'
    if csv_path.exists():
        return csv_path
    url = f'{SOURCE}/{stem}.tar.xz'
    archive = base / f'{stem}.tar.xz'
    temporary = archive.with_suffix('.part')
    with urlopen(url, timeout=60) as response, temporary.open('wb') as out:
        shutil.copyfileobj(response, out)
    temporary.replace(archive)
    csv_temp = csv_path.with_suffix('.part')
    with tarfile.open(archive) as tar:
        member = tar.getmember(csv_path.name)
        if not member.isfile():
            raise ValueError('Archive member is not a regular CSV file')
        with tar.extractfile(member) as source, csv_temp.open('wb') as dest:
            shutil.copyfileobj(source, dest)
    csv_temp.replace(csv_path)
    csv_path.with_suffix('.source.json').write_text(json.dumps({
        'url':url, 'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
        'csv_sha256':hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        'retrieved_utc':datetime.now(timezone.utc).isoformat()}, indent=2))
    print(f'Downloaded {csv_path}', flush=True)
    return csv_path


def download_archives(seasons: list[int], data_dir: Path) -> None:
    for season in seasons:
        if not 2001 <= season <= 2100:
            raise ValueError('Season is the ending year, e.g. 2025 for 2024–25')
        for kind in source_kinds(season):
            download_archive(kind, season, data_dir)
