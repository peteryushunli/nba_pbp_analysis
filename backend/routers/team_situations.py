"""Serve the reproducible team-state report independently of legacy player ratings."""
import json
from fastapi import APIRouter, HTTPException
from backend.config import settings

router = APIRouter(tags=['team situations'])

@router.get('/team-situations')
def get_team_situations():
    path = settings.PROJECT_ROOT / 'reports' / 'team_situations.json'
    if not path.exists():
        raise HTTPException(status_code=404, detail='Run python -m backend.analysis.run first')
    return json.loads(path.read_text())
