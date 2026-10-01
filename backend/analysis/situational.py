"""Team efficiency relative to the league in the same season, margin and clock cell."""
import numpy as np
import pandas as pd

MARGINS = ['Behind 21+', 'Behind 11–20', 'Behind 6–10', 'Within 5',
           'Ahead 6–10', 'Ahead 11–20', 'Ahead 21+']
TIMES = ['48–44', '44–40', '40–36', '36–32', '32–28', '28–24', '24–20', '20–16', '16–12', '12–8', '8–5', '5–2', '2–0', 'Overtime']
CELL_KEYS = ['season', 'margin_bucket', 'time_bucket']
SUM_COLS = ['off_n', 'def_n', 'off_pts', 'def_pts', 'off_expected', 'def_expected',
            'off_resid', 'def_resid', 'off_schedule_resid', 'def_schedule_resid', 'off_benchmark_fallback', 'def_benchmark_fallback']


def team_perspectives(p: pd.DataFrame) -> pd.DataFrame:
    """One offensive and one defensive team observation per source possession.

    Technical FTs scored by the defense are credited to the actual scoring team,
    without inventing an extra offensive possession.
    """
    frames = []
    for offense in [True, False]:
        v = p.copy()
        v['team'] = v['offense'] if offense else v['defense']
        v['opponent'] = v['defense'] if offense else v['offense']
        v['margin'] = v['margin'] if offense else -v['margin']
        v['off_n'], v['def_n'] = int(offense), int(not offense)
        v['off_pts'] = v['points'] if offense else v['other_points']
        v['def_pts'] = v['other_points'] if offense else v['points']
        frames.append(v)
    v = pd.concat(frames, ignore_index=True)
    v['margin_bucket'] = pd.cut(v.margin, [-np.inf, -20.5, -10.5, -5.5, 5.5, 10.5, 20.5, np.inf],
                                labels=MARGINS).astype(str)
    v['time_bucket'] = pd.cut(v.regulation_remaining,
        [-1,120,300,480,720,960,1200,1440,1680,1920,2160,2400,2640,2880],
        labels=TIMES[-2::-1]).astype(str)
    v.loc[v.period.gt(4),'time_bucket'] = 'Overtime'
    return v


def add_expectations(v: pd.DataFrame) -> pd.DataFrame:
    """Leave-focal-team-out possession-weighted league benchmarks within cells."""
    sums = ['off_n', 'def_n', 'off_pts', 'def_pts']
    totals = v.groupby(CELL_KEYS, observed=True)[sums].transform('sum')
    focal = v.groupby(['team'] + CELL_KEYS, observed=True)[sums].transform('sum')
    v = v.copy()
    for side in ['off', 'def']:
        rate = (totals[f'{side}_pts'] - focal[f'{side}_pts']) / (
            totals[f'{side}_n'] - focal[f'{side}_n']).replace(0, np.nan)
        # A handful of extreme OT cells have no other-team exposure. Fall back
        # to the same-season/time league, and expose the count in summaries.
        fallback = rate.isna() & v[f'{side}_n'].gt(0)
        v[f'{side}_benchmark_fallback'] = fallback.astype(int)
        time_keys = ['season','time_bucket']
        time_total = v.groupby(time_keys, observed=True)[sums].transform('sum')
        time_focal = v.groupby(['team']+time_keys, observed=True)[sums].transform('sum')
        pooled = (time_total[f'{side}_pts']-time_focal[f'{side}_pts']) / (
            time_total[f'{side}_n']-time_focal[f'{side}_n']).replace(0,np.nan)
        rate = rate.fillna(pooled)
        v[f'{side}_expected'] = (rate * v[f'{side}_n']).where(v[f'{side}_n'].gt(0), 0)
        # Both residuals use positive = better.
        sign = 1 if side == 'off' else -1
        v[f'{side}_resid'] = sign * (v[f'{side}_pts'] - v[f'{side}_expected'])
    # Full-season opponent strength adjustment as an exploratory sensitivity.
    overall = v.groupby(['season', 'team'])[sums].sum()
    league = v.groupby('season')[sums].sum()
    for side, opp_side, sign in [('off','def',1), ('def','off',-1)]:
        baseline = overall[f'{opp_side}_pts'] / overall[f'{opp_side}_n']
        league_rate = v.season.map(league[f'{opp_side}_pts'] / league[f'{opp_side}_n'])
        opp_rate = pd.MultiIndex.from_frame(v[['season','opponent']]).map(baseline)
        adjustment = np.asarray(opp_rate) - league_rate
        v[f'{side}_schedule_resid'] = v[f'{side}_resid'] - sign * adjustment * v[f'{side}_n']
    return v


def rate_summary(g: pd.DataFrame, keys: list[str], minimum: int = 300) -> pd.DataFrame:
    out = g.groupby(keys, observed=True)[SUM_COLS].sum()
    out['games'] = g.groupby(keys, observed=True).game_id.nunique()
    for side in ['off','def']:
        n = out[f'{side}_n'].replace(0, np.nan)
        out[f'{side}_rating'] = 100 * out[f'{side}_pts'] / n
        out[f'{side}_league_expected'] = 100 * out[f'{side}_expected'] / n
        out[f'{side}_relative'] = 100 * out[f'{side}_resid'] / n
        out[f'{side}_schedule_relative'] = 100 * out[f'{side}_schedule_resid'] / n
        out[f'{side}_shrunk_relative'] = 100 * out[f'{side}_resid'] / (n + 200)
    out['net_rating'] = out.off_rating - out.def_rating
    out['relative_net'] = out.off_relative + out.def_relative
    out['schedule_adjusted_relative_net'] = out.off_schedule_relative + out.def_schedule_relative
    out['shrunk_relative_net'] = out.off_shrunk_relative + out.def_shrunk_relative
    out['eligible'] = (out.off_n.ge(minimum) & out.def_n.ge(minimum) & out.games.ge(20))
    return out.reset_index()


def profile_masks(v: pd.DataFrame) -> dict[str, pd.Series]:
    last2 = v.period.eq(4) & v.regulation_remaining.le(120)
    close = v.margin.abs().le(5)
    return {
        'Overall': pd.Series(True, index=v.index),
        'Ahead 11+': v.margin.ge(11), 'Behind 11+': v.margin.le(-11),
        'Close (within 5)': close,
        'Clutch (last 5, within 5)': close & (v.period.gt(4) | (v.period.eq(4) & v.regulation_remaining.le(300))),
        'Ahead 11+, excluding final 2': v.margin.ge(11) & ~last2,
        'Behind 11+, excluding final 2': v.margin.le(-11) & ~last2,
        'Trailing 1–10 in Q4/OT': v.margin.between(-10,-1) & v.period.ge(4),
    }


def analyze(p: pd.DataFrame, draws: int = 1000, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if draws < 100:
        raise ValueError('Use at least 100 bootstrap draws')
    v = add_expectations(team_perspectives(p))
    cells = rate_summary(v, ['season', 'team', 'margin_bucket', 'time_bucket'])
    masks = profile_masks(v)
    profiles = pd.concat([rate_summary(v[m], ['season','team']).assign(profile=name)
                          for name,m in masks.items()], ignore_index=True)
    profiles['ci_low'] = np.nan
    profiles['ci_high'] = np.nan
    rng = np.random.default_rng(seed)
    contrasts = []
    # Resample whole games, keeping offense, defense and states paired. League
    # benchmarks are held fixed; intervals are exploratory, not simultaneous.
    for (season,team), group in v.groupby(['season','team'], sort=True):
        games = group.game_id.unique()
        weights = rng.multinomial(len(games), np.full(len(games),1/len(games)), size=draws)
        boot, point = {}, {}
        for name,mask in masks.items():
            part = group[mask.loc[group.index]]
            a = part.groupby('game_id')[['off_n','def_n','off_resid','def_resid']].sum().reindex(games,fill_value=0)
            totals = weights @ a.to_numpy()
            with np.errstate(divide='ignore',invalid='ignore'):
                boot[name] = 100 * (totals[:,2]/totals[:,0] + totals[:,3]/totals[:,1])
            row = profiles[(profiles.season.eq(season)) & profiles.team.eq(team) & profiles.profile.eq(name)]
            if row.empty:
                continue
            idx = row.index[0]
            if not np.isfinite(profiles.loc[idx,'relative_net']):
                continue
            point[name] = profiles.loc[idx,'relative_net']
            finite = boot[name][np.isfinite(boot[name])]
            if len(finite) >= draws / 2:
                profiles.loc[idx,['ci_low','ci_high']] = np.quantile(finite,[.025,.975])
        for name,lead,pressure in [
            ('Ahead vs behind', 'Ahead 11+', 'Behind 11+'),
            ('Ahead vs clutch', 'Ahead 11+', 'Clutch (last 5, within 5)'),
            ('Ahead vs behind, excluding final 2', 'Ahead 11+, excluding final 2', 'Behind 11+, excluding final 2')]:
            if lead not in point or pressure not in point:
                continue
            low,high = np.nanquantile(boot[lead]-boot[pressure],[.025,.975])
            pr = profiles[profiles.season.eq(season)&profiles.team.eq(team)&profiles.profile.isin([lead,pressure])]
            contrasts.append({'season':season, 'team':team,'contrast':name,
                'gap':point[lead]-point[pressure], 'ci_low':low,'ci_high':high,
                'eligible':bool(pr.eligible.all()),
                'positive_ci':bool(low>0),
                'schedule_adjusted_gap':float(pr.set_index('profile').loc[lead,'schedule_adjusted_relative_net'] -
                                             pr.set_index('profile').loc[pressure,'schedule_adjusted_relative_net'])})
    overall = profiles[profiles.profile.eq('Overall')].set_index(['season','team']).relative_net
    profiles['lift_vs_usual'] = profiles.relative_net - pd.MultiIndex.from_frame(profiles[['season','team']]).map(overall).to_numpy()
    return cells, profiles, pd.DataFrame(contrasts)
