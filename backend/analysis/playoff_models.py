"""Chronological, opponent-relative playoff prediction and bracket expectations."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

SEED = ['home_advantage','seed_advantage']
BASE = SEED + ['win_pct_diff','point_diff_diff','srs_diff']
SIGNALS = ['front_running_gap','big_deficit_lift','close_lift','clutch_lift','late_trailing_lift']
MODELS = {'Seed + home court':SEED,'Strength baseline':BASE,
          **{f'Baseline + {s}':BASE+[s+'_diff'] for s in SIGNALS},
          'Baseline + all situations':BASE+[s+'_diff' for s in SIGNALS],
          'Baseline + recent form':BASE+['last20_point_diff_diff']}


@dataclass
class Logistic:
    fields: list[str]
    scale: np.ndarray
    coef: np.ndarray

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return expit(frame[self.fields].to_numpy(dtype=float)/self.scale @ self.coef)


def fit(frame: pd.DataFrame, fields: list[str], penalty: float = 2) -> Logistic:
    """No intercept or centering: swapping teams must complement the probability.

    Training-only RMS standardization and a fixed L2 penalty avoid leakage and
    reduce instability from correlated wins, margin, SRS and situation features.
    """
    x=frame[fields].to_numpy(dtype=float);y=frame.outcome.to_numpy(dtype=float)
    if not np.isfinite(x).all() or not set(y).issubset({0,1}):
        raise ValueError('Invalid model inputs')
    scale=np.sqrt((x*x).mean(axis=0));scale[scale<1e-8]=1
    z=x/scale
    def objective(b):
        logits=z@b
        loss=np.sum(np.logaddexp(0,logits)-y*logits)+penalty*np.sum(b*b)/2
        gradient=z.T@(expit(logits)-y)+penalty*b
        return loss,gradient
    result=minimize(objective,np.zeros(len(fields)),jac=True,method='L-BFGS-B')
    if not result.success:
        raise RuntimeError(f'Logistic fit failed: {result.message}')
    return Logistic(fields,scale,result.x)


def pair_features(a: pd.Series, b: pd.Series, season: int, home_court: str) -> dict:
    row={'home_advantage':0 if season==2020 else (1 if home_court==a.team else -1),
         'seed_advantage':float(b.seed-a.seed)}
    for field in ['win_pct','point_diff','srs','last20_point_diff',*SIGNALS]:
        row[field+'_diff']=float(a[field]-b[field])
    return row


def join_series(series: pd.DataFrame, teams: pd.DataFrame, minimum_coverage: float = .8) -> tuple[pd.DataFrame, list[dict]]:
    lookup=teams.set_index(['season','team'],drop=False);rows,excluded=[],[]
    for _,r in series.iterrows():
        a=lookup.loc[(r.season,r.team_a)];b=lookup.loc[(r.season,r.team_b)]
        if min(a.possession_game_coverage,b.possession_game_coverage)<minimum_coverage:
            excluded.append({'season':int(r.season),'series_id':r.series_id,
                'teams':[r.team_a,r.team_b],'reason':f'At least one team has <{minimum_coverage:.0%} regular-game possession coverage'})
            continue
        rows.append({**r.to_dict(),**pair_features(a,b,int(r.season),r.home_court)})
    return pd.DataFrame(rows),excluded


def losses(y: np.ndarray, p: np.ndarray) -> dict:
    p=np.clip(p,1e-8,1-1e-8)
    return {'log_loss':float(np.mean(-y*np.log(p)-(1-y)*np.log(1-p))),
            'brier':float(np.mean((p-y)**2)), 'accuracy':float(np.mean((p>=.5)==y))}


def chronological(series: pd.DataFrame, first_test: int = 2020, penalty: float = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions,coefficients=[],[]
    for season in sorted(series.season.unique()):
        if season<first_test:
            continue
        train=series[series.season.lt(season)];test=series[series.season.eq(season)]
        if train.empty or test.empty:
            raise ValueError('Each fold needs earlier training seasons and test series')
        for name,fields in MODELS.items():
            model=fit(train,fields,penalty)
            for field,coefficient,scale in zip(fields,model.coef,model.scale):
                coefficients.append(dict(season=int(season),model=name,feature=field,
                    coefficient=float(coefficient),scale=float(scale),train_series=len(train),
                    training_last_season=int(train.season.max())))
            result=test[['season','series_id','round','team_a','team_b','winner','outcome']].copy()
            result['model']=name;result['probability_a']=model.predict(test)
            result['train_series']=len(train);result['training_last_season']=int(train.season.max())
            predictions.append(result)
    return pd.concat(predictions,ignore_index=True),pd.DataFrame(coefficients)


def evaluate(predictions: pd.DataFrame, draws: int = 5000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Paired loss differences; resample whole postseason years, not series rows."""
    yearly=[]
    for (season,model),g in predictions.groupby(['season','model']):
        yearly.append(dict(season=int(season),model=model,series=len(g),
                           **losses(g.outcome.to_numpy(),g.probability_a.to_numpy())))
    yearly=pd.DataFrame(yearly);summaries=[]
    for label,subset in [('Historical test through 2025',predictions[predictions.season.le(2025)]),
                         ('2026 final holdout',predictions[predictions.season.eq(2026)]),
                         ('All chronological test years',predictions)]:
        baseline=subset[subset.model.eq('Strength baseline')].set_index('series_id')
        for model,g in subset.groupby('model'):
            stats=losses(g.outcome.to_numpy(),g.probability_a.to_numpy())
            y=g.outcome.to_numpy();p=np.clip(g.probability_a.to_numpy(),1e-8,1-1e-8)
            bp=baseline.loc[g.series_id].probability_a.to_numpy()
            diff=(-y*np.log(p)-(1-y)*np.log(1-p))-(-y*np.log(bp)-(1-y)*np.log(1-bp))
            annual=pd.DataFrame({'season':g.season.to_numpy(),'difference':diff}).groupby('season').difference.agg(['sum','count'])
            if len(annual)>1:
                rng=np.random.default_rng(19)
                indices=rng.integers(0,len(annual),size=(draws,len(annual)))
                boot=annual['sum'].to_numpy()[indices].sum(axis=1)/annual['count'].to_numpy()[indices].sum(axis=1)
                low,high=np.quantile(boot,[.025,.975])
            else:
                low=high=np.nan
            summaries.append(dict(sample=label,model=model,series=len(g),seasons=g.season.nunique(),
                **stats,log_loss_change=float(diff.mean()),change_ci_low=low,change_ci_high=high))
    return pd.DataFrame(summaries),yearly


def bracket_expectations(teams: pd.DataFrame, model: Logistic, season: int) -> pd.DataFrame:
    """Exact pre-playoff bracket integration; no realized later opponents as inputs.

    Conference series use actual playoff seeds for home court. Finals home court
    uses regular win percentage, then regular wins, then SRS as an explicit proxy
    for rare tied-record tiebreaks. The 2020 bubble has zero home advantage.
    """
    if len(teams)!=16:
        raise ValueError('Bracket requires all sixteen playoff entrants')
    lookup=teams.set_index('team',drop=False);expectations={t:np.zeros(4) for t in lookup.index}
    seed_teams={(r.conference,int(r.seed)):r.team for _,r in teams.iterrows()}
    probabilities={}
    for a in lookup.index:
        for b in lookup.index:
            if a==b:
                continue
            ta,tb=lookup.loc[a],lookup.loc[b]
            if ta.conference==tb.conference:
                home=a if ta.seed<tb.seed else b
            else:
                home=max([a,b],key=lambda t:(lookup.loc[t].win_pct,lookup.loc[t].wins,lookup.loc[t].srs))
            probabilities[a,b]=model.predict(pd.DataFrame([pair_features(ta,tb,season,home)]))[0]
    def merge(left,right,round_):
        result={}
        for group,other in [(left,right),(right,left)]:
            for team,prob in group.items():
                result[team]=prob*sum(weight*probabilities[team,opponent] for opponent,weight in other.items())
                expectations[team][round_]=result[team]
        return result
    conference_winners=[]
    for conference in ['E','W']:
        nodes=[{seed_teams[conference,seed]:1.0} for seed in [1,8,4,5,2,7,3,6]]
        for round_ in range(3):
            nodes=[merge(nodes[i],nodes[i+1],round_) for i in range(0,len(nodes),2)]
        conference_winners.append(nodes[0])
    merge(*conference_winners,3)
    result=pd.DataFrame([dict(team=t,expected_series_wins=float(p.sum()),
        second_round_probability=p[0],conference_finals_probability=p[1],
        finals_probability=p[2],title_probability=p[3]) for t,p in expectations.items()])
    if not np.isclose(result.expected_series_wins.sum(),15) or not np.isclose(result.title_probability.sum(),1):
        raise ValueError('Bracket probabilities do not conserve total wins')
    return result


def team_outcomes(series: pd.DataFrame, teams: pd.DataFrame, first_test: int = 2020, penalty: float = 2,
                  all_series: pd.DataFrame | None = None) -> pd.DataFrame:
    rows=[]
    for season in sorted(teams.season.unique()):
        if season<first_test:
            continue
        train=series[series.season.lt(season)]
        entrants=teams[teams.season.eq(season)&teams.seed.notna()].copy()
        # The strength-only bracket can include entrants whose situation data
        # are thin; the joint model needs the same coverage floor as evaluation.
        baseline=fit(train,BASE,penalty)
        expected=bracket_expectations(entrants,baseline,int(season)).set_index('team')
        actual_source=series if all_series is None else all_series
        actual=actual_source[actual_source.season.eq(season)].winner.value_counts()
        joint=None
        if entrants.possession_game_coverage.ge(.8).all():
            joint=bracket_expectations(entrants,fit(train,MODELS['Baseline + all situations'],penalty),int(season)).set_index('team')
        for _,r in entrants.iterrows():
            e=expected.loc[r.team]
            rows.append({**r.to_dict(),**e.to_dict(),'actual_series_wins':int(actual.get(r.team,0)),
                'overperformance':float(actual.get(r.team,0)-e.expected_series_wins),
                'joint_expected_series_wins':np.nan if joint is None else joint.loc[r.team,'expected_series_wins']})
    return pd.DataFrame(rows)


def evaluate_brackets(outcomes: pd.DataFrame, draws: int = 5000) -> pd.DataFrame:
    """Compare pre-first-round run predictions on identical complete brackets.

    Whole-year resampling keeps the sixteen dependent team outcomes together.
    Years missing any entrant's situation data are excluded from both models.
    """
    rows=[]
    for sample,subset in [('Historical test through 2025',outcomes[outcomes.season.le(2025)]),
                         ('2026 final holdout',outcomes[outcomes.season.eq(2026)]),
                         ('All chronological test years',outcomes)]:
        matched=subset[subset.joint_expected_series_wins.notna()].copy()
        if matched.empty:
            continue
        baseline=matched.expected_series_wins-matched.actual_series_wins
        joint=matched.joint_expected_series_wins-matched.actual_series_wins
        annual=pd.DataFrame({'season':matched.season,'difference':joint**2-baseline**2}).groupby('season').difference.mean()
        low=high=np.nan
        if len(annual)>1:
            rng=np.random.default_rng(19)
            indices=rng.integers(0,len(annual),size=(draws,len(annual)))
            low,high=np.quantile(annual.to_numpy()[indices].mean(axis=1),[.025,.975])
        rows.append(dict(sample=sample,team_seasons=len(matched),seasons=matched.season.nunique(),
            baseline_rmse=float(np.sqrt(np.mean(baseline**2))),joint_rmse=float(np.sqrt(np.mean(joint**2))),
            baseline_mae=float(baseline.abs().mean()),joint_mae=float(joint.abs().mean()),
            mse_change=float(np.mean(joint**2-baseline**2)),change_ci_low=low,change_ci_high=high))
    return pd.DataFrame(rows)
