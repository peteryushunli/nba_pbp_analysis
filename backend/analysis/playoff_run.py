"""python -m backend.analysis.playoff_run --download --seasons 2016 ... 2026"""
import argparse
import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
from backend.analysis.download import download_archives, download_archive
from backend.analysis.possessions import load_season
from backend.analysis.situational import team_perspectives, add_expectations, profile_masks, rate_summary
from backend.analysis.playoff_data import load_games, strength, playoff_series
from backend.analysis.playoff_models import (SIGNALS,MODELS,chronological,evaluate,team_outcomes,evaluate_brackets,losses)

ROOT=Path(__file__).resolve().parents[2]


def season_profiles(possessions: pd.DataFrame) -> pd.DataFrame:
    v=add_expectations(team_perspectives(possessions))
    profiles=pd.concat([rate_summary(v[mask],['season','team']).assign(profile=name)
                        for name,mask in profile_masks(v).items()],ignore_index=True)
    profiles['shrunk_schedule_relative_net']=sum(100*profiles[f'{side}_schedule_resid']/(profiles[f'{side}_n']+200)
                                                for side in ['off','def'])
    return profiles


def team_features(stats: pd.DataFrame, profiles: pd.DataFrame, seeds: pd.DataFrame,
                  metric: str = 'shrunk_relative_net') -> pd.DataFrame:
    p=profiles.pivot(index='team',columns='profile',values=metric)
    overall=profiles[profiles.profile.eq('Overall')].set_index('team')
    result=stats.merge(seeds[['team','seed']],on='team',how='left')
    result['possession_games']=result.team.map(overall.games)
    result['possession_game_coverage']=result.possession_games/result.games
    result['overall_relative_net']=result.team.map(overall.relative_net)
    states={'big_deficit_lift':'Behind 11+, excluding final 2','close_lift':'Close (within 5)',
            'clutch_lift':'Clutch (last 5, within 5)','late_trailing_lift':'Trailing 1–10 in Q4/OT'}
    for signal,state in states.items():
        result[signal]=result.team.map(p[state]-p['Overall']).fillna(0)
        exposure=profiles[profiles.profile.eq(state)].set_index('team')
        for side in ['off','def']:
            result[signal+'_'+side+'_n']=result.team.map(exposure[side+'_n']).fillna(0)
    result['front_running_gap']=result.team.map(p['Ahead 11+, excluding final 2']-p['Behind 11+, excluding final 2']).fillna(0)
    for state,key in [('Ahead 11+, excluding final 2','lead'),('Behind 11+, excluding final 2','deficit')]:
        exposure=profiles[profiles.profile.eq(state)].set_index('team')
        for side in ['off','def']:
            result['front_running_gap_'+key+'_'+side+'_n']=result.team.map(exposure[side+'_n']).fillna(0)
    if not np.isfinite(result[SIGNALS].to_numpy()).all():
        raise ValueError('Nonfinite situation features')
    return result


def table(frame: pd.DataFrame, fields: list[str], precision: int = 3) -> str:
    d=frame[fields].copy()
    for key in fields:
        if pd.api.types.is_float_dtype(d[key]):
            d[key]=d[key].map(lambda x:f'{x:.{precision}f}' if np.isfinite(x) else '—')
    return '\n'.join(['| '+' | '.join(map(str,fields))+' |','| '+' | '.join(['---']*len(fields))+' |']+
                     ['| '+' | '.join(map(str,r))+' |' for r in d.to_numpy()])


def plot_results(summary: pd.DataFrame, out: Path) -> None:
    cache=ROOT/'data'/'processed'/'plot-cache'
    cache.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(cache))
    os.environ.setdefault('XDG_CACHE_HOME',str(cache))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    s=summary[summary['sample'].eq('Historical test through 2025') &
              ~summary.model.isin(['Strength baseline','Seed + home court'])].copy()
    labels=s.model.str.replace('Baseline + ','',regex=False).str.replace('_',' ').to_list()
    fig,ax=plt.subplots(figsize=(10.5,5.2),layout='constrained')
    x=s.log_loss_change.to_numpy();y=np.arange(len(s))
    ax.errorbar(x,y,xerr=np.vstack([x-s.change_ci_low.to_numpy(),s.change_ci_high.to_numpy()-x]),
        fmt='o',color='#285b83',ecolor='#718391',capsize=4)
    ax.axvline(0,color='#666666',linewidth=1)
    ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlabel('Change in held-out log loss vs strength baseline · negative = better')
    ax.set_title('Do regular-season game-state profiles add playoff predictive value?',loc='left',pad=15)
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.15)
    fig.text(.01,-.025,'2020–2025 chronological tests; 95% paired intervals resample postseason years. Exploratory, no multiple-testing correction.',fontsize=9)
    fig.savefig(out/'playoff_predictive_value.png',dpi=170,bbox_inches='tight')
    svg=out/'playoff_predictive_value.svg'
    fig.savefig(svg,bbox_inches='tight');plt.close(fig)
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')


def write_report(summary,yearly,coefficients,outcomes,bracket_summary,repeatability,coverage,audits,excluded,sensitivity,out):
    historical=summary[summary['sample'].eq('Historical test through 2025')]
    final=summary[summary['sample'].eq('2026 final holdout')]
    baseline=historical[historical.model.eq('Strength baseline')].iloc[0]
    sig=historical[historical.model.str.startswith('Baseline + ')&~historical.model.eq('Baseline + recent form')]
    latest=outcomes[outcomes.season.eq(outcomes.season.max())]
    nyk=outcomes[outcomes.season.eq(2026)&outcomes.team.eq('NYK')]
    nyk_note=('' if nyk.empty else f" The joint model lowered the eventual champion New York's expected run from "
              f"{nyk.iloc[0].expected_series_wins:.2f} to {nyk.iloc[0].joint_expected_series_wins:.2f} series wins.")
    examples=outcomes[outcomes.team.isin(['MIA','DAL','IND','NYK','CLE','OKC','SAS','HOU','BOS']) &
                      ((outcomes.season.eq(2023)&outcomes.team.eq('MIA'))|
                       (outcomes.season.eq(2024)&outcomes.team.eq('DAL'))|
                       (outcomes.season.eq(2025)&outcomes.team.eq('IND'))|outcomes.season.eq(2026))]
    strong=sig[sig.change_ci_high.lt(0)]
    conclusion=(f'{len(strong)} situation models improve historical log loss with paired year-bootstrap intervals entirely below zero.'
                if len(strong) else 'No situation model improves historical log loss with a paired year-bootstrap interval entirely below zero.')
    if sig.log_loss_change.ge(0).all():
        conclusion+=' Every individual signal and the joint model has a worse historical mean log loss than the strength baseline.'
    fields=['model','series','log_loss','brier','accuracy','log_loss_change','change_ci_low','change_ci_high']
    report=f'''# Do game-state profiles predict playoff overperformance?

## Answer

{conclusion} This is a retrospective chronological backtest, not a preregistered or live betting result. Do not equate a season-specific profile with a durable playoff trait.

The sample covers 2015–16 through 2025–26: eleven completed postseasons, 165 series and 176 playoff team-seasons before possession-coverage filtering. Train first on 2016–2019, test 2020, then expand the training window one year at a time. The historical comparison ends at 2025; 2026 is reported separately as the final chronological holdout. All regular-season inputs are frozen before that postseason, including actual playoff seeds after the play-in. We do not predict play-in qualification.

The strength baseline's historical test accuracy is {baseline.accuracy:.1%}, Brier score {baseline.brier:.3f} and log loss {baseline.log_loss:.3f}. The main question is whether a situation feature improves those probabilities on the **same held-out matchups**, rather than whether it correlates with playoff wins in the same sample.

## Historical predictive comparison

Lower log loss and Brier score are better. Negative loss change favors adding the feature. Accuracy alone ignores confidence and can obscure worse probability estimates.

{table(historical,fields)}

![Incremental playoff predictive value](playoff_predictive_value.png)

## Final 2026 holdout

{table(final,fields)}

One postseason has only fifteen series. A favorable 2026 result alone cannot establish reliability, particularly because these questions were motivated by inspecting 2026 regular-season profiles.

## What overachieving means

Start with the actual first-round bracket and predict every possible later matchup from **regular-season** team strength, seed and home court. Integrate the bracket exactly to calculate each entrant's expected number of series wins, conference-finals/Finals probabilities and title probability. Never plug in its eventual later opponents when calculating preseason expectations. Expected series wins sum to fifteen and title probabilities sum to one.

Overperformance = actual series wins − strength-baseline expected series wins. A positive residual means a deeper run than this model expected from that team's quality and draw. This is model-relative, not a verdict about coaching or mentality. It also does not represent market expectations or betting odds. Later-round series-level predictions are conditional on the actual matchup; the bracket expectation is the separate pre-first-round measure.

{table(examples.sort_values(['season','overperformance'],ascending=[True,False]),['season','team','seed','wins','games','srs','actual_series_wins','expected_series_wins','overperformance','title_probability'],2)}

All sixteen 2026 entrants:

{table(latest.sort_values('overperformance',ascending=False),['team','seed','win_pct','point_diff','srs','actual_series_wins','expected_series_wins','overperformance','joint_expected_series_wins'],2)}

### Direct test of predicting the depth of a playoff run

Compare each team's actual series wins with its pre-first-round bracket expectation. Both models use the same complete brackets; a year with an entrant below the situation coverage floor is excluded from both sides of this comparison. These team outcomes are dependent, so uncertainty again resamples whole postseason years. RMSE and MAE are in series wins; negative MSE change favors adding all situation features.

{table(bracket_summary,list(bracket_summary.columns))}

This tests whether the profiles identify overachievers before their run, rather than fitting a story to the eventual winners. The historical joint model slightly improves absolute error but slightly worsens squared error; the paired interval includes zero. That is mixed evidence, not a reliable improvement.

### Reading the team examples

Miami 2023, Dallas 2024 and Indiana 2025 all had positive clutch lifts and deep runs beyond baseline expectation. However, those selected examples do not validate the rule: Milwaukee 2021 won the championship despite a negative clutch lift, and Minnesota 2025 reached the conference finals with a strongly negative clutch lift. Dallas 2024 also had a positive frontrunning gap, showing that a relatively better leading-state profile does not preclude a successful run. In 2026 Cleveland and San Antonio's behind-state strength accompanied overperformance, but Denver's did not.{nyk_note} These are diagnostic counterexamples, not causal explanations.

## Controls and game-state features

- The baseline compares **both teams**: regular-season win percentage, average point differential per game, schedule-adjusted point differential (SRS), actual playoff seed and series home-court advantage. SRS solves point margin = own rating − opponent rating, with league mean zero. It adjusts for the strength of the full regular-season schedule rather than treating identical records as identical teams.
- Home court is zero for the 2020 bubble. In bracket integration it belongs to the higher seed within a conference; Finals home court uses regular-season win percentage, wins, then SRS as an explicit proxy for rare tied-record tiebreaks. Actual series evaluation uses the scheduled Game 1 home team. Seed comparisons in the Finals retain each team's conference seed.
- Situation metrics use signed **possession-start** margin and remaining clock. Benchmarks exclude the focal team and match season, margin and time. Positive offense/defense residuals always mean better. No playoff possessions enter these features.
- Shrink each offensive and defensive residual with 200 neutral possessions. State lifts subtract the team's shrunk overall league-relative net. This emphasizes performance above its usual level and reduces noisy small-state extremes. Missing state exposure is assigned a neutral zero lift. Individual-state display floors are not a predictive inclusion rule; low-count states remain heavily shrunk and their counts are exported.
- `front_running_gap`: shrunk relative net ahead 11+ minus behind 11+, excluding the final two regulation minutes. Positive means relatively stronger while ahead. The other signals are big-deficit lift (also excluding final two), close-game lift (within five), clutch lift (final five minutes or OT within five), and late-trailing lift (down 1–10 in Q4/OT).
- Test each of the five signals separately and all together. Fit symmetric logistic models with **no intercept**, training-only RMS scaling, and fixed L2 penalty 2. Swapping opponents complements the prediction. Wins, margin and SRS are correlated; regularization limits unstable coefficients rather than treating each as independent evidence. `Baseline + recent form` adds last-twenty-game point differential as a non-situation comparison.
- Situation models and the matched baseline use identical training/evaluation series. Exclude a matchup if either team has possession coverage below 80% of its accepted regular-game total. The strength-only bracket expectation can still cover all sixteen entrants; its joint-model counterpart is omitted for years with an entrant below the coverage floor. Actual playoff wins always come from the complete bracket, including any series excluded from prediction evaluation.

## Persistence and robustness

Year-to-year correlations compare the same franchise's successive regular-season signals, not roster continuity. Weak persistence can coexist with predictive value, but undermines calling a profile a lasting identity.

{table(repeatability,['signal','adjacent_team_seasons','pearson_correlation'],3)}

Sensitivity checks include L2 penalties 0.5 and 8, excluding 2020, excluding the 2026 provider change, first rounds alone, only teams with ≥90% possession-game coverage, and opponent-adjusted state profiles. That last check shifts each offensive/defensive possession's expected points by its opponent's regular-season defensive/offensive deviation from league average, then applies the same shrinkage and state lifts. This addresses easier opponents within a state as a limited additive sensitivity; it does not estimate opponent-by-state or lineup interactions. Negative loss changes favor adding all situations.

{table(sensitivity,['check','series','baseline_log_loss','joint_log_loss','joint_change'],3)}

Annual held-out performance and standardized coefficients are exported. Confidence intervals resample **whole postseason years**, pairing baseline and augmented losses for the same series (5,000 draws, seed 19). This accounts for within-postseason dependence but only six historical test years remain; intervals are coarse and do not include every source of training/model uncertainty. Five individual signals, a joint model and multiple sensitivities are exploratory comparisons, without multiplicity correction. Selecting whichever model looks best after this test would require a new validation sample.

## Data coverage and audit

{table(coverage,list(coverage.columns),2)}

{len(excluded)} series are excluded from the paired prediction dataset by the 80% team coverage rule. Exact IDs and reasons are in `playoff_prediction_audit.json`. Data gaps can be selective, not random. Legacy normalization first preserves fully score-reconciled exact FT matches; fallback matching removes historical suffix/initial/diacritic and FT-sequence annotation inconsistencies, verifies shooter/team/period/clock and made counts, and admits a recovered game only after per-team and final-score reconciliation. These inputs have a separate canonical cache. The four-season app keeps its stricter exact-description FT matching; its reports were also refreshed for the corrected final-score ordering.

Regular standings and playoff results use completed NBA cumulative game scores with independently identified home/away teams. Exact duplicated event rows are removed. Historical corrected events can have late event IDs while referring to an earlier period; cumulative scores are therefore ordered by period and remaining clock before event ID. Conflicting event-point tallies are flagged separately and do not override an unambiguous completed scoreboard. Any game with a tied or ambiguous final score is excluded. All source regular games are accepted in this run, including all 1,230 games in 2026. Baseline rates use win percentage, so shortened seasons are comparable.

Sources: [public NBA event/possession archives](https://github.com/shufinskiy/nba_data), with source hashes and exclusions in the audit. The bulk 2026 playoff archive contains only 60 games. Supplement its 25 missing later-round games from the [NBA's completed 2026 schedule](https://www.nba.com/news/2026-nba-playoffs-schedule), stored as auditable factual metadata in `backend/analysis/metadata/playoffs_2026.json`. Every overlapping later-round game agrees in teams and score. Validate all eleven postseasons as complete 8–4–2–1 series brackets, with exactly four wins for every series winner. Seeds come from first-round bracket slots and Game 1 home court, not regular-season win sorting or eventual advancement.

## Basketball interpretation

This controls seed, broad opponent quality and draw, but does not encode who was healthy, trade-deadline rotation changes, which regular-season minutes involved the playoff rotation, or matchup-specific shot/turnover/rebounding weaknesses. Recent form is a limited sensitivity, not an injury model. A postseason upset can reflect availability or matchup fit without validating a clutch or frontrunner trait. The model excludes betting odds, which could reflect those facts better than these inputs.

Winning efficiently while down big is not the same as winning a playoff game or making a comeback. Reserves, tactical responses, intentional fouling and state selection can affect the metrics. Compare a situational signal with a strong baseline first, require repeatable held-out improvement, and then inspect the basketball mechanism in individual series.

## Reproduce

```sh
.venv/bin/pip install -e '.[dev]'
.venv/bin/python -m backend.analysis.playoff_run --download --seasons 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025 2026
.venv/bin/python -m pytest -q
```

Outputs: team features/counts, complete playoff series, paired series features, chronological predictions, annual losses, coefficients, model summaries, bracket expectations/residuals, persistence, sensitivity results and full audits. See `playoff_*.csv` and `playoff_prediction_audit.json`. Figures are exported as PNG and SVG.
'''
    (out/'playoff_prediction.md').write_text(report)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seasons',type=int,nargs='+',default=list(range(2016,2027)))
    parser.add_argument('--download',action='store_true')
    parser.add_argument('--data-dir',type=Path,default=ROOT/'data')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'reports')
    args=parser.parse_args();seasons=sorted(set(args.seasons));out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    if args.download:
        download_archives(seasons,args.data_dir)
        for season in seasons:
            download_archive('cdnnba' if season>=2026 else 'nbastats',season,args.data_dir,postseason=True)
    stats_frames,profile_frames,series_frames,coverage,audits=[],[],[],[],{}
    for season in seasons:
        regular,regular_audit=load_games(season,args.data_dir)
        post,post_audit=load_games(season,args.data_dir,True)
        series,seeds=playoff_series(post)
        possessions,possession_audit=load_season(season,args.data_dir,robust_ft=True)
        profiles=season_profiles(possessions)
        ratings=strength(regular)
        stats=team_features(ratings,profiles,seeds)
        adjusted=team_features(ratings,profiles,seeds,metric='shrunk_schedule_relative_net').set_index('team')
        for signal in SIGNALS:
            stats['opponent_adjusted_'+signal]=stats.team.map(adjusted[signal])
        stats_frames.append(stats);profile_frames.append(profiles);series_frames.append(series)
        coverage.append(dict(season=season,regular_games=len(regular),possession_games=possessions.game_id.nunique(),
            postseason_games=len(post),series=len(series),minimum_team_coverage=stats.possession_game_coverage.min()))
        audits[season]={'regular_games':regular_audit,'postseason_games':post_audit,'possessions':possession_audit}
        print(f'{season}: {len(regular)} regular games, {len(series)} series, {possession_audit["games_analyzed"]} possession games',flush=True)
    teams=pd.concat(stats_frames,ignore_index=True);all_series=pd.concat(series_frames,ignore_index=True)
    from backend.analysis.playoff_models import join_series
    paired,excluded=join_series(all_series,teams)
    predictions,coefficients=chronological(paired);summary,yearly=evaluate(predictions)
    outcomes=team_outcomes(paired,teams,all_series=all_series)
    bracket_summary=evaluate_brackets(outcomes)
    repeated=[]
    for signal in SIGNALS:
        previous=teams[['season','team',signal,'possession_game_coverage']].copy();previous['season']+=1
        joined=teams.merge(previous,on=['season','team'],suffixes=('','_prior'))
        joined=joined[joined.possession_game_coverage.ge(.8)&joined.possession_game_coverage_prior.ge(.8)]
        repeated.append(dict(signal=signal,adjacent_team_seasons=len(joined),pearson_correlation=joined[signal].corr(joined[signal+'_prior'])))
    sensitivity=[]
    adjusted_teams=teams.copy()
    for signal in SIGNALS:
        adjusted_teams[signal]=adjusted_teams['opponent_adjusted_'+signal]
    for label,frame,penalty in [('L2 0.5',paired,.5),('L2 8',paired,8),('Exclude 2020',paired[paired.season.ne(2020)],2),
            ('Exclude 2026',paired[paired.season.ne(2026)],2),('First round only',paired[paired['round'].eq(1)],2),
            ('Coverage ≥90%',join_series(all_series,teams,.9)[0],2),
            ('Opponent-adjusted states',join_series(all_series,adjusted_teams)[0],2)]:
        pred,_=chronological(frame,penalty=penalty)
        base=pred[pred.model.eq('Strength baseline')];joint=pred[pred.model.eq('Baseline + all situations')]
        bl=losses(base.outcome.to_numpy(),base.probability_a.to_numpy())['log_loss']
        jl=losses(joint.outcome.to_numpy(),joint.probability_a.to_numpy())['log_loss']
        sensitivity.append(dict(check=label,series=len(base),baseline_log_loss=bl,joint_log_loss=jl,joint_change=jl-bl))
    repeatability=pd.DataFrame(repeated);coverage=pd.DataFrame(coverage);sensitivity=pd.DataFrame(sensitivity)
    for name,frame in [('teams',teams),('profiles',pd.concat(profile_frames)),('series',all_series),('paired_features',paired),
        ('predictions',predictions),('coefficients',coefficients),('model_summary',summary),('annual_performance',yearly),
        ('overperformance',outcomes),('bracket_summary',bracket_summary),('repeatability',repeatability),('sensitivity',sensitivity)]:
        frame.to_csv(out/f'playoff_{name}.csv',index=False)
    audit={'seasons':audits,'excluded_series':excluded,'configuration':{'first_test_season':2020,'historical_test_end':2025,
        'final_holdout':2026,'minimum_team_coverage':.8,'shrinkage_possessions':200,'l2_penalty':2,
        'season_bootstrap_draws':5000,'season_bootstrap_seed':19,'models':MODELS}}
    (out/'playoff_prediction_audit.json').write_text(json.dumps(audit,indent=2))
    plot_results(summary,out)
    write_report(summary,yearly,coefficients,outcomes,bracket_summary,repeatability,coverage,audits,excluded,sensitivity,out)
    print(summary[summary['sample'].eq('Historical test through 2025')][['model','series','log_loss','log_loss_change','change_ci_low','change_ci_high']].round(4).to_string(index=False))
    print(f'Playoff analysis written to {out}',flush=True)


if __name__=='__main__':main()
