"""Dean Oliver's individual Offensive and Defensive Rating calculations."""

from dataclasses import dataclass


@dataclass
class PlayerBoxStats:
    """Situational box score aggregates for one player."""
    player_id: int
    player_name: str
    team: str
    fgm: int
    fga: int
    fg3m: int
    fg3a: int
    ftm: int
    fta: int
    oreb: int
    dreb: int
    ast: int
    tov: int
    stl: int
    blk: int
    pf: int
    pts: int
    mp: float       # estimated minutes
    games: int


@dataclass
class TeamBoxStats:
    """Situational aggregates for one team."""
    team: str
    fgm: int
    fga: int
    fg3m: int
    ftm: int
    fta: int
    oreb: int
    dreb: int
    ast: int
    tov: int
    stl: int
    blk: int
    pf: int
    pts: int
    mp: float


@dataclass
class OpponentBoxStats:
    """Aggregated opponent stats for a team."""
    fgm: int
    fga: int
    fg3m: int
    ftm: int
    fta: int
    oreb: int
    dreb: int
    tov: int
    pts: int


@dataclass
class LeagueStats:
    fga: int
    fgm: int
    fg3m: int
    fta: int
    ftm: int
    oreb: int
    dreb: int
    reb: int  # oreb + dreb
    pts: int
    tov: int


def calculate_ortg(
    player: PlayerBoxStats,
    team: TeamBoxStats,
    league: LeagueStats,
) -> float | None:
    """
    Calculate individual Offensive Rating using Dean Oliver's method.

    Returns points produced per 100 possessions, or None if insufficient data.

    Reference: https://www.basketball-reference.com/about/ratings.html
    """
    if player.fga == 0 or team.fga == 0 or player.mp == 0 or team.mp == 0:
        return None

    # --- qAST: fraction of FGM that were assisted while player is on floor ---
    mp_fraction = player.mp / (team.mp / 5)
    team_fgm_ex = team.fgm - player.fgm

    if team_fgm_ex <= 0 or team.fgm == 0:
        qAST = 0.0
    else:
        part1 = mp_fraction * (1.14 * ((team.ast - player.ast) / team.fgm))
        team_ast_rate = team.ast / team.mp
        team_fgm_rate = team.fgm / team.mp
        player_mp5 = player.mp * 5
        denom = team_fgm_rate * player_mp5 - player.fgm
        if denom <= 0:
            part2 = 0.0
        else:
            part2 = ((team_ast_rate * player_mp5 - player.ast) / denom) * (1 - mp_fraction)
        qAST = part1 + part2

    # --- FG Part ---
    pts_minus_ft = player.pts - player.ftm
    fg_part = player.fgm * (1 - 0.5 * (pts_minus_ft / (2 * player.fga)) * qAST)

    # --- AST Part ---
    team_pts_minus_ft = team.pts - team.ftm
    team_fga_ex = team.fga - player.fga
    if team_fga_ex > 0:
        ast_part = 0.5 * (((team_pts_minus_ft) - pts_minus_ft) / (2 * team_fga_ex)) * player.ast
    else:
        ast_part = 0.0

    # --- FT Part ---
    if player.fta > 0:
        ft_miss_rate = 1 - (player.ftm / player.fta)
        ft_part = (1 - ft_miss_rate ** 2) * 0.4 * player.fta
    else:
        ft_miss_rate = 0.0
        ft_part = 0.0

    # --- Team Scoring Possessions ---
    team_ft_miss_rate = (1 - (team.ftm / team.fta)) if team.fta > 0 else 0.0
    team_scoring_poss = team.fgm + (1 - team_ft_miss_rate ** 2) * team.fta * 0.4
    if team_scoring_poss == 0:
        return None

    # --- Team ORB% ---
    # Use league DRB as proxy for opponent DRB
    opp_drb = league.dreb
    team_orb_pct = team.oreb / (team.oreb + opp_drb) if (team.oreb + opp_drb) > 0 else 0.0

    # --- Team Play% ---
    team_total_poss = team.fga + team.fta * 0.4 + team.tov
    team_play_pct = team_scoring_poss / team_total_poss if team_total_poss > 0 else 0.0

    # --- Team ORB Weight ---
    denom = (1 - team_orb_pct) * team_play_pct + team_orb_pct * (1 - team_play_pct)
    team_orb_weight = ((1 - team_orb_pct) * team_play_pct) / denom if denom > 0 else 0.0

    # --- ORB Part ---
    orb_part = player.oreb * team_orb_weight * team_play_pct

    # --- Scoring Possessions ---
    orb_adj = (team.oreb / team_scoring_poss) * team_orb_weight * team_play_pct
    sc_poss = (fg_part + ast_part + ft_part) * (1 - orb_adj) + orb_part

    # --- Missed FG Possessions ---
    fg_x_poss = (player.fga - player.fgm) * (1 - 1.07 * team_orb_pct)

    # --- Missed FT Possessions ---
    ft_x_poss = (ft_miss_rate ** 2) * 0.4 * player.fta if player.fta > 0 else 0.0

    # --- Total Possessions ---
    tot_poss = sc_poss + fg_x_poss + ft_x_poss + player.tov
    if tot_poss == 0:
        return None

    # ========== POINTS PRODUCED ==========

    # --- FG Points Produced ---
    pprod_fg = 2 * (player.fgm + 0.5 * player.fg3m) * (
        1 - 0.5 * (pts_minus_ft / (2 * player.fga)) * qAST
    )

    # --- AST Points Produced ---
    if team_fgm_ex > 0:
        team_3pm_ex = team.fg3m - player.fg3m
        pprod_ast = 2 * ((team_fgm_ex + 0.5 * team_3pm_ex) / team_fgm_ex) * ast_part
    else:
        pprod_ast = 0.0

    # --- ORB Points Produced ---
    pprod_orb = orb_part * (team.pts / team_scoring_poss)

    # --- Total Points Produced ---
    pprod = (pprod_fg + pprod_ast + player.ftm) * (1 - orb_adj) + pprod_orb

    # --- Final ORtg ---
    return round(100 * pprod / tot_poss, 1)


def calculate_drtg(
    player: PlayerBoxStats,
    team: TeamBoxStats,
    opponent: OpponentBoxStats,
) -> float | None:
    """
    Calculate individual Defensive Rating using Dean Oliver's method.

    Returns points allowed per 100 possessions, or None if insufficient data.
    """
    if player.mp == 0 or team.mp == 0:
        return None

    # Team possessions
    team_poss = team.fga - team.oreb + team.tov + 0.44 * team.fta
    if team_poss == 0:
        return None

    # Team DRtg
    team_drtg = 100 * opponent.pts / team_poss

    if opponent.fga == 0:
        return round(team_drtg, 1)

    # DOR% (opponent offensive rebound rate)
    dor_pct = opponent.oreb / (opponent.oreb + team.dreb) if (opponent.oreb + team.dreb) > 0 else 0.0

    # DFG% (opponent field goal percentage)
    dfg_pct = opponent.fgm / opponent.fga

    # FMwt
    d1 = dfg_pct * (1 - dor_pct)
    d2 = (1 - dfg_pct) * dor_pct
    fm_wt = d1 / (d1 + d2) if (d1 + d2) > 0 else 0.0

    # Stops1: directly attributable defensive stops
    stops1 = (
        player.stl
        + player.blk * fm_wt * (1 - 1.07 * dor_pct)
        + player.dreb * (1 - fm_wt)
    )

    # Stops2: estimated share of team defense
    opp_missed = opponent.fga - opponent.fgm - team.blk
    stops2_a = (opp_missed / team.mp) * fm_wt * (1 - 1.07 * dor_pct)
    stops2_b = (opponent.tov - team.stl) / team.mp if team.mp > 0 else 0.0
    opp_ft_miss = (1 - (opponent.ftm / opponent.fta)) ** 2 if opponent.fta > 0 else 0.0
    stops2 = (
        (stops2_a + stops2_b) * player.mp
        + (player.pf / team.pf if team.pf > 0 else 0.0) * 0.4 * opponent.fta * opp_ft_miss
    )

    stops = stops1 + stops2

    # Stop%
    stop_pct = (stops * (team.mp / 5)) / (team_poss * player.mp) if (team_poss * player.mp) > 0 else 0.0

    # D_Pts_per_ScPoss
    opp_ft_miss_rate = (1 - (opponent.ftm / opponent.fta)) if opponent.fta > 0 else 0.0
    opp_scoring_poss = opponent.fgm + (1 - opp_ft_miss_rate ** 2) * opponent.fta * 0.4
    d_pts_per_scposs = opponent.pts / opp_scoring_poss if opp_scoring_poss > 0 else 0.0

    # Final DRtg
    drtg = team_drtg + 0.2 * (100 * d_pts_per_scposs * (1 - stop_pct) - team_drtg)
    return round(drtg, 1)
