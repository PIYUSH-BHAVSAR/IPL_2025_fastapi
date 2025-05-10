
import pandas as pd
import numpy as np
from datetime import datetime

def extract_point_table(matches_df):
    """
    Extract and calculate point table data from matches dataframe for the current season.
    
    Args:
        matches_df (pd.DataFrame): DataFrame containing match data
        
    Returns:
        pd.DataFrame: Point table with team statistics
    """
    # Filter matches for 2025 season
    current_season_matches = matches_df[matches_df['season'] == 2025].copy()
    
    # Initialize point table dictionary with default values
    point_table = {}
    teams = set(current_season_matches['team1'].unique()) | set(current_season_matches['team2'].unique())
    
    for team in teams:
        point_table[team] = {
            'matches_played': 0,
            'matches_won': 0,
            'matches_lost': 0,
            'points': 0,  # Initialize points
            'runs_scored': 0,
            'runs_conceded': 0,
            'net_run_rate': 0.0
        }
        
        # Calculate points based on matches
        team_matches = current_season_matches[
            ((current_season_matches['team1'] == team) | 
             (current_season_matches['team2'] == team))
        ]
        wins = team_matches[team_matches['winner'] == team].shape[0]
        point_table[team]['matches_won'] = wins
        point_table[team]['points'] = wins * 2  # 2 points per win
    
    # Process each match
    for _, match in current_season_matches.iterrows():
        team1 = match['team1']
        team2 = match['team2']
        winner = match['winner']
        
        # Update matches played
        point_table[team1]['matches_played'] += 1
        point_table[team2]['matches_played'] += 1
        
        # Update wins, losses and points
        if winner == team1:
            point_table[team1]['matches_won'] += 1
            point_table[team1]['points'] += 2
            point_table[team2]['matches_lost'] += 1
        elif winner == team2:
            point_table[team2]['matches_won'] += 1
            point_table[team2]['points'] += 2
            point_table[team1]['matches_lost'] += 1
            
        # Update runs (assuming first_innings_score and second_innings_score are available)
        if 'first_innings_score' in match and 'second_innings_score' in match:
            if match['toss_winner'] == team1 and match['toss_decision'] == 'bat':
                point_table[team1]['runs_scored'] += match['first_innings_score']
                point_table[team2]['runs_conceded'] += match['first_innings_score']
                point_table[team2]['runs_scored'] += match['second_innings_score']
                point_table[team1]['runs_conceded'] += match['second_innings_score']
            else:
                point_table[team2]['runs_scored'] += match['first_innings_score']
                point_table[team1]['runs_conceded'] += match['first_innings_score']
                point_table[team1]['runs_scored'] += match['second_innings_score']
                point_table[team2]['runs_conceded'] += match['second_innings_score']
    
    # Calculate Net Run Rate
    for team in teams:
        total_overs = point_table[team]['matches_played'] * 20  # Assuming all matches are 20 overs
        if total_overs > 0:
            run_rate_scored = point_table[team]['runs_scored'] / total_overs
            run_rate_conceded = point_table[team]['runs_conceded'] / total_overs
            point_table[team]['net_run_rate'] = round(run_rate_scored - run_rate_conceded, 3)
    
    # Convert to DataFrame with explicit column order
    columns = ['team', 'matches_played', 'matches_won', 'matches_lost', 'points', 
               'runs_scored', 'runs_conceded', 'net_run_rate']
    
    data = []
    for team, stats in point_table.items():
        row = {
            'team': team,
            'matches_played': stats['matches_played'],
            'matches_won': stats['matches_won'],
            'matches_lost': stats['matches_lost'],
            'points': stats['points'],
            'runs_scored': stats['runs_scored'],
            'runs_conceded': stats['runs_conceded'],
            'net_run_rate': stats['net_run_rate']
        }
        data.append(row)
    
    point_table_df = pd.DataFrame(data, columns=columns)
    
    # Sort by points and NRR
    point_table_df = point_table_df.sort_values(['points', 'net_run_rate'], 
                                              ascending=[False, False]).reset_index(drop=True)
    
    return point_table_df

def get_team_form_metrics(matches_df, team):
    """
    Calculate detailed form metrics for a team in the current season.
    
    Args:
        matches_df (pd.DataFrame): DataFrame containing match data
        team (str): Team name
        
    Returns:
        dict: Dictionary containing team form metrics
    """
    current_season_matches = matches_df[
        (matches_df['season'] == 2025) & 
        ((matches_df['team1'] == team) | (matches_df['team2'] == team))
    ].copy()
    
    # Sort matches by date
    current_season_matches = current_season_matches.sort_values('date')
    
    # Calculate form metrics
    total_matches = len(current_season_matches)
    wins = current_season_matches[current_season_matches['winner'] == team].shape[0]
    losses = total_matches - wins
    
    # Calculate win streak
    results = []
    for _, match in current_season_matches.iterrows():
        results.append(1 if match['winner'] == team else 0)
    
    current_streak = 0
    streak_type = 'none'
    for result in results[::-1]:  # Reverse to get most recent matches first
        if len(results) == 0:
            break
        if result == results[-1]:
            current_streak += 1
        else:
            break
    streak_type = 'wins' if results and results[-1] == 1 else 'losses'
    
    # Calculate recent form (last 5 matches)
    recent_matches = results[-5:] if len(results) >= 5 else results
    recent_form = sum(recent_matches) / len(recent_matches) if recent_matches else 0
    
    return {
        'total_matches': total_matches,
        'wins': wins,
        'losses': losses,
        'win_percentage': (wins/total_matches * 100) if total_matches > 0 else 0,
        'current_streak': current_streak,
        'streak_type': streak_type,
        'recent_form': recent_form
    }

def get_head_to_head_stats(matches_df, team1, team2):
    """
    Calculate head-to-head statistics between two teams in the current season.
    
    Args:
        matches_df (pd.DataFrame): DataFrame containing match data
        team1 (str): First team name
        team2 (str): Second team name
        
    Returns:
        dict: Dictionary containing head-to-head statistics
    """
    # Filter matches between these teams in current season
    head_to_head_matches = matches_df[
        (matches_df['season'] == 2025) &
        (((matches_df['team1'] == team1) & (matches_df['team2'] == team2)) |
         ((matches_df['team1'] == team2) & (matches_df['team2'] == team1)))
    ].copy()
    
    total_matches = len(head_to_head_matches)
    team1_wins = head_to_head_matches[head_to_head_matches['winner'] == team1].shape[0]
    team2_wins = head_to_head_matches[head_to_head_matches['winner'] == team2].shape[0]
    
    return {
        'total_matches': total_matches,
        'team1_wins': team1_wins,
        'team2_wins': team2_wins,
        'team1_win_percentage': (team1_wins/total_matches * 100) if total_matches > 0 else 0,
        'team2_win_percentage': (team2_wins/total_matches * 100) if total_matches > 0 else 0
    }
