import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
import itertools
import optuna
from tqdm import tqdm
import warnings
import tensorflow as tf
from fastapi.encoders import jsonable_encoder
import os

warnings.filterwarnings('ignore')

# 1. Data Loading and Initial Exploration

def load_data():
    print("Loading datasets...")

    # Get base directory of the current file (last_try.py)
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, 'data')

    # Load datasets using relative paths
    matches_df = pd.read_csv(os.path.join(data_dir, 'Final_Dataset.csv'))
    deliveries_df = pd.read_csv(os.path.join(data_dir, 'deliveries_cleaned.csv'))
    points_table_df = pd.read_csv(os.path.join(data_dir, 'ipl_2025_predictions.csv'))

    print(f"Matches dataset shape: {matches_df.shape}")
    print(f"Deliveries dataset shape: {deliveries_df.shape}")
    print(f"Points table loaded with {len(points_table_df)} teams")

    # Convert date to datetime
    matches_df['date'] = pd.to_datetime(matches_df['date'])

    return matches_df, deliveries_df, points_table_df
# 2. Exploratory Data Analysis
def perform_eda(matches_df, deliveries_df):
    print("\nPerforming exploratory data analysis...")
    
    # Analysis of matches over seasons
    plt.figure(figsize=(12, 6))
    matches_per_season = matches_df['season'].value_counts().sort_index()
    matches_per_season.plot(kind='bar')
    plt.title('Number of Matches per Season')
    plt.xlabel('Season')
    plt.ylabel('Number of Matches')
    plt.savefig('matches_per_season.png')
    
    # Analysis of team performance
    plt.figure(figsize=(15, 8))
    team_wins = matches_df['winner'].value_counts()
    team_wins.plot(kind='bar')
    plt.title('Number of Wins by Team')
    plt.xlabel('Team')
    plt.ylabel('Number of Wins')
    plt.tight_layout()
    plt.savefig('team_wins.png')
    
    # Toss factor analysis
    toss_win = matches_df[matches_df['toss_winner'] == matches_df['winner']].shape[0]
    toss_loss = matches_df.shape[0] - toss_win
    plt.figure(figsize=(8, 8))
    plt.pie([toss_win, toss_loss], labels=['Won Toss & Match', 'Lost Toss, Won Match'], 
            autopct='%1.1f%%', startangle=90)
    plt.title('Impact of Winning Toss on Match Outcome')
    plt.savefig('toss_impact.png')
    
    return

def calculate_team_consistency(matches, team):
    """Calculate consistency statistics for a team"""
    team_seasons = matches[
        ((matches['team1'] == team) | (matches['team2'] == team))
    ]['season'].unique()
    
    season_wise_percentages = []
    season_patterns = []
    current_streak = 0
    streak_type = None
    
    for season in sorted(team_seasons):
        season_matches = matches[
            (matches['season'] == season) & 
            ((matches['team1'] == team) | (matches['team2'] == team))
        ].sort_values('date')
        
        # Calculate basic season statistics
        wins = season_matches[season_matches['winner'] == team].shape[0]
        total = season_matches.shape[0]
        win_percentage = (wins / total * 100) if total > 0 else 0
        season_wise_percentages.append(win_percentage)
        
        # Calculate winning patterns
        results = []
        for _, match in season_matches.iterrows():
            if match['winner'] == team:
                results.append('W')
                if streak_type == 'W':
                    current_streak += 1
                else:
                    streak_type = 'W'
                    current_streak = 1
            else:
                results.append('L')
                if streak_type == 'L':
                    current_streak += 1
                else:
                    streak_type = 'L'
                    current_streak = 1
        
        # Calculate season pattern metrics
        win_streaks = [len(list(g)) for k, g in itertools.groupby(results) if k == 'W']
        loss_streaks = [len(list(g)) for k, g in itertools.groupby(results) if k == 'L']
        
        season_patterns.append({
            'season': season,
            'max_win_streak': max(win_streaks) if win_streaks else 0,
            'max_loss_streak': max(loss_streaks) if loss_streaks else 0,
            'avg_win_streak': np.mean(win_streaks) if win_streaks else 0,
            'avg_loss_streak': np.mean(loss_streaks) if loss_streaks else 0,
            'pattern_volatility': len(list(itertools.groupby(results)))
        })
    
    # Calculate overall statistics
    total_matches = len(matches[((matches['team1'] == team) | (matches['team2'] == team))])
    total_wins = len(matches[matches['winner'] == team])
    overall_win_percentage = (total_wins / total_matches * 100) if total_matches > 0 else 0
    win_percentage_variance = np.var(season_wise_percentages) if season_wise_percentages else 0
    
    return {
        'overall_win_percentage': overall_win_percentage,
        'win_variance': win_percentage_variance,
        'is_consistent': 1 if win_percentage_variance < 100 else 0,
        'season_patterns': season_patterns,
        'current_streak_length': current_streak,
        'current_streak_type': streak_type,
        'avg_max_win_streak': np.mean([p['max_win_streak'] for p in season_patterns]),
        'avg_max_loss_streak': np.mean([p['max_loss_streak'] for p in season_patterns]),
        'pattern_consistency': np.std([p['pattern_volatility'] for p in season_patterns])
    }

def get_recent_form(matches, team, season, n=5):
    """Calculate recent form for a team"""
    past_matches = matches[(matches['season'] < season) & 
                         ((matches['team1'] == team) | (matches['team2'] == team))]
    past_matches = past_matches.sort_values('date', ascending=False).head(n)
    
    wins = len(past_matches[past_matches['winner'] == team])
    total_matches = len(past_matches)
    win_percentage = (wins / total_matches * 100) if total_matches > 0 else 0
    
    current_streak = 0
    streak_type = None
    if total_matches > 0:
        for _, match in past_matches.iterrows():
            if match['winner'] == team:
                if streak_type == 'W':
                    current_streak += 1
                else:
                    streak_type = 'W'
                    current_streak = 1
            else:
                if streak_type == 'L':
                    current_streak += 1
                else:
                    streak_type = 'L'
                    current_streak = 1
    
    return {
        'win_percentage': win_percentage,
        'current_streak': current_streak if streak_type == 'W' else -current_streak
    }

def calculate_venue_stats(matches, deliveries):
    """Calculate statistics for each venue"""
    venue_stats = {}
    for venue in matches['venue'].unique():
        venue_matches = matches[matches['venue'] == venue]
        venue_deliveries = deliveries[deliveries['match_id'].isin(venue_matches['match_id'])]
        
        # Calculate first innings scores
        first_innings_scores = []
        for match_id in venue_deliveries[venue_deliveries['inning'] == 1]['match_id'].unique():
            total_score = venue_deliveries[
                (venue_deliveries['match_id'] == match_id) & 
                (venue_deliveries['inning'] == 1)
            ]['total_runs'].sum()
            first_innings_scores.append(total_score)
        
        # Calculate batting first win percentage
        batting_first_matches = venue_matches[venue_matches['toss_decision'] == 'bat']
        batting_first_wins = len(batting_first_matches[
            batting_first_matches['toss_winner'] == batting_first_matches['winner']
        ])
        batting_first_win_percentage = (
            batting_first_wins / len(batting_first_matches) * 100 
            if len(batting_first_matches) > 0 else 0
        )
        
        venue_stats[venue] = {
            'total_matches': len(venue_matches),
            'avg_first_innings_score': np.mean(first_innings_scores) if first_innings_scores else 0,
            'batting_first_win_percentage': batting_first_win_percentage
        }
    
    return venue_stats

# 3. Feature Engineering
# Import point table extractor
from point_table_extractor import extract_point_table, get_team_form_metrics, get_head_to_head_stats

def analyze_current_season(matches_df, points_table_df):
    """
    Analyze current season data and return updated predictions
    """
    # Get current season point table
     # Use actual points table
    point_table = points_table_df
    
    # Filter out eliminated teams (teams with no mathematical chance)
    max_possible_points = {
        team: points + (remaining * 2) 
        for team, points, remaining in zip(
            point_table['Team'], 
            point_table['Points'], 
            point_table['Remaining_Matches']
        )
    }

    # Determine eliminated teams
    current_max_points = point_table['Points'].max()
    eliminated_teams = []
    for team in point_table['Team']:
        if max_possible_points[team] < current_max_points:
            eliminated_teams.append(team)
        
    
    # Get remaining matches for each team
    remaining_matches_info = {}
    for team in point_table['Team']:
        if team not in eliminated_teams:
            remaining = point_table [point_table['Team'] == team]['Remaining_Matches'].iloc[0]

            opponents = []
            future_matches = matches_df[
                (matches_df['season'] == 2025) & ((matches_df['team1'] == team) | (matches_df['team2'] == team)) &
                (matches_df['winner'].isna())  # Matches without results are upcoming
            ]

            for _, match in future_matches.iterrows():
                opponent = match['team2'] if match['team1'] == team else match['team1']
                opponents.append(opponent)

            remaining_matches_info[team] = {
                'matches_remaining': remaining,
                'opponents': opponents,
                'current_points': point_table[point_table['Team'] == team]['Points'].iloc[0],
                'nrr': point_table[point_table['Team'] == team]['NRR'].iloc[0],
                'max_possible_points': max_possible_points[team]
            }


    
    return {
        'point_table': point_table,
        'eliminated_teams': eliminated_teams,
        'remaining_matches': remaining_matches_info
    }

def predict_playoffs(matches_df, model, features_df, points_table_df):
    """
    Predict playoff teams considering current season data
    """
    # Get current season analysis
    current_season_data = analyze_current_season(matches_df, points_table_df)
    
    # Filter out eliminated teams
    valid_teams = [team for team in current_season_data['point_table']['Team'] 
                  if team not in current_season_data['eliminated_teams']]
    
    # Make predictions considering current form and remaining matches
    team_chances = {}
    for team in valid_teams:
        # Get team's current data from points table
        team_data = current_season_data['remaining_matches'].get(team, {})
        current_points = team_data.get('current_points', 0)
        nrr = team_data.get('nrr', 0)
        
        # Create a sample dataframe for prediction
        team_features = features_df[features_df['team1'] == team].copy()
        
        # Convert categorical columns to numeric using one-hot encoding
        categorical_cols = ['team1', 'team2', 'venue', 'toss_winner', 'toss_decision']
        team_data_encoded = pd.get_dummies(team_features, columns=categorical_cols)
        
        # Handle feature names based on model type
        if hasattr(model, 'feature_names_'):
            feature_names = model.feature_names_
        elif hasattr(model, 'feature_name_'):
            feature_names = model.feature_name_
        else:
            feature_names = team_data_encoded.columns
        
        # Ensure all columns from training are present
        for col in feature_names:
            if col not in team_data_encoded.columns:
                team_data_encoded[col] = 0
                
        # Select only the features used during training
        X_pred = team_data_encoded[feature_names]
        
        # Calculate weighted probability considering points and NRR
        points_weight = 0.4
        nrr_weight = 0.2
        model_weight = 0.4
        
        # Normalize points (assuming max possible points is 20)
        points_probability = current_points / 20
        
        # Normalize NRR (typical range is -2 to 2)
        nrr_normalized = (nrr + 2) / 4
        
        # Get model probability
        model_probability = model.predict_proba(X_pred)[0][1] if len(X_pred) > 0 else 0.5
        
        # Calculate final weighted probability
        weighted_probability = (
            (points_weight * points_probability) + 
            (nrr_weight * nrr_normalized) + 
            (model_weight * model_probability)
        )
        
        team_chances[team] = weighted_probability
    
    # Sort teams by probability and get top 4
    playoff_teams = sorted(team_chances.items(), key=lambda x: x[1], reverse=True)[:4]
    
    return [team for team, _ in playoff_teams]

def engineer_features(matches_df, deliveries_df):
    """Main feature engineering function"""
    print("\nEngineering features...")
    
    matches = matches_df.copy()
    deliveries = deliveries_df.copy()
    
    # Calculate team consistency stats first
    teams = set(matches['team1'].unique()) | set(matches['team2'].unique())
    team_consistency_stats = {
        team: calculate_team_consistency(matches, team) 
        for team in tqdm(teams, desc="Calculating team consistency")
    }
    
    # Calculate venue stats
    venue_stats = calculate_venue_stats(matches, deliveries)
    
    # Extract current season's point table
    current_point_table = extract_point_table(matches)
    
    # Extract year from date
    matches['year'] = matches['date'].dt.year
    
    # Create team vs team features
    team_pairs = []
    teams = set(matches['team1'].unique()) | set(matches['team2'].unique())
    for team1 in teams:
        for team2 in teams:
            if team1 != team2:
                team_pairs.append((team1, team2))
    
    # Calculate team consistency features
    team_consistency_stats = {}
    for team in teams:
        # Get all seasons where this team played
        team_seasons = matches[
            ((matches['team1'] == team) | (matches['team2'] == team))
        ]['season'].unique()
        
        season_wise_wins = []
        season_wise_percentages = []
        season_patterns = []
        current_streak = 0
        streak_type = None
        
        for season in sorted(team_seasons):
            season_matches = matches[
                (matches['season'] == season) & 
                ((matches['team1'] == team) | (matches['team2'] == team))
            ].sort_values('date')
            
            # Calculate basic season statistics
            wins = season_matches[season_matches['winner'] == team].shape[0]
            total = season_matches.shape[0]
            win_percentage = (wins / total * 100) if total > 0 else 0
            season_wise_percentages.append(win_percentage)
            season_wise_wins.append(wins)
            
            # Calculate winning patterns
            results = []
            for _, match in season_matches.iterrows():
                if match['winner'] == team:
                    results.append('W')
                    if streak_type == 'W':
                        current_streak += 1
                    else:
                        streak_type = 'W'
                        current_streak = 1
                else:
                    results.append('L')
                    if streak_type == 'L':
                        current_streak += 1
                    else:
                        streak_type = 'L'
                        current_streak = 1
            
            # Calculate season pattern metrics
            win_streaks = [len(list(g)) for k, g in itertools.groupby(results) if k == 'W']
            loss_streaks = [len(list(g)) for k, g in itertools.groupby(results) if k == 'L']
            
            season_patterns.append({
                'season': season,
                'max_win_streak': max(win_streaks) if win_streaks else 0,
                'max_loss_streak': max(loss_streaks) if loss_streaks else 0,
                'avg_win_streak': np.mean(win_streaks) if win_streaks else 0,
                'avg_loss_streak': np.mean(loss_streaks) if loss_streaks else 0,
                'pattern_volatility': len(list(itertools.groupby(results)))
            })
        
        # Calculate overall statistics
        total_matches = matches[
            ((matches['team1'] == team) | (matches['team2'] == team))
        ].shape[0]
        total_wins = matches[matches['winner'] == team].shape[0]
        overall_win_percentage = (total_wins / total_matches * 100) if total_matches > 0 else 0
        
        # Calculate variance and consistency
        win_percentage_variance = np.var(season_wise_percentages) if season_wise_percentages else 0
        is_consistent = 1 if win_percentage_variance < 100 else 0  # Threshold of 100 for variance
        
        # Store all team statistics
        team_consistency_stats[team] = {
            'overall_win_percentage': overall_win_percentage,
            'season_wise_percentages': season_wise_percentages,
            'win_variance': win_percentage_variance,
            'is_consistent': is_consistent,
            'season_patterns': season_patterns,
            'current_streak_length': current_streak,
            'current_streak_type': streak_type,
            'avg_max_win_streak': np.mean([p['max_win_streak'] for p in season_patterns]),
            'avg_max_loss_streak': np.mean([p['max_loss_streak'] for p in season_patterns]),
            'pattern_consistency': np.std([p['pattern_volatility'] for p in season_patterns])
        }

    # Create dictionary to store team vs team statistics
    team_vs_team_stats = {}
    for team1, team2 in team_pairs:
        team1_team2_matches = matches[((matches['team1'] == team1) & (matches['team2'] == team2)) |
                                    ((matches['team1'] == team2) & (matches['team2'] == team1))]
        
        team1_wins = team1_team2_matches[team1_team2_matches['winner'] == team1].shape[0]
        team2_wins = team1_team2_matches[team1_team2_matches['winner'] == team2].shape[0]
        total_matches = team1_team2_matches.shape[0]
        
        if total_matches > 0:
            team1_win_percentage = team1_wins / total_matches * 100
            team2_win_percentage = team2_wins / total_matches * 100
        else:
            team1_win_percentage = 0
            team2_win_percentage = 0
        
        team_vs_team_stats[(team1, team2)] = {
            'total_matches': total_matches,
            'team1_wins': team1_wins,
            'team2_wins': team2_wins,
            'team1_win_percentage': team1_win_percentage,
            'team2_win_percentage': team2_win_percentage
        }

        total_matches = matches[
            ((matches['team1'] == team) | (matches['team2'] == team))
        ].shape[0]
        total_wins = matches[matches['winner'] == team].shape[0]
        overall_win_percentage = (total_wins / total_matches * 100) if total_matches > 0 else 0
        
        # Calculate season form patterns
        season_patterns = []
        current_streak = 0
        streak_type = None
        
        for season in sorted(team_seasons):
            season_matches = matches[
                (matches['season'] == season) & 
                ((matches['team1'] == team) | (matches['team2'] == team))
            ].sort_values('date')
            
            results = []
            for _, match in season_matches.iterrows():
                if match['winner'] == team:
                    results.append('W')
                    if streak_type == 'W':
                        current_streak += 1
                    else:
                        streak_type = 'W'
                        current_streak = 1
                else:
                    results.append('L')
                    if streak_type == 'L':
                        current_streak += 1
                    else:
                        streak_type = 'L'
                        current_streak = 1
            
            # Calculate season pattern metrics
            win_streaks = [len(list(g)) for k, g in itertools.groupby(results) if k == 'W']
            loss_streaks = [len(list(g)) for k, g in itertools.groupby(results) if k == 'L']
            
            season_patterns.append({
                'season': season,
                'max_win_streak': max(win_streaks) if win_streaks else 0,
                'max_loss_streak': max(loss_streaks) if loss_streaks else 0,
                'avg_win_streak': np.mean(win_streaks) if win_streaks else 0,
                'avg_loss_streak': np.mean(loss_streaks) if loss_streaks else 0,
                'pattern_volatility': len(list(itertools.groupby(results)))  # Number of streak changes
            })
        
        team_consistency_stats[team] = {
            'overall_win_percentage': overall_win_percentage,
            'season_wise_percentages': season_wise_percentages,
            'win_variance': win_percentage_variance,
            'is_consistent': is_consistent,
            'season_patterns': season_patterns,
            'current_streak_length': current_streak,
            'current_streak_type': streak_type,
            'avg_max_win_streak': np.mean([p['max_win_streak'] for p in season_patterns]),
            'avg_max_loss_streak': np.mean([p['max_loss_streak'] for p in season_patterns]),
            'pattern_consistency': np.std([p['pattern_volatility'] for p in season_patterns])
        }

    # Create venue-specific features
    venue_stats = {}
    venues = matches['venue'].unique()
    for venue in venues:
        venue_matches = matches[matches['venue'] == venue]
        
        # Calculate average scores at venue
        venue_deliveries = deliveries[deliveries['match_id'].isin(venue_matches['match_id'])]
        first_innings_matches = venue_deliveries[venue_deliveries['inning'] == 1]['match_id'].unique()
        first_innings_scores = []
        
        for match_id in first_innings_matches:
            match_deliveries = venue_deliveries[venue_deliveries['match_id'] == match_id]
            total_score = match_deliveries['total_runs'].sum()
            first_innings_scores.append(total_score)
        
        # Batting first win percentage
        batting_first_matches = venue_matches[venue_matches['toss_decision'] == 'bat']
        if len(batting_first_matches) > 0:
            batting_first_wins = batting_first_matches[batting_first_matches['toss_winner'] == batting_first_matches['winner']].shape[0]
            batting_first_win_percentage = batting_first_wins / len(batting_first_matches) * 100
        else:
            batting_first_win_percentage = 0
        
        venue_stats[venue] = {
            'total_matches': venue_matches.shape[0],
            'avg_first_innings_score': np.mean(first_innings_scores) if first_innings_scores else 0,
            'batting_first_win_percentage': batting_first_win_percentage
        }
    
    # Create recent form features for teams
    def get_recent_form(team, season, n=5):
        # Get matches before the current season
        past_matches = matches[(matches['season'] < season) & 
                             ((matches['team1'] == team) | (matches['team2'] == team))]
        past_matches = past_matches.sort_values('date', ascending=False).head(n)
        
        # Calculate wins in recent matches
        wins = past_matches[past_matches['winner'] == team].shape[0]
        total_matches = len(past_matches)
        
        # Calculate win percentage and current streak
        win_percentage = (wins / total_matches * 100) if total_matches > 0 else 0
        
        # Calculate current streak
        current_streak = 0
        streak_type = None
        if total_matches > 0:
            for _, match in past_matches.iterrows():
                if match['winner'] == team:
                    if streak_type == 'W':
                        current_streak += 1
                    else:
                        streak_type = 'W'
                        current_streak = 1
                else:
                    if streak_type == 'L':
                        current_streak += 1
                    else:
                        streak_type = 'L'
                        current_streak = 1
        
        return {
            'win_percentage': win_percentage,
            'current_streak': current_streak if streak_type == 'W' else -current_streak
        }
    
    # Create a list to store match features
    match_features = []
    
    # Create features for each match
    for _, match in tqdm(matches.iterrows(), total=matches.shape[0], desc="Creating match features"):
        # Get current season point table data if match is from 2025
        if match['season'] == 2025:
            team1_form = get_team_form_metrics(matches, match['team1'])
            team2_form = get_team_form_metrics(matches, match['team2'])
            head_to_head = get_head_to_head_stats(matches, match['team1'], match['team2'])
            
            # Get team standings from point table
            team1_standing = current_point_table[current_point_table['team'] == match['team1']].iloc[0]
            team2_standing = current_point_table[current_point_table['team'] == match['team2']].iloc[0]
        team1 = match['team1']
        team2 = match['team2']
        season = match['season']
        venue = match['venue']
        
        # Team vs Team features
        if (team1, team2) in team_vs_team_stats:
            t1_vs_t2_stats = team_vs_team_stats[(team1, team2)]
        elif (team2, team1) in team_vs_team_stats:
            t1_vs_t2_stats = team_vs_team_stats[(team2, team1)]
            # Swap team1 and team2 statistics
            t1_vs_t2_stats = {
                'total_matches': t1_vs_t2_stats['total_matches'],
                'team1_wins': t1_vs_t2_stats['team2_wins'],
                'team2_wins': t1_vs_t2_stats['team1_wins'],
                'team1_win_percentage': t1_vs_t2_stats['team2_win_percentage'],
                'team2_win_percentage': t1_vs_t2_stats['team1_win_percentage']
            }
        else:
            t1_vs_t2_stats = {
                'total_matches': 0,
                'team1_wins': 0,
                'team2_wins': 0,
                'team1_win_percentage': 0,
                'team2_win_percentage': 0
            }
        
        # Venue features
        if venue in venue_stats:
            venue_features = venue_stats[venue]
        else:
            venue_features = {
                'total_matches': 0,
                'avg_first_innings_score': 0,
                'batting_first_win_percentage': 0
            }
        
        # Recent form features
        team1_form = get_recent_form(team1, season)
        team2_form = get_recent_form(team2, season)
        
        # Create a feature dictionary for this match
        match_feature = {
            'match_id': match['match_id'],
            'season': season,
            'team1': team1,
            'team2': team2,
            'venue': venue,
            'toss_winner': match['toss_winner'],
            'toss_decision': match['toss_decision'],
            'total_matches_between': t1_vs_t2_stats['total_matches'],
            'team1_wins_against_team2': t1_vs_t2_stats['team1_wins'],
            'team2_wins_against_team1': t1_vs_t2_stats['team2_wins'],
            'team1_win_percentage_against_team2': t1_vs_t2_stats['team1_win_percentage'],
            'team2_win_percentage_against_team1': t1_vs_t2_stats['team2_win_percentage'],
            'venue_total_matches': venue_features['total_matches'],
            'venue_avg_first_innings_score': venue_features['avg_first_innings_score'],
            'venue_batting_first_win_percentage': venue_features['batting_first_win_percentage'],
            'team1_recent_form': team1_form['win_percentage'],
            'team2_recent_form': team2_form['win_percentage'],
            # Add current season performance metrics if available
            'team1_current_points': team1_standing['points'] if match['season'] == 2025 else 0,
            'team2_current_points': team2_standing['points'] if match['season'] == 2025 else 0,
            'team1_current_nrr': team1_standing['net_run_rate'] if match['season'] == 2025 else 0,
            'team2_current_nrr': team2_standing['net_run_rate'] if match['season'] == 2025 else 0,
            'team1_current_win_percentage': team1_form['win_percentage'] if match['season'] == 2025 else 0,
            'team2_current_win_percentage': team2_form['win_percentage'] if match['season'] == 2025 else 0,
            'team1_current_streak': team1_form['current_streak'] if match['season'] == 2025 else 0,
            'team2_current_streak': team2_form['current_streak'] if match['season'] == 2025 else 0,
            'current_season_h2h_matches': head_to_head['total_matches'] if match['season'] == 2025 else 0,
            'team1_current_season_h2h_wins': head_to_head['team1_wins'] if match['season'] == 2025 else 0,
            'team2_current_season_h2h_wins': head_to_head['team2_wins'] if match['season'] == 2025 else 0,
            'team1_win_percentage_over_seasons': team_consistency_stats[team1]['overall_win_percentage'],
            'team2_win_percentage_over_seasons': team_consistency_stats[team2]['overall_win_percentage'],
            'team1_win_variance_over_seasons': team_consistency_stats[team1]['win_variance'],
            'team2_win_variance_over_seasons': team_consistency_stats[team2]['win_variance'],
            'team1_consistent_performance': team_consistency_stats[team1]['is_consistent'],
            'team2_consistent_performance': team_consistency_stats[team2]['is_consistent'],
            'team1_current_streak_length': team_consistency_stats[team1].get('current_streak_length', 0),
            'team2_current_streak_length': team_consistency_stats[team2].get('current_streak_length', 0),
            'team1_avg_max_win_streak': team_consistency_stats[team1].get('avg_max_win_streak', 0),
            'team2_avg_max_win_streak': team_consistency_stats[team2].get('avg_max_win_streak', 0),
            'team1_avg_max_loss_streak': team_consistency_stats[team1].get('avg_max_loss_streak', 0),
            'team2_avg_max_loss_streak': team_consistency_stats[team2].get('avg_max_loss_streak', 0),
            'team1_pattern_consistency': team_consistency_stats[team1].get('pattern_consistency', 0),
            'team2_pattern_consistency': team_consistency_stats[team2].get('pattern_consistency', 0),
            'winner': match['winner'] if pd.notna(match['winner']) else None
        }
        
        match_features.append(match_feature)
    
    # Convert to DataFrame
    features_df = pd.DataFrame(match_features)
    
    # Move the team consistency calculation before the match feature creation loop
    # Calculate team consistency features
    team_consistency_stats = {}
    for team in teams:
        # Get all seasons where this team played
        team_seasons = matches[
            ((matches['team1'] == team) | (matches['team2'] == team))
        ]['season'].unique()
        
        season_wise_wins = []
        season_wise_percentages = []
        season_patterns = []
        current_streak = 0
        streak_type = None
        
        for season in sorted(team_seasons):
            season_matches = matches[
                (matches['season'] == season) & 
                ((matches['team1'] == team) | (matches['team2'] == team))
            ].sort_values('date')
            
            # Calculate basic season statistics
            wins = season_matches[season_matches['winner'] == team].shape[0]
            total = season_matches.shape[0]
            win_percentage = (wins / total * 100) if total > 0 else 0
            season_wise_percentages.append(win_percentage)
            season_wise_wins.append(wins)
            
            # Calculate winning patterns
            results = []
            for _, match in season_matches.iterrows():
                if match['winner'] == team:
                    results.append('W')
                    if streak_type == 'W':
                        current_streak += 1
                    else:
                        streak_type = 'W'
                        current_streak = 1
                else:
                    results.append('L')
                    if streak_type == 'L':
                        current_streak += 1
                    else:
                        streak_type = 'L'
                        current_streak = 1
            
            # Calculate season pattern metrics
            win_streaks = [len(list(g)) for k, g in itertools.groupby(results) if k == 'W']
            loss_streaks = [len(list(g)) for k, g in itertools.groupby(results) if k == 'L']
            
            season_patterns.append({
                'season': season,
                'max_win_streak': max(win_streaks) if win_streaks else 0,
                'max_loss_streak': max(loss_streaks) if loss_streaks else 0,
                'avg_win_streak': np.mean(win_streaks) if win_streaks else 0,
                'avg_loss_streak': np.mean(loss_streaks) if loss_streaks else 0,
                'pattern_volatility': len(list(itertools.groupby(results)))
            })
        
        # Calculate overall statistics
        total_matches = matches[
            ((matches['team1'] == team) | (matches['team2'] == team))
        ].shape[0]
        total_wins = matches[matches['winner'] == team].shape[0]
        overall_win_percentage = (total_wins / total_matches * 100) if total_matches > 0 else 0
        
        # Calculate variance and consistency
        win_percentage_variance = np.var(season_wise_percentages) if season_wise_percentages else 0
        is_consistent = 1 if win_percentage_variance < 100 else 0  # Threshold of 100 for variance
        
        # Store all team statistics
        team_consistency_stats[team] = {
            'overall_win_percentage': overall_win_percentage,
            'season_wise_percentages': season_wise_percentages,
            'win_variance': win_percentage_variance,
            'is_consistent': is_consistent,
            'season_patterns': season_patterns,
            'current_streak_length': current_streak,
            'current_streak_type': streak_type,
            'avg_max_win_streak': np.mean([p['max_win_streak'] for p in season_patterns]),
            'avg_max_loss_streak': np.mean([p['max_loss_streak'] for p in season_patterns]),
            'pattern_consistency': np.std([p['pattern_volatility'] for p in season_patterns])
        }

    # Then in the match feature creation loop, update match_feature to include consistency features
    for _, match in tqdm(matches.iterrows(), total=matches.shape[0], desc="Creating match features"):
        team1 = match['team1']
        team2 = match['team2']
        season = match['season']
        venue = match['venue']
        
        # Team vs Team features
        if (team1, team2) in team_vs_team_stats:
            t1_vs_t2_stats = team_vs_team_stats[(team1, team2)]
        elif (team2, team1) in team_vs_team_stats:
            t1_vs_t2_stats = team_vs_team_stats[(team2, team1)]
            # Swap team1 and team2 statistics
            t1_vs_t2_stats = {
                'total_matches': t1_vs_t2_stats['total_matches'],
                'team1_wins': t1_vs_t2_stats['team2_wins'],
                'team2_wins': t1_vs_t2_stats['team1_wins'],
                'team1_win_percentage': t1_vs_t2_stats['team2_win_percentage'],
                'team2_win_percentage': t1_vs_t2_stats['team1_win_percentage']
            }
        else:
            t1_vs_t2_stats = {
                'total_matches': 0,
                'team1_wins': 0,
                'team2_wins': 0,
                'team1_win_percentage': 0,
                'team2_win_percentage': 0
            }
        
        # Venue features
        if venue in venue_stats:
            venue_features = venue_stats[venue]
        else:
            venue_features = {
                'total_matches': 0,
                'avg_first_innings_score': 0,
                'batting_first_win_percentage': 0
            }
        
        # Recent form features
        team1_form = get_recent_form(team1, season)
        team2_form = get_recent_form(team2, season)
        
        # Create a feature dictionary for this match
        match_feature = {
            'match_id': match['match_id'],
            'season': season,
            'team1': team1,
            'team2': team2,
            'venue': venue,
            'toss_winner': match['toss_winner'],
            'toss_decision': match['toss_decision'],
            'total_matches_between': t1_vs_t2_stats['total_matches'],
            'team1_wins_against_team2': t1_vs_t2_stats['team1_wins'],
            'team2_wins_against_team1': t1_vs_t2_stats['team2_wins'],
            'team1_win_percentage_against_team2': t1_vs_t2_stats['team1_win_percentage'],
            'team2_win_percentage_against_team1': t1_vs_t2_stats['team2_win_percentage'],
            'venue_total_matches': venue_features['total_matches'],
            'venue_avg_first_innings_score': venue_features['avg_first_innings_score'],
            'venue_batting_first_win_percentage': venue_features['batting_first_win_percentage'],
            'team1_recent_form': team1_form['win_percentage'],
            'team2_recent_form': team2_form['win_percentage'],
            # Add current season performance metrics if available
            'team1_current_points': team1_standing['points'] if match['season'] == 2025 else 0,
            'team2_current_points': team2_standing['points'] if match['season'] == 2025 else 0,
            'team1_current_nrr': team1_standing['net_run_rate'] if match['season'] == 2025 else 0,
            'team2_current_nrr': team2_standing['net_run_rate'] if match['season'] == 2025 else 0,
            'team1_current_win_percentage': team1_form['win_percentage'] if match['season'] == 2025 else 0,
            'team2_current_win_percentage': team2_form['win_percentage'] if match['season'] == 2025 else 0,
            'team1_current_streak': team1_form['current_streak'] if match['season'] == 2025 else 0,
            'team2_current_streak': team2_form['current_streak'] if match['season'] == 2025 else 0,
            'current_season_h2h_matches': head_to_head['total_matches'] if match['season'] == 2025 else 0,
            'team1_current_season_h2h_wins': head_to_head['team1_wins'] if match['season'] == 2025 else 0,
            'team2_current_season_h2h_wins': head_to_head['team2_wins'] if match['season'] == 2025 else 0,
            'team1_win_percentage_over_seasons': team_consistency_stats[team1]['overall_win_percentage'],
            'team2_win_percentage_over_seasons': team_consistency_stats[team2]['overall_win_percentage'],
            'team1_win_variance_over_seasons': team_consistency_stats[team1]['win_variance'],
            'team2_win_variance_over_seasons': team_consistency_stats[team2]['win_variance'],
            'team1_consistent_performance': team_consistency_stats[team1]['is_consistent'],
            'team2_consistent_performance': team_consistency_stats[team2]['is_consistent'],
            'team1_current_streak_length': team_consistency_stats[team1].get('current_streak_length', 0),
            'team2_current_streak_length': team_consistency_stats[team2].get('current_streak_length', 0),
            'team1_avg_max_win_streak': team_consistency_stats[team1].get('avg_max_win_streak', 0),
            'team2_avg_max_win_streak': team_consistency_stats[team2].get('avg_max_win_streak', 0),
            'team1_avg_max_loss_streak': team_consistency_stats[team1].get('avg_max_loss_streak', 0),
            'team2_avg_max_loss_streak': team_consistency_stats[team2].get('avg_max_loss_streak', 0),
            'team1_pattern_consistency': team_consistency_stats[team1].get('pattern_consistency', 0),
            'team2_pattern_consistency': team_consistency_stats[team2].get('pattern_consistency', 0),
            'winner': match['winner'] if pd.notna(match['winner']) else None
        }
        
        # Add consistency features to match_feature
        match_feature.update({
            'team1_win_percentage_over_seasons': team_consistency_stats[team1]['overall_win_percentage'],
            'team2_win_percentage_over_seasons': team_consistency_stats[team2]['overall_win_percentage'],
            'team1_win_variance_over_seasons': team_consistency_stats[team1]['win_variance'],
            'team2_win_variance_over_seasons': team_consistency_stats[team2]['win_variance'],
            'team1_consistent_performance': team_consistency_stats[team1]['is_consistent'],
            'team2_consistent_performance': team_consistency_stats[team2]['is_consistent']
        })
        
        match_features.append(match_feature)
    
    # Calculate aggregate team-level batter vs bowler statistics for each match
    for match_feature in match_features:
        match_id = match_feature['match_id']
        team1 = match_feature['team1']
        team2 = match_feature['team2']
        
        # Get match-specific deliveries
        match_deliveries = deliveries[deliveries['match_id'] == match_id]
        
        # Initialize team-level statistics
        team1_bat_stats = {'runs': 0, 'balls': 0, 'dismissals': 0}
        team2_bat_stats = {'runs': 0, 'balls': 0, 'dismissals': 0}
        
        # Calculate team1 batting vs team2 bowling stats
        team1_batting = match_deliveries[match_deliveries['batting_team'] == team1]
        if not team1_batting.empty:
            team1_bat_stats['runs'] = team1_batting['batsman_runs'].sum()
            team1_bat_stats['balls'] = len(team1_batting)
            team1_bat_stats['dismissals'] = team1_batting['player_dismissed'].notna().sum()
        
        # Calculate team2 batting vs team1 bowling stats
        team2_batting = match_deliveries[match_deliveries['batting_team'] == team2]
        if not team2_batting.empty:
            team2_bat_stats['runs'] = team2_batting['batsman_runs'].sum()
            team2_bat_stats['balls'] = len(team2_batting)
            team2_bat_stats['dismissals'] = team2_batting['player_dismissed'].notna().sum()
        
        # Calculate derived statistics for team1
        match_feature.update({
            'team1_bat_vs_team2_bowl_runs': team1_bat_stats['runs'],
            'team1_bat_vs_team2_bowl_balls': team1_bat_stats['balls'],
            'team1_bat_vs_team2_bowl_dismissals': team1_bat_stats['dismissals'],
            'team1_bat_vs_team2_bowl_sr': (team1_bat_stats['runs'] / team1_bat_stats['balls'] * 100) if team1_bat_stats['balls'] > 0 else 0,
            'team1_bat_vs_team2_bowl_avg': (team1_bat_stats['runs'] / team1_bat_stats['dismissals']) if team1_bat_stats['dismissals'] > 0 else team1_bat_stats['runs'],
            'team1_bat_vs_team2_bowl_economy': (team1_bat_stats['runs'] / (team1_bat_stats['balls']/6)) if team1_bat_stats['balls'] > 0 else 0,
            'team1_bat_vs_team2_bowl_frequency': 1,
            
            'team2_bat_vs_team1_bowl_runs': team2_bat_stats['runs'],
            'team2_bat_vs_team1_bowl_balls': team2_bat_stats['balls'],
            'team2_bat_vs_team1_bowl_dismissals': team2_bat_stats['dismissals'],
            'team2_bat_vs_team1_bowl_sr': (team2_bat_stats['runs'] / team2_bat_stats['balls'] * 100) if team2_bat_stats['balls'] > 0 else 0,
            'team2_bat_vs_team1_bowl_avg': (team2_bat_stats['runs'] / team2_bat_stats['dismissals']) if team2_bat_stats['dismissals'] > 0 else team2_bat_stats['runs'],
            'team2_bat_vs_team1_bowl_economy': (team2_bat_stats['runs'] / (team2_bat_stats['balls']/6)) if team2_bat_stats['balls'] > 0 else 0,
            'team2_bat_vs_team1_bowl_frequency': 1
        })
    
    # Convert to DataFrame
    features_df = pd.DataFrame(match_features)
    
    # Calculate team consistency features (add this before creating match_features list)
    team_consistency_stats = {}
    for team in teams:
        # Get all seasons where this team played
        team_seasons = matches[
            ((matches['team1'] == team) | (matches['team2'] == team))
        ]['season'].unique()
        
        season_wise_wins = []
        season_wise_percentages = []
        
        for season in team_seasons:
            season_matches = matches[
                (matches['season'] == season) & 
                ((matches['team1'] == team) | (matches['team2'] == team))
            ]
            wins = season_matches[season_matches['winner'] == team].shape[0]
            total = season_matches.shape[0]
            win_percentage = (wins / total * 100) if total > 0 else 0
            season_wise_percentages.append(win_percentage)
            season_wise_wins.append(wins)
        
        # Calculate overall statistics
        total_matches = matches[
            ((matches['team1'] == team) | (matches['team2'] == team))
        ].shape[0]
        total_wins = matches[matches['winner'] == team].shape[0]
        overall_win_percentage = (total_wins / total_matches * 100) if total_matches > 0 else 0
        
        # Calculate variance and consistency
        win_percentage_variance = np.var(season_wise_percentages) if season_wise_percentages else 0
        is_consistent = 1 if win_percentage_variance < 100 else 0  # Threshold of 100 for variance
        
        team_consistency_stats[team] = {
            'overall_win_percentage': overall_win_percentage,
            'season_wise_percentages': season_wise_percentages,
            'win_variance': win_percentage_variance,
            'is_consistent': is_consistent
        }

    # Then in the match feature creation loop, add the consistency features
    match_feature.update({
        'team1_win_percentage_over_seasons': team_consistency_stats[team1]['overall_win_percentage'],
        'team2_win_percentage_over_seasons': team_consistency_stats[team2]['overall_win_percentage'],
        'team1_win_variance_over_seasons': team_consistency_stats[team1]['win_variance'],
        'team2_win_variance_over_seasons': team_consistency_stats[team2]['win_variance'],
        'team1_consistent_performance': team_consistency_stats[team1]['is_consistent'],
        'team2_consistent_performance': team_consistency_stats[team2]['is_consistent']
    })
    
    # Add this after creating team_vs_team_stats and before venue_stats
    
    # Calculate aggregate team-level batter vs bowler statistics for each match
    for match_feature in match_features:
        match_id = match_feature['match_id']
        team1 = match_feature['team1']
        team2 = match_feature['team2']
        
        # Get match-specific deliveries
        match_deliveries = deliveries[deliveries['match_id'] == match_id]
        
        # Initialize team-level statistics
        team1_bat_stats = {'runs': 0, 'balls': 0, 'dismissals': 0}
        team2_bat_stats = {'runs': 0, 'balls': 0, 'dismissals': 0}
        
        # Calculate team1 batting vs team2 bowling stats
        team1_batting = match_deliveries[match_deliveries['batting_team'] == team1]
        if not team1_batting.empty:
            team1_bat_stats['runs'] = team1_batting['batsman_runs'].sum()
            team1_bat_stats['balls'] = len(team1_batting)
            team1_bat_stats['dismissals'] = team1_batting['player_dismissed'].notna().sum()
        
        # Calculate team2 batting vs team1 bowling stats
        team2_batting = match_deliveries[match_deliveries['batting_team'] == team2]
        if not team2_batting.empty:
            team2_bat_stats['runs'] = team2_batting['batsman_runs'].sum()
            team2_bat_stats['balls'] = len(team2_batting)
            team2_bat_stats['dismissals'] = team2_batting['player_dismissed'].notna().sum()
        
        # Calculate derived statistics for team1
        match_feature['team1_bat_vs_team2_bowl_runs'] = team1_bat_stats['runs']
        match_feature['team1_bat_vs_team2_bowl_balls'] = team1_bat_stats['balls']
        match_feature['team1_bat_vs_team2_bowl_dismissals'] = team1_bat_stats['dismissals']
        match_feature['team1_bat_vs_team2_bowl_sr'] = (team1_bat_stats['runs'] / team1_bat_stats['balls'] * 100) if team1_bat_stats['balls'] > 0 else 0
        match_feature['team1_bat_vs_team2_bowl_avg'] = (team1_bat_stats['runs'] / team1_bat_stats['dismissals']) if team1_bat_stats['dismissals'] > 0 else team1_bat_stats['runs']
        match_feature['team1_bat_vs_team2_bowl_economy'] = (team1_bat_stats['runs'] / (team1_bat_stats['balls']/6)) if team1_bat_stats['balls'] > 0 else 0
        match_feature['team1_bat_vs_team2_bowl_frequency'] = 1
        
        # Calculate derived statistics for team2
        match_feature['team2_bat_vs_team1_bowl_runs'] = team2_bat_stats['runs']
        match_feature['team2_bat_vs_team1_bowl_balls'] = team2_bat_stats['balls']
        match_feature['team2_bat_vs_team1_bowl_dismissals'] = team2_bat_stats['dismissals']
        match_feature['team2_bat_vs_team1_bowl_sr'] = (team2_bat_stats['runs'] / team2_bat_stats['balls'] * 100) if team2_bat_stats['balls'] > 0 else 0
        match_feature['team2_bat_vs_team1_bowl_avg'] = (team2_bat_stats['runs'] / team2_bat_stats['dismissals']) if team2_bat_stats['dismissals'] > 0 else team2_bat_stats['runs']
        match_feature['team2_bat_vs_team1_bowl_economy'] = (team2_bat_stats['runs'] / (team2_bat_stats['balls']/6)) if team2_bat_stats['balls'] > 0 else 0
        match_feature['team2_bat_vs_team1_bowl_frequency',
        ] = 1
    
    # Add this after creating team_vs_team_stats and before venue_stats
    
    # Create batter vs bowler statistics
    batter_bowler_stats = {}
    for _, delivery in tqdm(deliveries.iterrows(), desc="Calculating batter-bowler matchups"):
        batter = delivery['batter']  # Changed from 'batsman' to 'batter'
        bowler = delivery['bowler']
        pair_key = (batter, bowler)
        
        if pair_key not in batter_bowler_stats:
            batter_bowler_stats[pair_key] = {
                'runs': 0,
                'balls': 0,
                'dismissals': 0,
                'matches': set()
            }
        
        # Update statistics
        stats = batter_bowler_stats[pair_key]
        stats['runs'] += delivery['batsman_runs']
        stats['balls'] += 1
        stats['matches'].add(delivery['match_id'])
        
        # Check for dismissal
        if pd.notna(delivery['player_dismissed']) and delivery['player_dismissed'] == batter:
            stats['dismissals'] += 1
    
    # Calculate derived statistics
    for pair_key, stats in batter_bowler_stats.items():
        batter, bowler = pair_key
        balls = stats['balls']
        runs = stats['runs']
        dismissals = stats['dismissals']
        matches = len(stats['matches'])
        
        # Calculate averages and rates
        stats['strike_rate'] = (runs / balls * 100) if balls > 0 else 0
        stats['average'] = (runs / dismissals) if dismissals > 0 else runs if runs > 0 else 0
        stats['economy'] = (runs / (balls/6)) if balls > 0 else 0
        stats['bowling_strike_rate'] = (balls / dismissals) if dismissals > 0 else balls if balls > 0 else 0
        stats['frequency'] = matches
    
    # Add batter-bowler features to match features
    def get_batter_bowler_stats(batter, bowler):
        pair_key = (batter, bowler)
        if pair_key in batter_bowler_stats:
            return batter_bowler_stats[pair_key]
        return {
            'runs': 0, 'balls': 0, 'dismissals': 0, 'strike_rate': 0,
            'average': 0, 'economy': 0, 'bowling_strike_rate': 0, 'frequency': 0
        }
    
    # In the match feature creation loop, add these features
    # Add this inside the match features loop after creating the basic match_feature dictionary:
        
        # Get playing XIs for the match
        match_deliveries = deliveries[deliveries['match_id'] == match['match_id']]
        team1_batters = set(match_deliveries[match_deliveries['batting_team'] == team1]['batsman'])
        team1_bowlers = set(match_deliveries[match_deliveries['bowling_team'] == team1]['bowler'])
        team2_batters = set(match_deliveries[match_deliveries['batting_team'] == team2]['batsman'])
        team2_bowlers = set(match_deliveries[match_deliveries['bowling_team'] == team2]['bowler'])
        
        # Calculate aggregate matchup statistics
        team1_bat_vs_team2_bowl = {
            'runs': 0, 'balls': 0, 'dismissals': 0, 'strike_rate': 0,
            'average': 0, 'economy': 0, 'bowling_strike_rate': 0, 'frequency': 0
        }
        
        team2_bat_vs_team1_bowl = {
            'runs': 0, 'balls': 0, 'dismissals': 0, 'strike_rate': 0,
            'average': 0, 'economy': 0, 'bowling_strike_rate': 0, 'frequency': 0
        }
        
        # Aggregate statistics for team1 batters vs team2 bowlers
        for batter in team1_batters:
            for bowler in team2_bowlers:
                stats = get_batter_bowler_stats(batter, bowler)
                for key in ['runs', 'balls', 'dismissals', 'frequency']:
                    team1_bat_vs_team2_bowl[key] += stats[key]
        
        # Aggregate statistics for team2 batters vs team1 bowlers
        for batter in team2_batters:
            for bowler in team1_bowlers:
                stats = get_batter_bowler_stats(batter, bowler)
                for key in ['runs', 'balls', 'dismissals', 'frequency']:
                    team2_bat_vs_team1_bowl[key] += stats[key]
        
        # Calculate aggregate rates
        for team_stats in [team1_bat_vs_team2_bowl, team2_bat_vs_team1_bowl]:
            balls = team_stats['balls']
            runs = team_stats['runs']
            dismissals = team_stats['dismissals']
            
            team_stats['strike_rate'] = (runs / balls * 100) if balls > 0 else 0
            team_stats['average'] = (runs / dismissals) if dismissals > 0 else runs if runs > 0 else 0
            team_stats['economy'] = (runs / (balls/6)) if balls > 0 else 0
            team_stats['bowling_strike_rate'] = (balls / dismissals) if dismissals > 0 else balls if balls > 0 else 0
        
        # Add features to match_feature
        match_feature.update({
            'team1_bat_vs_team2_bowl_runs': team1_bat_vs_team2_bowl['runs'],
            'team1_bat_vs_team2_bowl_balls': team1_bat_vs_team2_bowl['balls'],
            'team1_bat_vs_team2_bowl_dismissals': team1_bat_vs_team2_bowl['dismissals'],
            'team1_bat_vs_team2_bowl_sr': team1_bat_vs_team2_bowl['strike_rate'],
            'team1_bat_vs_team2_bowl_avg': team1_bat_vs_team2_bowl['average'],
            'team1_bat_vs_team2_bowl_economy': team1_bat_vs_team2_bowl['economy'],
            'team1_bat_vs_team2_bowl_frequency': team1_bat_vs_team2_bowl['frequency'],
            
            'team2_bat_vs_team1_bowl_runs': team2_bat_vs_team1_bowl['runs'],
            'team2_bat_vs_team1_bowl_balls': team2_bat_vs_team1_bowl['balls'],
            'team2_bat_vs_team1_bowl_dismissals': team2_bat_vs_team1_bowl['dismissals'],
            'team2_bat_vs_team1_bowl_sr': team2_bat_vs_team1_bowl['strike_rate'],
            'team2_bat_vs_team1_bowl_avg': team2_bat_vs_team1_bowl['average'],
            'team2_bat_vs_team1_bowl_economy': team2_bat_vs_team1_bowl['economy'],
            'team2_bat_vs_team1_bowl_frequency': team2_bat_vs_team1_bowl['frequency']
        })


    return features_df

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    print("\nTraining and evaluating multiple models...")
    
    # Dictionary to store model results
    model_results = {}
    
    # 1. Random Forest
    from sklearn.ensemble import RandomForestClassifier
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_scores = {
        'accuracy': accuracy_score(y_test, rf_pred),
        'precision': precision_score(y_test, rf_pred, average='weighted'),
        'recall': recall_score(y_test, rf_pred, average='weighted'),
        'f1': f1_score(y_test, rf_pred, average='weighted')
    }
    model_results['Random Forest'] = rf_scores
    
    # 2. XGBoost
    import xgboost as xgb
    xgb_model = xgb.XGBClassifier(random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_scores = {
        'accuracy': accuracy_score(y_test, xgb_pred),
        'precision': precision_score(y_test, xgb_pred, average='weighted'),
        'recall': recall_score(y_test, xgb_pred, average='weighted'),
        'f1': f1_score(y_test, xgb_pred, average='weighted')
    }
    model_results['XGBoost'] = xgb_scores
    
    # 3. Gradient Boosting
    from sklearn.ensemble import GradientBoostingClassifier
    gb_model = GradientBoostingClassifier(random_state=42)
    gb_model.fit(X_train, y_train)
    gb_pred = gb_model.predict(X_test)
    gb_scores = {
        'accuracy': accuracy_score(y_test, gb_pred),
        'precision': precision_score(y_test, gb_pred, average='weighted'),
        'recall': recall_score(y_test, gb_pred, average='weighted'),
        'f1': f1_score(y_test, gb_pred, average='weighted')
    }
    model_results['Gradient Boosting'] = gb_scores
    
    # 4. Support Vector Machine
    from sklearn.svm import SVC
    svm_model = SVC(random_state=42, probability=True)
    svm_model.fit(X_train, y_train)
    svm_pred = svm_model.predict(X_test)
    svm_scores = {
        'accuracy': accuracy_score(y_test, svm_pred),
        'precision': precision_score(y_test, svm_pred, average='weighted'),
        'recall': recall_score(y_test, svm_pred, average='weighted'),
        'f1': f1_score(y_test, svm_pred, average='weighted')
    }
    model_results['SVM'] = svm_scores
    
    # 5. Decision Tree
    from sklearn.tree import DecisionTreeClassifier
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)
    dt_pred = dt_model.predict(X_test)
    dt_scores = {
        'accuracy': accuracy_score(y_test, dt_pred),
        'precision': precision_score(y_test, dt_pred, average='weighted'),
        'recall': recall_score(y_test, dt_pred, average='weighted'),
        'f1': f1_score(y_test, dt_pred, average='weighted')
    }
    model_results['Decision Tree'] = dt_scores
    
    # 6. RNN (using Keras)
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import SimpleRNN, Dense
    rnn_model = Sequential([
        SimpleRNN(64, input_shape=(X_train.shape[1], 1)),
        Dense(32, activation='relu'),
        Dense(len(np.unique(y_train)), activation='softmax')
    ])
    rnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Reshape data for RNN
    X_train_rnn = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_rnn = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    rnn_model.fit(X_train_rnn, y_train, epochs=10, batch_size=32, verbose=0)
    rnn_pred = np.argmax(rnn_model.predict(X_test_rnn), axis=1)
    rnn_scores = {
        'accuracy': accuracy_score(y_test, rnn_pred),
        'precision': precision_score(y_test, rnn_pred, average='weighted'),
        'recall': recall_score(y_test, rnn_pred, average='weighted'),
        'f1': f1_score(y_test, rnn_pred, average='weighted')
    }
    model_results['RNN'] = rnn_scores
    
    # 7. LightGBM
    import lightgbm as lgb
    lgb_model = lgb.LGBMClassifier(random_state=42)
    lgb_model.fit(X_train, y_train)
    lgb_pred = lgb_model.predict(X_test)
    lgb_scores = {
        'accuracy': accuracy_score(y_test, lgb_pred),
        'precision': precision_score(y_test, lgb_pred, average='weighted'),
        'recall': recall_score(y_test, lgb_pred, average='weighted'),
        'f1': f1_score(y_test, lgb_pred, average='weighted')
    }
    model_results['LightGBM'] = lgb_scores
    
    # Compare models
    print("\nModel Comparison:")
    comparison_df = pd.DataFrame(model_results).round(4)
    print(comparison_df)
    
    # Visualize model comparison
    plt.figure(figsize=(12, 6))
    comparison_df.plot(kind='bar')
    plt.title('Model Performance Comparison')
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    
    # Find best model
    best_model = max(model_results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\nBest performing model: {best_model[0]}")
    print(f"Accuracy: {best_model[1]['accuracy']:.4f}")
    
    return model_results, best_model[0]

# 4. Model Training
def train_model(features_df):
    print("\nTraining prediction models...")
    
    # Filter out matches with no winner
    features_df = features_df.dropna(subset=['winner'])
    
    # Create target variable
    features_df['target'] = (features_df['winner'] == features_df['team1']).astype(int)
    
    # Split data by season for time-based validation
    train_data = features_df[features_df['season'] < 2024]
    validation_data = features_df[features_df['season'] == 2024]
    
    if validation_data.empty:
        # If no 2024 data available, use the latest available season for validation
        max_season = features_df['season'].max()
        validation_data = features_df[features_df['season'] == max_season]
        train_data = features_df[features_df['season'] < max_season]
    
    # Initialize feature columns list with all features
    feature_columns = [
        'total_matches_between',
        'team1_wins_against_team2',
        'team2_wins_against_team1',
        'team1_win_percentage_against_team2',
        'team2_win_percentage_against_team1',
        'venue_total_matches',
        'venue_avg_first_innings_score',
        'venue_batting_first_win_percentage',
        'team1_recent_form',
        'team2_recent_form',
        'team1_win_percentage_over_seasons',
        'team2_win_percentage_over_seasons',
        'team1_win_variance_over_seasons',
        'team2_win_variance_over_seasons',
        'team1_consistent_performance',
        'team2_consistent_performance',
        'team1_bat_vs_team2_bowl_runs',
        'team1_bat_vs_team2_bowl_balls',
        'team1_bat_vs_team2_bowl_dismissals',
        'team1_bat_vs_team2_bowl_sr',
        'team1_bat_vs_team2_bowl_avg',
        'team1_bat_vs_team2_bowl_economy',
        'team1_bat_vs_team2_bowl_frequency',
        'team2_bat_vs_team1_bowl_runs',
        'team2_bat_vs_team1_bowl_balls',
        'team2_bat_vs_team1_bowl_dismissals',
        'team2_bat_vs_team1_bowl_sr',
        'team2_bat_vs_team1_bowl_avg',
        'team2_bat_vs_team1_bowl_economy',
        'team2_bat_vs_team1_bowl_frequency',
        
        'team1_current_streak_length',
        'team2_current_streak_length',
        'team1_avg_max_win_streak',
        'team2_avg_max_win_streak',
        'team1_avg_max_loss_streak',
        'team2_avg_max_loss_streak',
        'team1_pattern_consistency',
        'team2_pattern_consistency'
    ]
    
    # Add batter-bowler features
    feature_columns.extend([
        'team1_bat_vs_team2_bowl_runs', 'team1_bat_vs_team2_bowl_balls',
        'team1_bat_vs_team2_bowl_dismissals', 'team1_bat_vs_team2_bowl_sr',
        'team1_bat_vs_team2_bowl_avg', 'team1_bat_vs_team2_bowl_economy',
        'team1_bat_vs_team2_bowl_frequency',
        
        'team2_bat_vs_team1_bowl_runs', 'team2_bat_vs_team1_bowl_balls',
        'team2_bat_vs_team1_bowl_dismissals', 'team2_bat_vs_team1_bowl_sr',
        'team2_bat_vs_team1_bowl_avg', 'team2_bat_vs_team1_bowl_economy',
        'team2_bat_vs_team1_bowl_frequency',
        
    ])
    
    # Add categorical variables
    categorical_columns = ['team1', 'team2', 'venue', 'toss_winner', 'toss_decision']
    
    # Initialize list to store all encoded feature names
    encoded_features = []
    
    # One-hot encode categorical features
    for col in categorical_columns:
        # Get all unique values from both train and validation sets
        all_values = pd.concat([train_data[col], validation_data[col]]).unique()
        
        # Create dummy variables for both sets using the same categories
        train_dummies = pd.get_dummies(train_data[col], prefix=col)
        val_dummies = pd.get_dummies(validation_data[col], prefix=col)
        
        # Add missing columns to each set
        for value in all_values:
            dummy_col = f"{col}_{value}"
            if dummy_col not in train_dummies.columns:
                train_dummies[dummy_col] = 0
            if dummy_col not in val_dummies.columns:
                val_dummies[dummy_col] = 0
        
        # Sort columns to ensure same order
        dummy_cols = sorted(train_dummies.columns)
        train_dummies = train_dummies[dummy_cols]
        val_dummies = val_dummies[dummy_cols]
        
        # Add to encoded features list (excluding first dummy to avoid multicollinearity)
        encoded_features.extend(dummy_cols[1:])
        
        # Add encoded features to respective datasets
        train_data = pd.concat([train_data, train_dummies], axis=1)
        validation_data = pd.concat([validation_data, val_dummies], axis=1)
    
    # Combine all features
    all_features = feature_columns + encoded_features
    
    # Prepare final feature matrices and ensure proper data types
    X_train = train_data[all_features].astype(float)
    y_train = train_data['target'].astype(int)
    X_val = validation_data[all_features].astype(float)
    y_val = validation_data['target'].astype(int)
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    # Train and evaluate multiple models
    model_results, best_model_name = train_and_evaluate_models(X_train, X_val, y_train, y_val)
    
    # Get the best model
    if best_model_name == 'Gradient Boosting':
        best_model = GradientBoostingClassifier(random_state=42)
    elif best_model_name == 'SVM':
        best_model = SVC(random_state=42, probability=True)
    elif best_model_name == 'Decision Tree':
        best_model = DecisionTreeClassifier(random_state=42)
    elif best_model_name == 'RNN':
        best_model = Sequential([
            SimpleRNN(64, input_shape=(X_train.shape[1], 1)),
            Dense(32, activation='relu'),
            Dense(len(np.unique(y_train)), activation='softmax')
        ])
    elif best_model_name == 'GNN':
        best_model = GNNModel(X_train.shape[1], 64, len(np.unique(y_train)))
    else:  # LightGBM
        best_model = lgb.LGBMClassifier(random_state=42)
    
    # Train the best model on the full training data
    best_model.fit(X_train, y_train)
    
    return best_model, all_features

# 5. Tournament Simulation
def simulate_ipl_2025(model, features_df, common_features, teams, venues):
    print("\nSimulating IPL 2025 Tournament...")
    
    # Create all possible match combinations for IPL 2025
    ipl_2025_matches = []
    for i, team1 in enumerate(teams):
        for team2 in teams[i+1:]:
            # Each team plays each other twice (home and away)
            for venue in [venues[team1], venues[team2]]:
                # Create match data
                match = {
                    'season': 2025,
                    'team1': team1,
                    'team2': team2,
                    'venue': venue
                }
                ipl_2025_matches.append(match)
    
    # Convert to DataFrame
    ipl_2025_df = pd.DataFrame(ipl_2025_matches)
    
    # Add features similar to training data
    for i, match in tqdm(ipl_2025_df.iterrows(), total=ipl_2025_df.shape[0], desc="Preparing 2025 matches"):
        team1 = match['team1']
        team2 = match['team2']
        venue = match['venue']
        
        # Set default values for features
        ipl_2025_df.loc[i, 'total_matches_between'] = features_df[(
            ((features_df['team1'] == team1) & (features_df['team2'] == team2)) | 
            ((features_df['team1'] == team2) & (features_df['team2'] == team1))
        )].shape[0]
        
        # Team1 vs Team2 stats
        team1_wins = features_df[
            (((features_df['team1'] == team1) & (features_df['team2'] == team2)) | 
             ((features_df['team1'] == team2) & (features_df['team2'] == team1))) & 
            (features_df['winner'] == team1)
        ].shape[0]
        
        team2_wins = features_df[
            (((features_df['team1'] == team1) & (features_df['team2'] == team2)) | 
             ((features_df['team1'] == team2) & (features_df['team2'] == team1))) & 
            (features_df['winner'] == team2)
        ].shape[0]
        
        total_matches = team1_wins + team2_wins
        
        ipl_2025_df.loc[i, 'team1_wins_against_team2'] = team1_wins
        ipl_2025_df.loc[i, 'team2_wins_against_team1'] = team2_wins
        ipl_2025_df.loc[i, 'team1_win_percentage_against_team2'] = (team1_wins / total_matches * 100) if total_matches > 0 else 50
        ipl_2025_df.loc[i, 'team2_win_percentage_against_team1'] = (team2_wins / total_matches * 100) if total_matches > 0 else 50
        
        # Venue stats
        venue_matches = features_df[features_df['venue'] == venue]
        venue_total_matches = venue_matches.shape[0]
        
        ipl_2025_df.loc[i, 'venue_total_matches'] = venue_total_matches
        
        avg_first_innings = features_df[features_df['venue'] == venue]['venue_avg_first_innings_score'].mean()
        ipl_2025_df.loc[i, 'venue_avg_first_innings_score'] = avg_first_innings if not np.isnan(avg_first_innings) else 160
        
        bat_first_win_pct = features_df[features_df['venue'] == venue]['venue_batting_first_win_percentage'].mean()
        ipl_2025_df.loc[i, 'venue_batting_first_win_percentage'] = bat_first_win_pct if not np.isnan(bat_first_win_pct) else 50
        
        # Recent form - use last season's performance
        team1_last_season = features_df[
            (features_df['season'] == 2024) & 
            ((features_df['team1'] == team1) | (features_df['team2'] == team1))
        ]
        
        team2_last_season = features_df[
            (features_df['season'] == 2024) & 
            ((features_df['team1'] == team2) | (features_df['team2'] == team2))
        ]
        
        team1_wins_last_season = team1_last_season[team1_last_season['winner'] == team1].shape[0]
        team2_wins_last_season = team2_last_season[team2_last_season['winner'] == team2].shape[0]
        
        ipl_2025_df.loc[i, 'team1_recent_form'] = team1_wins_last_season / len(team1_last_season) if len(team1_last_season) > 0 else 0.5
        ipl_2025_df.loc[i, 'team2_recent_form'] = team2_wins_last_season / len(team2_last_season) if len(team2_last_season) > 0 else 0.5
        
        # Randomly assign toss winner and decision for simulation
        toss_winner = np.random.choice([team1, team2])
        toss_decision = np.random.choice(['bat', 'field'])
        
        ipl_2025_df.loc[i, 'toss_winner'] = toss_winner
        ipl_2025_df.loc[i, 'toss_decision'] = toss_decision
    
    # Perform one-hot encoding similar to training data
    categorical_columns = ['team1', 'team2', 'venue', 'toss_winner', 'toss_decision']
    
    for col in categorical_columns:
        dummies = pd.get_dummies(ipl_2025_df[col], prefix=col, drop_first=True)
        ipl_2025_df = pd.concat([ipl_2025_df, dummies], axis=1)
    
    # Ensure all required features exist
    for feature in common_features:
        if feature not in ipl_2025_df.columns:
            ipl_2025_df[feature] = 0
    
    # Prepare features for prediction
    X_2025 = ipl_2025_df[common_features].astype(float)
    
    # Scale features using the same scaler used in training
    scaler = StandardScaler()
    X_2025_scaled = scaler.fit_transform(X_2025)
    
    # Make predictions
    predictions = model.predict_proba(X_2025_scaled)
    
    # Add predictions to dataframe
    ipl_2025_df['team1_win_prob'] = predictions[:, 1]
    ipl_2025_df['team2_win_prob'] = predictions[:, 0]
    
    # Simulate league stage
    num_simulations = 1000
    team_standings = {team: {'points': 0, 'nrr': 0} for team in teams}
    
    for sim in tqdm(range(num_simulations), desc="Running simulations"):
        # Reset standings for this simulation
        for team in teams:
            team_standings[team] = {'points': 0, 'nrr': 0}
        
        # Simulate each match
        for idx, match in ipl_2025_df.iterrows():
            team1 = match['team1']
            team2 = match['team2']
            team1_win_prob = match['team1_win_prob']
            
            # Determine winner based on probability
            if np.random.random() < team1_win_prob:
                winner = team1
                # Add points and NRR
                team_standings[team1]['points'] += 2
                team_standings[team1]['nrr'] += np.random.uniform(0.1, 0.5)
                team_standings[team2]['nrr'] -= np.random.uniform(0.1, 0.5)
            else:
                winner = team2
                # Add points and NRR
                team_standings[team2]['points'] += 2
                team_standings[team2]['nrr'] += np.random.uniform(0.1, 0.5)
                team_standings[team1]['nrr'] -= np.random.uniform(0.1, 0.5)
        
        # Sort teams by points and NRR
        sorted_teams = sorted(team_standings.items(), 
                             key=lambda x: (x[1]['points'], x[1]['nrr']), 
                             reverse=True)
        
        # Record playoff teams
        playoff_teams = [team for team, _ in sorted_teams[:4]]
        
        # Simulate playoffs
        # Qualifier 1: Team 1 vs Team 2
        team1 = playoff_teams[0]
        team2 = playoff_teams[1]
        team1_win_prob = estimate_win_probability(team1, team2, ipl_2025_df)
        
        if np.random.random() < team1_win_prob:
            finalist1 = team1
            qualifier2_team1 = team2
        else:
            finalist1 = team2
            qualifier2_team1 = team1
        
        # Eliminator: Team 3 vs Team 4
        team3 = playoff_teams[2]
        team4 = playoff_teams[3]
        team3_win_prob = estimate_win_probability(team3, team4, ipl_2025_df)
        
        if np.random.random() < team3_win_prob:
            qualifier2_team2 = team3
            eliminated = team4
        else:
            qualifier2_team2 = team4
            eliminated = team3
        
        # Qualifier 2: Loser of Qualifier 1 vs Winner of Eliminator
        q2_team1_win_prob = estimate_win_probability(qualifier2_team1, qualifier2_team2, ipl_2025_df)
        
        if np.random.random() < q2_team1_win_prob:
            finalist2 = qualifier2_team1
            second_runnerup = qualifier2_team2
        else:
            finalist2 = qualifier2_team2
            second_runnerup = qualifier2_team1
        
        # Final
        final_team1_win_prob = estimate_win_probability(finalist1, finalist2, ipl_2025_df)
        
        if np.random.random() < final_team1_win_prob:
            winner = finalist1
            runnerup = finalist2
        else:
            winner = finalist2
            runnerup = finalist1
    
    # Calculate final predictions based on simulation results
    return team_standings, playoff_teams

def estimate_win_probability(team1, team2, matches_df):
    """Estimate win probability based on previous matches"""
    # Find matches between these teams
    team_matches = matches_df[
        ((matches_df['team1'] == team1) & (matches_df['team2'] == team2)) |
        ((matches_df['team1'] == team2) & (matches_df['team2'] == team1))
    ]
    
    if team_matches.empty:
        # No previous matches, return default probability
        return 0.5
    
    # Calculate probability based on team1_win_prob or team2_win_prob
    if team_matches.iloc[0]['team1'] == team1:
        return team_matches['team1_win_prob'].mean()
    else:
        return 1 - team_matches['team2_win_prob'].mean()



def run_ipl_analysis():
    # Load data including points table
    matches_df, deliveries_df, points_table_df = load_data()
    
    # Engineer features
    features_df = engineer_features(matches_df, deliveries_df)
    
    # Train model
    best_model, results = train_model(features_df)
    
    # Get playoff predictions using actual points table
    current_season_data = analyze_current_season(matches_df, points_table_df)
    playoff_teams = predict_playoffs(matches_df, best_model, features_df, points_table_df)

    # Clean up standings for safe serialization
    current_standings = [
        {
            "Team": str(row["Team"]),
            "Points": int(row["Points"]),
            "NRR": round(float(row["NRR"]), 3)
        }
        for row in points_table_df.to_dict(orient="records")
    ]
    
    # Prepare response dictionary
    analysis = {
        "Current Standings": current_standings,
        "Team Analysis": {},
        "Eliminated Teams": [str(team) for team in current_season_data['eliminated_teams']],
        "Predicted Playoff Teams": [str(team) for team in playoff_teams]
    }
    
    # Populate team-specific data (safely convert all types)
    for team in points_table_df['Team']:
        if team in current_season_data['remaining_matches']:
            data = current_season_data['remaining_matches'][team]
            analysis["Team Analysis"][str(team)] = {
                "Current Points": int(data['current_points']),
                "NRR": round(float(data['nrr']), 3),
                "Remaining Matches": int(data['matches_remaining']),
                "Max Possible Points": int(data['max_possible_points']),
                "Upcoming Opponents": [str(opponent) for opponent in data['opponents']]
            }

    # Ensure all data is JSON-serializable
    return jsonable_encoder(analysis)
