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
warnings.filterwarnings('ignore')

# 1. Data Loading and Initial Exploration
def load_data():
    print("Loading datasets...")
    matches_df = pd.read_csv('Final_Dataset.csv')
    deliveries_df = pd.read_csv('deliveries_cleaned.csv')
    
    print(f"Matches dataset shape: {matches_df.shape}")
    print(f"Deliveries dataset shape: {deliveries_df.shape}")
    
    # Convert date to datetime
    matches_df['date'] = pd.to_datetime(matches_df['date'])
    
    return matches_df, deliveries_df

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

# 3. Feature Engineering
# Import point table extractor
from point_table_extractor import extract_point_table, get_team_form_metrics, get_head_to_head_stats

def engineer_features(matches_df, deliveries_df):
    print("\nEngineering features...")
    
    # Create a copy of the dataframes
    matches = matches_df.copy()
    deliveries = deliveries_df.copy()
    
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
        if len(past_matches) > 0:
            win_rate = wins / len(past_matches)
        else:
            win_rate = 0
        
        return win_rate
    
    # Create a list to store match features
    match_features = []
    
    # Create features for each match
    for _, match in tqdm(matches.iterrows(), total=matches.shape[0], desc="Creating match features"):
        # Get current season point table data if match is from 2025
        if match['season'] == 2025:
            
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
         
            # Add current season performance metrics if available
            'team1_current_points': team1_standing['points'] if match['season'] == 2025 else 0,
            'team2_current_points': team2_standing['points'] if match['season'] == 2025 else 0,
            'team1_current_nrr': team1_standing['net_run_rate'] if match['season'] == 2025 else 0,
            'team2_current_nrr': team2_standing['net_run_rate'] if match['season'] == 2025 else 0,
        
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
           
            # Add current season performance metrics if available
            'team1_current_points': team1_standing['points'] if match['season'] == 2025 else 0,
            'team2_current_points': team2_standing['points'] if match['season'] == 2025 else 0,
            'team1_current_nrr': team1_standing['net_run_rate'] if match['season'] == 2025 else 0,
            'team2_current_nrr': team2_standing['net_run_rate'] if match['season'] == 2025 else 0,
            
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
   
    
    # 4. Support Vector Machine
  
    
    # 5. Decision Tree
   
    
    # 6. RNN (using Keras)
    
    
    
    
    # 7. LightGBM
    
    
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
    from sklearn.preprocessing import LabelEncoder
    # Define the columns to exclude from features
    exclude_cols = [
        'winner', 'match_id', 'team1', 'team2', 'season', 'venue', 
        'toss_winner', 'toss_decision'
    ]
    # Select only numeric columns and exclude the above
    common_features = [col for col in features_df.columns if col not in exclude_cols and features_df[col].dtype in [np.int64, np.float64, np.int32, np.float32]]
    
    X = features_df[common_features]
    y = features_df['winner']

    # Encode target labels as integers
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    # XGBoost model
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    xgb_model.fit(X_train, y_train)

    # Evaluate
    y_pred = xgb_model.predict(X_test)
    print("XGBoost Accuracy:", accuracy_score(y_test, y_pred))
    print("XGBoost Precision:", precision_score(y_test, y_pred, average='weighted'))
    print("XGBoost Recall:", recall_score(y_test, y_pred, average='weighted'))
    print("XGBoost F1 Score:", f1_score(y_test, y_pred, average='weighted'))

    return xgb_model, common_features
    # ... existing code ...

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

# 6. Main function
def main():
    print("IPL 2025 Prediction System")
    print("=========================")
    
    # Load data
    matches_df, deliveries_df = load_data()
    
    # Perform EDA
    perform_eda(matches_df, deliveries_df)
    
    # Engineer features
    features_df = engineer_features(matches_df, deliveries_df)
    
    # Train model
    model, common_features = train_model(features_df)
    
    # Define IPL 2025 teams
    teams = [
        'Mumbai Indians', 'Chennai Super Kings', 'Royal Challengers Bangalore',
        'Kolkata Knight Riders', 'Delhi Capitals', 'Sunrisers Hyderabad',
        'Punjab Kings', 'Rajasthan Royals', 'Gujarat Titans', 'Lucknow Super Giants'
    ]
    
    # Define home venues for teams
    venues = {
        'Mumbai Indians': 'Wankhede Stadium',
        'Chennai Super Kings': 'M.A. Chidambaram Stadium',
        'Royal Challengers Bangalore': 'M. Chinnaswamy Stadium',
        'Kolkata Knight Riders': 'Eden Gardens',
        'Delhi Capitals': 'Arun Jaitley Stadium',
        'Sunrisers Hyderabad': 'Rajiv Gandhi International Stadium',
        'Punjab Kings': 'Punjab Cricket Association Stadium',
        'Rajasthan Royals': 'Sawai Mansingh Stadium',
        'Gujarat Titans': 'Narendra Modi Stadium',
        'Lucknow Super Giants': 'Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium'
    }
    
    # Simulate IPL 2025
    team_standings, playoff_teams = simulate_ipl_2025(model, features_df, common_features, teams, venues)
    
    # Print predictions
    print("\nIPL 2025 Predictions:")
    print("=====================")
    
    # Convert team standings to DataFrame for better visualization
    standings_df = pd.DataFrame([
        {'Team': team, 'Points': stats['points'], 'NRR': stats['nrr']}
        for team, stats in team_standings.items()
    ])
    
    # Sort by points and NRR
    standings_df = standings_df.sort_values(['Points', 'NRR'], ascending=False).reset_index(drop=True)
    
    # Print final standings
    print("\nPredicted League Standings:")
    print(standings_df)
    
    # Print playoff teams
    print("\nPredicted Playoff Teams:")
    for i, team in enumerate(playoff_teams[:4], 1):
        print(f"{i}. {team}")
    
    # Print final predictions
    print("\nFinal Predictions:")
    print(f"Winner: {playoff_teams[0]}")
    print(f"Runner-up: {playoff_teams[1]}")
    print(f"Second Runner-up: {playoff_teams[2]}")
    print(f"Eliminator Participant: {playoff_teams[3]}")
    
    # Save predictions to CSV
    standings_df.to_csv('ipl_2025_predictions.csv', index=False)
    
    print("\nPredictions saved to 'ipl_2025_predictions.csv'")
    
    # Create visualization of predictions
    plt.figure(figsize=(12, 8))
    colors = ['gold', 'silver', '#CD7F32', 'royalblue']
    top_teams = standings_df.head(4)
    sns.barplot(x='Team', y='Points', data=top_teams, palette=colors)
    plt.title('IPL 2025 Predicted Top 4 Teams')
    plt.ylabel('Predicted Points')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('ipl_2025_top4_predictions.png')
    
    print("Visualization saved to 'ipl_2025_top4_predictions.png'")

if __name__ == "__main__":
    main()
    