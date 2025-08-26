#!/usr/bin/env python3
"""
Fantasy Draft Data Trainer
Scrapes real draft data and trains ML models to improve draft recommendations
"""

import requests
import pandas as pd
import numpy as np
import json
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import sqlite3
from datetime import datetime, timedelta
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DraftPick:
    round_num: int
    pick_num: int
    player_name: str
    position: str
    team: str
    adp: float
    projection: float
    drafted_by_team: int
    league_size: int
    scoring_format: str
    draft_date: str
    available_players: List[str] = field(default_factory=list)
    roster_before_pick: List[str] = field(default_factory=list)


class DraftDataCollector:
    """Collects draft data from multiple sources."""

    def __init__(self, db_path: str = "draft_data.db"):
        self.db_path = db_path
        self.setup_database()

    def setup_database(self):
        """Create SQLite database to store draft data."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS draft_picks
                       (
                           id
                           INTEGER
                           PRIMARY
                           KEY
                           AUTOINCREMENT,
                           round_num
                           INTEGER,
                           pick_num
                           INTEGER,
                           player_name
                           TEXT,
                           position
                           TEXT,
                           team
                           TEXT,
                           adp
                           REAL,
                           projection
                           REAL,
                           drafted_by_team
                           INTEGER,
                           league_size
                           INTEGER,
                           scoring_format
                           TEXT,
                           draft_date
                           TEXT,
                           available_players
                           TEXT,
                           roster_before_pick
                           TEXT
                       )
                       ''')

        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS player_projections
                       (
                           id
                           INTEGER
                           PRIMARY
                           KEY
                           AUTOINCREMENT,
                           player_name
                           TEXT,
                           position
                           TEXT,
                           team
                           TEXT,
                           week
                           INTEGER,
                           season
                           INTEGER,
                           projection
                           REAL,
                           actual_points
                           REAL,
                           source
                           TEXT,
                           date_updated
                           TEXT
                       )
                       ''')

        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS draft_metadata
                       (
                           id
                           INTEGER
                           PRIMARY
                           KEY
                           AUTOINCREMENT,
                           draft_id
                           TEXT
                           UNIQUE,
                           league_size
                           INTEGER,
                           scoring_format
                           TEXT,
                           draft_date
                           TEXT,
                           platform
                           TEXT,
                           total_picks
                           INTEGER
                       )
                       ''')

        conn.commit()
        conn.close()

    def scrape_sleeper_drafts(self, limit: int = 100) -> List[DraftPick]:
        """Scrape recent draft data from Sleeper API."""
        logger.info(f"Scraping {limit} Sleeper drafts...")

        all_picks = []

        # Get recent drafts
        try:
            # Sleeper doesn't have a public "recent drafts" endpoint
            # So we'll use a different approach - get drafts from popular leagues

            # First, get some league IDs (you'd need to collect these manually or from other sources)
            sample_league_ids = [
                "784462448355590144",  # Example league IDs
                "784462448355590145",
                # Add more league IDs here
            ]

            for league_id in sample_league_ids[:limit // 10]:  # Limit API calls
                draft_data = self.get_sleeper_draft_data(league_id)
                if draft_data:
                    picks = self.parse_sleeper_draft(draft_data)
                    all_picks.extend(picks)
                    time.sleep(1)  # Rate limiting

        except Exception as e:
            logger.error(f"Error scraping Sleeper: {e}")

        return all_picks

    def get_sleeper_draft_data(self, league_id: str) -> Optional[Dict]:
        """Get draft data for a specific Sleeper league."""
        try:
            # Get drafts for league
            drafts_url = f"https://api.sleeper.app/v1/league/{league_id}/drafts"
            response = requests.get(drafts_url)

            if response.status_code != 200:
                return None

            drafts = response.json()
            if not drafts:
                return None

            # Get the most recent draft
            draft_id = drafts[0]['draft_id']

            # Get draft picks
            picks_url = f"https://api.sleeper.app/v1/draft/{draft_id}/picks"
            picks_response = requests.get(picks_url)

            if picks_response.status_code != 200:
                return None

            return {
                'draft_info': drafts[0],
                'picks': picks_response.json()
            }

        except Exception as e:
            logger.error(f"Error getting Sleeper draft {league_id}: {e}")
            return None

    def parse_sleeper_draft(self, draft_data: Dict) -> List[DraftPick]:
        """Parse Sleeper draft data into DraftPick objects."""
        picks = []

        try:
            draft_info = draft_data['draft_info']
            pick_data = draft_data['picks']

            # Get player info
            players_response = requests.get("https://api.sleeper.app/v1/players/nfl")
            if players_response.status_code == 200:
                players = players_response.json()
            else:
                players = {}

            # Track rosters for each team
            rosters = {i: [] for i in range(draft_info.get('settings', {}).get('teams', 12))}

            for pick in pick_data:
                player_id = pick.get('player_id')
                if not player_id or player_id not in players:
                    continue

                player = players[player_id]
                team_id = pick.get('picked_by', 0) - 1  # Convert to 0-indexed

                # Get available players at time of pick (simplified)
                available_players = [p['full_name'] for p in players.values()
                                     if p.get('position') in ['QB', 'RB', 'WR', 'TE', 'K', 'DEF']]

                draft_pick = DraftPick(
                    round_num=pick.get('round', 0),
                    pick_num=pick.get('pick_no', 0),
                    player_name=player.get('full_name', ''),
                    position=player.get('position', ''),
                    team=player.get('team', ''),
                    adp=0.0,  # We'll calculate this later
                    projection=0.0,  # We'll get this from other sources
                    drafted_by_team=team_id,
                    league_size=draft_info.get('settings', {}).get('teams', 12),
                    scoring_format=self.parse_sleeper_scoring(draft_info.get('scoring_settings', {})),
                    draft_date=datetime.now().isoformat(),
                    available_players=available_players[:50],  # Limit size
                    roster_before_pick=rosters[team_id].copy()
                )

                picks.append(draft_pick)
                rosters[team_id].append(player.get('full_name', ''))

        except Exception as e:
            logger.error(f"Error parsing Sleeper draft: {e}")

        return picks

    def parse_sleeper_scoring(self, scoring_settings: Dict) -> str:
        """Parse Sleeper scoring settings to determine format."""
        rec_points = scoring_settings.get('rec', 0)
        if rec_points >= 1.0:
            return 'PPR'
        elif rec_points >= 0.5:
            return 'Half PPR'
        else:
            return 'Standard'

    def scrape_fantasypros_adp(self) -> Dict[str, float]:
        """Scrape ADP data from FantasyPros."""
        logger.info("Scraping FantasyPros ADP...")

        adp_data = {}

        try:
            # FantasyPros ADP URL (you might need to adjust this)
            url = "https://www.fantasypros.com/nfl/adp/overall.php"

            # Note: This is a simplified example. You'd need to handle:
            # - Different scoring formats
            # - Different time periods
            # - Proper web scraping with headers, etc.

            response = requests.get(url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })

            if response.status_code == 200:
                # Parse HTML (you'd use BeautifulSoup in reality)
                # This is just a placeholder

                # Sample data structure
                sample_adp = {
                    'Christian McCaffrey': 1.2,
                    'Austin Ekeler': 3.4,
                    'Tyreek Hill': 2.1,
                    # ... more players
                }

                adp_data.update(sample_adp)

        except Exception as e:
            logger.error(f"Error scraping FantasyPros ADP: {e}")

        return adp_data

    def get_historical_projections(self, season: int = 2023) -> Dict[str, float]:
        """Get historical projections for players."""
        logger.info(f"Getting projections for {season} season...")

        # This would typically come from:
        # - FantasyPros historical data
        # - ESPN/Yahoo APIs
        # - Your own projection models

        projections = {
            'Christian McCaffrey': 285.5,
            'Austin Ekeler': 268.3,
            'Tyreek Hill': 245.8,
            'Cooper Kupp': 235.6,
            # ... more players
        }

        return projections

    def save_picks_to_db(self, picks: List[DraftPick]):
        """Save draft picks to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for pick in picks:
            cursor.execute('''
                           INSERT
                           OR IGNORE INTO draft_picks 
                (round_num, pick_num, player_name, position, team, adp, projection,
                 drafted_by_team, league_size, scoring_format, draft_date,
                 available_players, roster_before_pick)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                           ''', (
                               pick.round_num, pick.pick_num, pick.player_name, pick.position,
                               pick.team, pick.adp, pick.projection, pick.drafted_by_team,
                               pick.league_size, pick.scoring_format, pick.draft_date,
                               json.dumps(pick.available_players), json.dumps(pick.roster_before_pick)
                           ))

        conn.commit()
        conn.close()
        logger.info(f"Saved {len(picks)} picks to database")


class DraftPredictor:
    """ML model to predict draft picks and outcomes."""

    def __init__(self, db_path: str = "draft_data.db"):
        self.db_path = db_path
        self.pick_predictor = None
        self.value_predictor = None
        self.label_encoders = {}

    def load_training_data(self) -> pd.DataFrame:
        """Load draft data from database for training."""
        conn = sqlite3.connect(self.db_path)

        query = '''
                SELECT * \
                FROM draft_picks
                WHERE player_name != '' AND position != ''
                ORDER BY draft_date, round_num, pick_num \
                '''

        df = pd.read_sql_query(query, conn)
        conn.close()

        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features for ML model."""
        logger.info("Engineering features...")

        # Basic features
        df['pick_overall'] = (df['round_num'] - 1) * df['league_size'] + df['pick_num']
        df['is_snake_draft'] = True  # Assume snake draft

        # Position scarcity features
        df['qb_taken_so_far'] = df.groupby(['draft_date'])['position'].apply(
            lambda x: (x == 'QB').cumsum() - (x == 'QB')
        ).values
        df['rb_taken_so_far'] = df.groupby(['draft_date'])['position'].apply(
            lambda x: (x == 'RB').cumsum() - (x == 'RB')
        ).values
        df['wr_taken_so_far'] = df.groupby(['draft_date'])['position'].apply(
            lambda x: (x == 'WR').cumsum() - (x == 'WR')
        ).values
        df['te_taken_so_far'] = df.groupby(['draft_date'])['position'].apply(
            lambda x: (x == 'TE').cumsum() - (x == 'TE')
        ).values

        # Team need features (simplified)
        # In reality, you'd track each team's roster composition
        df['team_qb_count'] = 0  # Placeholder
        df['team_rb_count'] = 0  # Placeholder
        df['team_wr_count'] = 0  # Placeholder
        df['team_te_count'] = 0  # Placeholder

        # Draft position features
        df['is_early_draft_position'] = (df['drafted_by_team'] <= 4).astype(int)
        df['is_late_draft_position'] = (df['drafted_by_team'] >= 9).astype(int)

        # Round-based features
        df['is_early_round'] = (df['round_num'] <= 3).astype(int)
        df['is_middle_round'] = ((df['round_num'] > 3) & (df['round_num'] <= 10)).astype(int)
        df['is_late_round'] = (df['round_num'] > 10).astype(int)

        return df

    def train_pick_predictor(self, df: pd.DataFrame):
        """Train model to predict which player will be picked."""
        logger.info("Training pick prediction model...")

        # Prepare features
        feature_columns = [
            'round_num', 'pick_overall', 'adp', 'projection', 'league_size',
            'qb_taken_so_far', 'rb_taken_so_far', 'wr_taken_so_far', 'te_taken_so_far',
            'is_early_draft_position', 'is_late_draft_position',
            'is_early_round', 'is_middle_round', 'is_late_round'
        ]

        # Encode categorical variables
        for col in ['position', 'scoring_format']:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col])
            feature_columns.append(f'{col}_encoded')

        X = df[feature_columns].fillna(0)
        y = df['player_name']  # Predict player name (or player ID)

        # For simplicity, let's predict position instead of specific player
        y_position = df['position']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_position, test_size=0.2, random_state=42
        )

        # Train Random Forest
        self.pick_predictor = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )

        self.pick_predictor.fit(X_train, y_train)

        # Evaluate
        y_pred = self.pick_predictor.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Pick prediction accuracy: {accuracy:.3f}")

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': self.pick_predictor.feature_importances_
        }).sort_values('importance', ascending=False)

        logger.info("Top 5 most important features:")
        print(feature_importance.head())

    def train_value_predictor(self, df: pd.DataFrame):
        """Train model to predict player value/success."""
        logger.info("Training value prediction model...")

        # For this example, we'll use projection as target
        # In reality, you'd use end-of-season fantasy points

        feature_columns = [
            'round_num', 'pick_overall', 'adp', 'league_size',
            'qb_taken_so_far', 'rb_taken_so_far', 'wr_taken_so_far', 'te_taken_so_far',
            'position_encoded', 'scoring_format_encoded'
        ]

        X = df[feature_columns].fillna(0)
        y = df['projection'].fillna(df['projection'].mean())

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train Gradient Boosting
        self.value_predictor = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=6,
            random_state=42
        )

        self.value_predictor.fit(X_train, y_train)

        # Evaluate
        y_pred = self.value_predictor.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        logger.info(f"Value prediction MSE: {mse:.3f}")

    def save_models(self, models_dir: str = "models"):
        """Save trained models."""
        import os
        os.makedirs(models_dir, exist_ok=True)

        if self.pick_predictor:
            with open(f"{models_dir}/pick_predictor.pkl", "wb") as f:
                pickle.dump(self.pick_predictor, f)

        if self.value_predictor:
            with open(f"{models_dir}/value_predictor.pkl", "wb") as f:
                pickle.dump(self.value_predictor, f)

        with open(f"{models_dir}/label_encoders.pkl", "wb") as f:
            pickle.dump(self.label_encoders, f)

        logger.info(f"Models saved to {models_dir}/")

    def load_models(self, models_dir: str = "models"):
        """Load trained models."""
        try:
            with open(f"{models_dir}/pick_predictor.pkl", "rb") as f:
                self.pick_predictor = pickle.load(f)

            with open(f"{models_dir}/value_predictor.pkl", "rb") as f:
                self.value_predictor = pickle.load(f)

            with open(f"{models_dir}/label_encoders.pkl", "rb") as f:
                self.label_encoders = pickle.load(f)

            logger.info("Models loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False

    def predict_next_picks(self, current_state: Dict) -> List[Tuple[str, float]]:
        """Predict what other teams will pick next."""
        if not self.pick_predictor:
            return []

        # This would use the current draft state to predict next picks
        # Implementation depends on your specific use case

        predictions = [
            ('RB', 0.35),
            ('WR', 0.40),
            ('QB', 0.15),
            ('TE', 0.10)
        ]

        return predictions


def create_sample_data():
    """Create sample draft data for testing."""
    sample_picks = [
        DraftPick(1, 1, "Christian McCaffrey", "RB", "SF", 1.2, 285.5, 0, 12, "PPR", "2023-08-15"),
        DraftPick(1, 2, "Austin Ekeler", "RB", "LAC", 3.4, 268.3, 1, 12, "PPR", "2023-08-15"),
        DraftPick(1, 3, "Tyreek Hill", "WR", "MIA", 2.1, 245.8, 2, 12, "PPR", "2023-08-15"),
        # Add more sample data...
    ]

    return sample_picks


def main():
    """Main training pipeline."""
    print("Fantasy Draft Data Trainer")
    print("=" * 50)

    # Initialize data collector
    collector = DraftDataCollector()

    # Option 1: Collect real data (requires API access)
    try:
        print("Collecting draft data...")
        # picks = collector.scrape_sleeper_drafts(limit=10)
        # collector.save_picks_to_db(picks)

        # For demo, use sample data
        sample_picks = create_sample_data()
        collector.save_picks_to_db(sample_picks)
        print(f"Saved {len(sample_picks)} sample picks")

    except Exception as e:
        print(f"Error collecting data: {e}")
        print("Using sample data instead...")
        sample_picks = create_sample_data()
        collector.save_picks_to_db(sample_picks)

    # Train models
    predictor = DraftPredictor()

    print("\nLoading training data...")
    df = predictor.load_training_data()

    if len(df) == 0:
        print("No training data available!")
        return

    print(f"Loaded {len(df)} draft picks")

    # Engineer features
    df = predictor.engineer_features(df)

    # Train models
    print("\nTraining models...")
    predictor.train_pick_predictor(df)
    predictor.train_value_predictor(df)

    # Save models
    predictor.save_models()

    print("\nTraining complete!")
    print("Models saved to models/ directory")
    print("You can now use these models in your draft advisor app")


if __name__ == "__main__":
    main()