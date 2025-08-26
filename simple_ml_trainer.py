#!/usr/bin/env python3
"""
Simple ML Trainer - Standalone
Creates training data and trains models for the fantasy draft advisor
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error
import warnings

warnings.filterwarnings('ignore')


def create_training_data():
    """Create realistic training data for ML models."""
    print("📊 Creating training data...")

    np.random.seed(42)  # For reproducible results

    # Player pool with realistic stats
    players_data = [
        # RBs
        ('Christian McCaffrey', 'RB', 'SF', 285.5, 1.2),
        ('Austin Ekeler', 'RB', 'LAC', 268.3, 3.4),
        ('Derrick Henry', 'RB', 'TEN', 255.1, 5.8),
        ('Nick Chubb', 'RB', 'CLE', 248.7, 7.3),
        ('Saquon Barkley', 'RB', 'NYG', 210.3, 12.8),
        ('Josh Jacobs', 'RB', 'LV', 205.1, 14.2),
        ('Joe Mixon', 'RB', 'CIN', 198.2, 16.4),
        ('Kenneth Walker III', 'RB', 'SEA', 190.8, 18.7),
        ('Tony Pollard', 'RB', 'DAL', 195.4, 10.5),

        # WRs
        ('Tyreek Hill', 'WR', 'MIA', 245.8, 2.1),
        ('Cooper Kupp', 'WR', 'LAR', 235.6, 4.2),
        ('Davante Adams', 'WR', 'LV', 228.4, 6.1),
        ('Stefon Diggs', 'WR', 'BUF', 222.9, 8.2),
        ('Ja\'Marr Chase', 'WR', 'CIN', 215.7, 11.2),
        ('CeeDee Lamb', 'WR', 'DAL', 208.9, 13.5),
        ('A.J. Brown', 'WR', 'PHI', 201.8, 15.3),
        ('Amon-Ra St. Brown', 'WR', 'DET', 195.6, 17.1),
        ('Mike Evans', 'WR', 'TB', 188.3, 19.2),
        ('Jaylen Waddle', 'WR', 'MIA', 185.9, 20.1),

        # QBs
        ('Josh Allen', 'QB', 'BUF', 310.5, 25.3),
        ('Patrick Mahomes', 'QB', 'KC', 305.2, 28.1),
        ('Lamar Jackson', 'QB', 'BAL', 298.7, 32.4),
        ('Jalen Hurts', 'QB', 'PHI', 290.3, 35.6),
        ('Joe Burrow', 'QB', 'CIN', 275.8, 45.2),

        # TEs
        ('Travis Kelce', 'TE', 'KC', 198.5, 9.1),
        ('Mark Andrews', 'TE', 'BAL', 165.4, 55.3),
        ('George Kittle', 'TE', 'SF', 158.7, 62.1),
        ('T.J. Hockenson', 'TE', 'MIN', 145.2, 78.4),
    ]

    training_data = []

    # Generate 200 mock drafts
    for draft_id in range(200):
        league_size = np.random.choice([10, 12, 14], p=[0.2, 0.7, 0.1])
        scoring = np.random.choice(['PPR', 'Half PPR', 'Standard'], p=[0.6, 0.3, 0.1])

        # Simulate draft order and picks
        draft_order = list(range(league_size))
        np.random.shuffle(draft_order)

        # Track positions drafted by each team
        team_rosters = {i: {'QB': 0, 'RB': 0, 'WR': 0, 'TE': 0} for i in range(league_size)}

        # Simulate 16 rounds
        for round_num in range(1, 17):
            pick_order = draft_order if round_num % 2 == 1 else draft_order[::-1]

            for pick_in_round, team_id in enumerate(pick_order):
                pick_overall = (round_num - 1) * league_size + pick_in_round + 1

                # Determine what position this team is likely to draft
                position = simulate_position_choice(round_num, team_rosters[team_id], scoring)

                # Find available players at this position
                available_players = [p for p in players_data if p[1] == position]
                if not available_players:
                    continue

                # Pick player (weighted by projection/ADP)
                player = pick_player_weighted(available_players, round_num)
                if not player:
                    continue

                # Count positions drafted so far (across all teams)
                total_qb = sum(sum(1 for r in training_data if r['round_num'] <= round_num and r['position'] == 'QB'))
                total_rb = sum(sum(1 for r in training_data if r['round_num'] <= round_num and r['position'] == 'RB'))
                total_wr = sum(sum(1 for r in training_data if r['round_num'] <= round_num and r['position'] == 'WR'))
                total_te = sum(sum(1 for r in training_data if r['round_num'] <= round_num and r['position'] == 'TE'))

                # Create training record
                record = {
                    'draft_id': draft_id,
                    'round_num': round_num,
                    'pick_overall': pick_overall,
                    'player_name': player[0],
                    'position': player[1],
                    'team': player[2],
                    'projection': player[3],
                    'adp': player[4],
                    'drafted_by_team': team_id,
                    'league_size': league_size,
                    'scoring_format': scoring,
                    # Position scarcity features
                    'qb_taken_so_far': total_qb,
                    'rb_taken_so_far': total_rb,
                    'wr_taken_so_far': total_wr,
                    'te_taken_so_far': total_te,
                    # Team need features
                    'team_qb_count': team_rosters[team_id]['QB'],
                    'team_rb_count': team_rosters[team_id]['RB'],
                    'team_wr_count': team_rosters[team_id]['WR'],
                    'team_te_count': team_rosters[team_id]['TE'],
                    # Draft position features
                    'is_early_draft_position': int(team_id <= 3),
                    'is_late_draft_position': int(team_id >= league_size - 3),
                    # Round features
                    'is_early_round': int(round_num <= 3),
                    'is_middle_round': int(3 < round_num <= 10),
                    'is_late_round': int(round_num > 10)
                }

                training_data.append(record)
                team_rosters[team_id][position] += 1

                # Remove player from available pool
                players_data = [p for p in players_data if p[0] != player[0]]

    return pd.DataFrame(training_data)


def simulate_position_choice(round_num, team_roster, scoring):
    """Simulate what position a team would draft based on round and needs."""

    # Early rounds: focus on RB/WR
    if round_num <= 3:
        if team_roster['RB'] < 2:
            return np.random.choice(['RB', 'WR'], p=[0.6, 0.4])
        elif team_roster['WR'] < 2:
            return np.random.choice(['WR', 'RB'], p=[0.6, 0.4])
        else:
            return np.random.choice(['RB', 'WR', 'TE'], p=[0.4, 0.4, 0.2])

    # Middle rounds: fill needs
    elif round_num <= 8:
        needs = []
        weights = []

        if team_roster['QB'] == 0:
            needs.extend(['QB'] * 3)
        if team_roster['RB'] < 2:
            needs.extend(['RB'] * 2)
        if team_roster['WR'] < 3:
            needs.extend(['WR'] * 2)
        if team_roster['TE'] == 0:
            needs.extend(['TE'] * 2)

        # Default to skill positions if no major needs
        if not needs:
            needs = ['RB', 'WR', 'QB', 'TE']

        return np.random.choice(needs)

    # Late rounds: depth and kicker/defense
    else:
        options = ['RB', 'WR', 'QB', 'TE', 'K', 'DST']
        weights = [0.25, 0.25, 0.15, 0.15, 0.1, 0.1]
        return np.random.choice(options, p=weights)


def pick_player_weighted(available_players, round_num):
    """Pick a player weighted by projection and round appropriateness."""
    if not available_players:
        return None

    # Sort by projection (descending)
    sorted_players = sorted(available_players, key=lambda x: x[3], reverse=True)

    # Early rounds: favor top players
    if round_num <= 5:
        return sorted_players[0] if sorted_players else None

    # Later rounds: add some randomness but still favor better players
    weights = [max(0.1, 1.0 - i * 0.1) for i in range(len(sorted_players))]
    weights = np.array(weights) / sum(weights)

    chosen_idx = np.random.choice(len(sorted_players), p=weights)
    return sorted_players[chosen_idx]


def train_models(df):
    """Train ML models on the generated data."""
    print("🧠 Training ML models...")

    # Features for training
    feature_columns = [
        'round_num', 'pick_overall', 'adp', 'projection', 'league_size',
        'qb_taken_so_far', 'rb_taken_so_far', 'wr_taken_so_far', 'te_taken_so_far',
        'team_qb_count', 'team_rb_count', 'team_wr_count', 'team_te_count',
        'is_early_draft_position', 'is_late_draft_position',
        'is_early_round', 'is_middle_round', 'is_late_round'
    ]

    # Encode categorical variables
    label_encoders = {}

    for col in ['position', 'scoring_format']:
        label_encoders[col] = LabelEncoder()
        df[f'{col}_encoded'] = label_encoders[col].fit_transform(df[col])
        feature_columns.append(f'{col}_encoded')

    # Prepare training data
    X = df[feature_columns].fillna(0)

    # Train position predictor
    print("  Training position prediction model...")
    y_position = df['position']

    X_train_pos, X_test_pos, y_train_pos, y_test_pos = train_test_split(
        X, y_position, test_size=0.2, random_state=42
    )

    pick_predictor = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )

    pick_predictor.fit(X_train_pos, y_train_pos)

    # Evaluate position predictor
    y_pred_pos = pick_predictor.predict(X_test_pos)
    pos_accuracy = accuracy_score(y_test_pos, y_pred_pos)
    print(f"    Position prediction accuracy: {pos_accuracy:.3f}")

    # Train value predictor
    print("  Training value prediction model...")
    y_value = df['projection']

    X_train_val, X_test_val, y_train_val, y_test_val = train_test_split(
        X, y_value, test_size=0.2, random_state=42
    )

    value_predictor = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=6,
        random_state=42
    )

    value_predictor.fit(X_train_val, y_train_val)

    # Evaluate value predictor
    y_pred_val = value_predictor.predict(X_test_val)
    val_mse = mean_squared_error(y_test_val, y_pred_val)
    print(f"    Value prediction MSE: {val_mse:.1f}")

    return pick_predictor, value_predictor, label_encoders


def save_models(pick_predictor, value_predictor, label_encoders, models_dir="models"):
    """Save trained models to disk."""
    print("💾 Saving models...")

    os.makedirs(models_dir, exist_ok=True)

    with open(f"{models_dir}/pick_predictor.pkl", "wb") as f:
        pickle.dump(pick_predictor, f)

    with open(f"{models_dir}/value_predictor.pkl", "wb") as f:
        pickle.dump(value_predictor, f)

    with open(f"{models_dir}/label_encoders.pkl", "wb") as f:
        pickle.dump(label_encoders, f)

    print(f"✅ Models saved to {models_dir}/")


def test_models(models_dir="models"):
    """Test loading and using the trained models."""
    print("🧪 Testing trained models...")

    try:
        # Load models
        with open(f"{models_dir}/pick_predictor.pkl", "rb") as f:
            pick_predictor = pickle.load(f)

        with open(f"{models_dir}/value_predictor.pkl", "rb") as f:
            value_predictor = pickle.load(f)

        with open(f"{models_dir}/label_encoders.pkl", "rb") as f:
            label_encoders = pickle.load(f)

        # Test prediction
        test_features = [
            3,  # round_num
            25,  # pick_overall
            15.0,  # adp
            200.0,  # projection
            12,  # league_size
            2,  # qb_taken_so_far
            8,  # rb_taken_so_far
            12,  # wr_taken_so_far
            1,  # te_taken_so_far
            0,  # team_qb_count
            1,  # team_rb_count
            1,  # team_wr_count
            0,  # team_te_count
            0,  # is_early_draft_position
            0,  # is_late_draft_position
            0,  # is_early_round
            1,  # is_middle_round
            0,  # is_late_round
            1,  # position_encoded (RB)
            0  # scoring_format_encoded (PPR)
        ]

        # Test position prediction
        position_probs = pick_predictor.predict_proba([test_features])[0]
        position_classes = pick_predictor.classes_

        print("  Position prediction probabilities:")
        for pos, prob in zip(position_classes, position_probs):
            print(f"    {pos}: {prob:.3f}")

        # Test value prediction
        predicted_value = value_predictor.predict([test_features])[0]
        print(f"  Predicted player value: {predicted_value:.1f}")

        print("✅ Model testing successful!")
        return True

    except Exception as e:
        print(f"❌ Model testing failed: {e}")
        return False


def main():
    """Main training pipeline."""
    print("🚀 Fantasy Draft ML Training Pipeline")
    print("=" * 50)

    # Step 1: Create training data
    df = create_training_data()
    print(f"✅ Generated {len(df)} training examples")

    # Show data summary
    print(f"\nTraining data summary:")
    print(f"  Drafts simulated: {df['draft_id'].nunique()}")
    print(f"  Position distribution:")
    for pos, count in df['position'].value_counts().head().items():
        print(f"    {pos}: {count}")

    # Step 2: Train models
    pick_predictor, value_predictor, label_encoders = train_models(df)

    # Step 3: Save models
    save_models(pick_predictor, value_predictor, label_encoders)

    # Step 4: Test models
    success = test_models()

    if success:
        print("\n🎉 Training pipeline completed successfully!")
        print("🎯 Your ML models are ready to use!")
        print("\nNext steps:")
        print("1. Add ML integration to your Flask app")
        print("2. Use the enhanced recommendations")
        print("3. Enjoy smarter draft picks! 🏈")
    else:
        print("\n❌ Training pipeline failed")

    return success


if __name__ == "__main__":
    main()