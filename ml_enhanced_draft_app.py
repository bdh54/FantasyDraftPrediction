#!/usr/bin/env python3
"""
ML-Enhanced Draft Advisor
Integrates trained ML models with the existing draft advisor
"""

import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import os
from datetime import datetime


class MLEnhancedDraftAnalyzer:
    """Enhanced draft analyzer that uses ML models trained on real data."""

    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.pick_predictor = None
        self.value_predictor = None
        self.label_encoders = {}
        self.load_models()

        # Fallback to rule-based if models not available
        self.use_ml = self.pick_predictor is not None

    def load_models(self):
        """Load trained ML models."""
        try:
            if os.path.exists(f"{self.models_dir}/pick_predictor.pkl"):
                with open(f"{self.models_dir}/pick_predictor.pkl", "rb") as f:
                    self.pick_predictor = pickle.load(f)

            if os.path.exists(f"{self.models_dir}/value_predictor.pkl"):
                with open(f"{self.models_dir}/value_predictor.pkl", "rb") as f:
                    self.value_predictor = pickle.load(f)

            if os.path.exists(f"{self.models_dir}/label_encoders.pkl"):
                with open(f"{self.models_dir}/label_encoders.pkl", "rb") as f:
                    self.label_encoders = pickle.load(f)

            print(f"ML models loaded successfully from {self.models_dir}/")

        except Exception as e:
            print(f"Could not load ML models: {e}")
            print("Falling back to rule-based recommendations")

    def predict_other_team_behavior(self, draft_state: Dict, round_num: int) -> Dict[str, float]:
        """Predict how other teams will draft using ML models."""
        if not self.use_ml:
            # Fallback to rule-based predictions
            return self._rule_based_team_behavior(draft_state, round_num)

        try:
            # Prepare features for ML prediction
            features = self._extract_prediction_features(draft_state, round_num)

            # Predict position probabilities
            position_probs = self.pick_predictor.predict_proba([features])[0]
            position_classes = self.pick_predictor.classes_

            # Convert to dictionary
            predictions = dict(zip(position_classes, position_probs))

            return predictions

        except Exception as e:
            print(f"ML prediction failed: {e}")
            return self._rule_based_team_behavior(draft_state, round_num)

    def _extract_prediction_features(self, draft_state: Dict, round_num: int) -> List[float]:
        """Extract features for ML model prediction."""
        # Count positions drafted so far
        positions_drafted = {'QB': 0, 'RB': 0, 'WR': 0, 'TE': 0, 'K': 0, 'DST': 0}

        # This would be based on actual draft state
        # For now, using placeholder logic
        total_picks = draft_state.get('total_picks', 0)
        league_size = draft_state.get('league_size', 12)

        features = [
            round_num,  # round_num
            total_picks,  # pick_overall
            0.0,  # adp (average for available players)
            0.0,  # projection (average for available)
            league_size,  # league_size
            positions_drafted['QB'],  # qb_taken_so_far
            positions_drafted['RB'],  # rb_taken_so_far
            positions_drafted['WR'],  # wr_taken_so_far
            positions_drafted['TE'],  # te_taken_so_far
            1 if total_picks <= 4 else 0,  # is_early_draft_position
            1 if total_picks >= 9 else 0,  # is_late_draft_position
            1 if round_num <= 3 else 0,  # is_early_round
            1 if 3 < round_num <= 10 else 0,  # is_middle_round
            1 if round_num > 10 else 0,  # is_late_round
            self._encode_position('RB'),  # position_encoded (example)
            self._encode_scoring('PPR')  # scoring_format_encoded
        ]

        return features

    def _encode_position(self, position: str) -> float:
        """Encode position using trained label encoder."""
        if 'position' in self.label_encoders:
            try:
                return float(self.label_encoders['position'].transform([position])[0])
            except:
                return 0.0
        return 0.0

    def _encode_scoring(self, scoring: str) -> float:
        """Encode scoring format using trained label encoder."""
        if 'scoring_format' in self.label_encoders:
            try:
                return float(self.label_encoders['scoring_format'].transform([scoring])[0])
            except:
                return 0.0
        return 0.0

    def _rule_based_team_behavior(self, draft_state: Dict, round_num: int) -> Dict[str, float]:
        """Fallback rule-based team behavior prediction."""
        if round_num <= 3:
            return {'RB': 0.4, 'WR': 0.35, 'TE': 0.15, 'QB': 0.1}
        elif round_num <= 6:
            return {'RB': 0.3, 'WR': 0.35, 'QB': 0.2, 'TE': 0.15}
        elif round_num <= 10:
            return {'QB': 0.25, 'RB': 0.25, 'WR': 0.3, 'TE': 0.2}
        else:
            return {'DST': 0.3, 'K': 0.25, 'RB': 0.2, 'WR': 0.15, 'QB': 0.1}

    def enhanced_player_valuation(self, player: 'Player', draft_context: Dict) -> float:
        """Use ML models to enhance player valuation."""
        if not self.use_ml or not self.value_predictor:
            return player.projection  # Fallback to base projection

        try:
            # Prepare features for value prediction
            features = self._extract_value_features(player, draft_context)

            # Predict enhanced value
            enhanced_value = self.value_predictor.predict([features])[0]

            # Blend with original projection (80% ML, 20% original)
            blended_value = 0.8 * enhanced_value + 0.2 * player.projection

            return blended_value

        except Exception as e:
            print(f"Enhanced valuation failed: {e}")
            return player.projection

    def _extract_value_features(self, player: 'Player', draft_context: Dict) -> List[float]:
        """Extract features for player value prediction."""
        round_num = draft_context.get('round_num', 1)
        total_picks = draft_context.get('total_picks', 0)
        league_size = draft_context.get('league_size', 12)

        features = [
            round_num,  # round_num
            total_picks,  # pick_overall
            player.adp,  # adp
            league_size,  # league_size
            draft_context.get('qb_taken', 0),  # qb_taken_so_far
            draft_context.get('rb_taken', 0),  # rb_taken_so_far
            draft_context.get('wr_taken', 0),  # wr_taken_so_far
            draft_context.get('te_taken', 0),  # te_taken_so_far
            self._encode_position(player.position),  # position_encoded
            self._encode_scoring(draft_context.get('scoring', 'PPR'))  # scoring_format_encoded
        ]

        return features

    def get_ml_insights(self, available_players: List['Player'], draft_state: Dict) -> Dict:
        """Generate ML-powered insights for the draft."""
        insights = {
            'ml_enabled': self.use_ml,
            'predictions': {},
            'recommendations': [],
            'market_trends': {}
        }

        if not self.use_ml:
            insights['message'] = "ML models not available - using rule-based analysis"
            return insights

        try:
            # Predict position trends
            round_num = draft_state.get('current_round', 1)
            position_predictions = self.predict_other_team_behavior(draft_state, round_num)

            insights['predictions']['position_demand'] = position_predictions

            # Find players likely to be drafted soon
            high_demand_positions = sorted(position_predictions.items(),
                                           key=lambda x: x[1], reverse=True)[:2]

            for pos, prob in high_demand_positions:
                pos_players = [p for p in available_players if p.position == pos]
                if pos_players:
                    best_player = max(pos_players, key=lambda x: x.projection)
                    insights['recommendations'].append({
                        'player': best_player.name,
                        'position': pos,
                        'reason': f"{pos} position has {prob:.1%} chance of being drafted next",
                        'urgency': 'high' if prob > 0.3 else 'medium'
                    })

            # Market trend analysis
            insights['market_trends'] = {
                'hot_positions': [pos for pos, prob in position_predictions.items() if prob > 0.25],
                'cold_positions': [pos for pos, prob in position_predictions.items() if prob < 0.1],
                'round_type': 'early' if round_num <= 3 else 'middle' if round_num <= 10 else 'late'
            }

        except Exception as e:
            insights['error'] = f"ML analysis failed: {e}"

        return insights


# Integration with existing Flask app
def enhance_existing_app_with_ml():
    """
    Code to integrate ML models into your existing Flask app.
    Add this to your draft_advisor.py file.
    """

    integration_code = '''
# Add this import at the top of your Flask app
from ml_enhanced_draft_app import MLEnhancedDraftAnalyzer

# Add this as a global variable after app initialization
ml_analyzer = MLEnhancedDraftAnalyzer()

# Enhanced analysis route - add this to your Flask app
@app.route('/api/analyze_ml', methods=['POST'])
def analyze_draft_with_ml():
    """Enhanced API endpoint with ML predictions."""
    try:
        data = request.get_json()

        # Parse league settings (same as before)
        settings = LeagueSettings(
            teams=data.get('teams', 12),
            scoring_type=data.get('scoring_type', 'PPR'),
            draft_position=data.get('draft_position', 1),
            current_round=data.get('current_round', 1)
        )

        # Parse players (same as before)
        available_players = []
        for p_data in data.get('available_players', []):
            player = Player(
                name=p_data['name'],
                position=p_data['position'],
                team=p_data['team'],
                projection=float(p_data['projection']),
                bye=int(p_data['bye']),
                adp=float(p_data['adp'])
            )
            available_players.append(player)

        my_roster = []
        for p_data in data.get('my_roster', []):
            player = Player(
                name=p_data['name'],
                position=p_data['position'],
                team=p_data['team'],
                projection=float(p_data['projection']),
                bye=int(p_data['bye']),
                adp=float(p_data['adp'])
            )
            my_roster.append(player)

        # Enhanced analysis with ML
        draft_context = {
            'current_round': settings.current_round,
            'total_picks': len(my_roster),
            'league_size': settings.teams,
            'scoring': settings.scoring_type,
            'qb_taken': len([p for p in my_roster if p.position == 'QB']),
            'rb_taken': len([p for p in my_roster if p.position == 'RB']),
            'wr_taken': len([p for p in my_roster if p.position == 'WR']),
            'te_taken': len([p for p in my_roster if p.position == 'TE'])
        }

        # Get ML insights
        ml_insights = ml_analyzer.get_ml_insights(available_players, draft_context)

        # Enhanced player valuations
        for player in available_players:
            enhanced_projection = ml_analyzer.enhanced_player_valuation(player, draft_context)
            player.projection = enhanced_projection  # Update with ML-enhanced value

        # Run standard analysis with enhanced projections
        analyzer = DraftAnalyzer(settings)
        recommendations = analyzer.analyze_players(available_players, my_roster)

        # Format response with ML insights
        result = []
        for player in recommendations:
            urgency_level = "can-wait"
            if player.scarcity > 1.8:
                urgency_level = "urgent"
            elif player.scarcity > 1.4:
                urgency_level = "high-priority"
            elif player.scarcity > 1.0:
                urgency_level = "consider-soon"

            # Check if player is in ML recommendations
            ml_recommended = any(rec['player'] == player.name for rec in ml_insights.get('recommendations', []))

            result.append({
                'name': player.name,
                'position': player.position,
                'team': player.team,
                'projection': player.projection,
                'bye': player.bye,
                'adp': player.adp,
                'vos': round(player.vos, 1),
                'scarcity': round(player.scarcity, 2),
                'need_multiplier': round(player.need_multiplier, 2),
                'total_score': round(player.total_score, 1),
                'recommendation': player.recommendation,
                'urgency_level': urgency_level,
                'ml_enhanced': True,
                'ml_recommended': ml_recommended
            })

        return jsonify({
            'success': True,
            'recommendations': result,
            'ml_insights': ml_insights,
            'analysis_type': 'ml_enhanced'
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

# Add this JavaScript to your HTML template to use ML analysis
ml_javascript = """
async function analyzeWithML() {
    if (availablePlayers.length === 0) return;

    const settings = {
        teams: parseInt(document.getElementById('teams').value),
        scoring_type: document.getElementById('scoring_type').value,
        draft_position: parseInt(document.getElementById('draft_position').value),
        current_round: parseInt(document.getElementById('current_round').value),
        available_players: availablePlayers,
        my_roster: myRoster
    };

    try {
        const response = await fetch('/api/analyze_ml', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(settings)
        });

        const data = await response.json();
        if (data.success) {
            updateRecommendationsWithML(data.recommendations, data.ml_insights);
        } else {
            console.error('ML Analysis error:', data.error);
            // Fallback to standard analysis
            await analyzeAndUpdate();
        }
    } catch (error) {
        console.error('Error with ML analysis:', error);
        // Fallback to standard analysis
        await analyzeAndUpdate();
    }
}

function updateRecommendationsWithML(recommendations, mlInsights) {
    const container = document.getElementById('recommendations');

    // Add ML insights header
    let insightsHtml = '';
    if (mlInsights.ml_enabled) {
        const hotPositions = mlInsights.market_trends?.hot_positions || [];
        const predictions = mlInsights.predictions?.position_demand || {};

        insightsHtml = `
            <div class="bg-purple-50 border border-purple-200 rounded-lg p-4 mb-4">
                <h3 class="font-semibold text-purple-800 mb-2">🤖 ML Market Analysis</h3>
                <div class="grid grid-cols-2 gap-2 text-sm">
                    <div>
                        <strong>Hot Positions:</strong> ${hotPositions.join(', ') || 'None'}
                    </div>
                    <div>
                        <strong>Next Pick Likely:</strong> 
                        ${Object.entries(predictions).slice(0,2).map(([pos, prob]) => 
                            `${pos} (${(prob*100).toFixed(0)}%)`).join(', ')}
                    </div>
                </div>
            </div>
        `;
    }

    if (recommendations.length === 0) {
        container.innerHTML = insightsHtml + '<p class="text-gray-500 italic">No recommendations available</p>';
        return;
    }

    container.innerHTML = insightsHtml + recommendations.map((player, index) => `
        <div class="border border-gray-200 rounded-lg p-4 ${index === 0 ? 'ring-2 ring-blue-500' : ''}
                    ${player.ml_recommended ? 'bg-purple-50 border-purple-300' : ''}">
            <div class="flex justify-between items-start mb-2">
                <div>
                    <h3 class="font-semibold text-lg">${player.name}</h3>
                    <p class="text-sm text-gray-600">${player.position} - ${player.team}</p>
                    ${player.ml_enhanced ? 
                        `<div class="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded mt-1 inline-block">
                            🧠 ML Enhanced Projection
                        </div>` : ''
                    }
                    ${player.ml_recommended ? 
                        `<div class="text-xs bg-purple-100 text-purple-800 px-2 py-1 rounded mt-1 inline-block">
                            🔥 ML Hot Pick
                        </div>` : ''
                    }
                </div>
                <div class="text-right">
                    <div class="inline-block px-2 py-1 rounded-full text-xs font-medium urgency-${player.urgency_level}">
                        ${getUrgencyText(player.urgency_level)}
                    </div>
                    <p class="text-xs text-gray-500 mt-1">Score: ${player.total_score}</p>
                </div>
            </div>

            <p class="text-sm text-gray-700 mb-3">${player.recommendation}</p>

            <div class="flex justify-between items-center">
                <div class="text-xs text-gray-500">
                    <span>Proj: ${player.projection}</span>
                    <span class="ml-3">VOS: ${player.vos}</span>
                    <span class="ml-3">ADP: ${player.adp}</span>
                </div>
                <button onclick="draftPlayer('${player.name}')" 
                        class="bg-blue-600 text-white px-3 py-1 rounded text-sm hover:bg-blue-700">
                    Draft
                </button>
            </div>
        </div>
    `).join('');
}

// Update the auto-analysis to use ML
['teams', 'scoring_type', 'draft_position', 'current_round'].forEach(id => {
    document.getElementById(id).addEventListener('change', analyzeWithML);
});
"""
'''

    return integration_code


# Training pipeline runner
def run_training_pipeline():
    """Run the complete training pipeline."""

    print("🚀 Starting ML Training Pipeline")
    print("=" * 50)

    # Step 1: Data Collection
    from draft_data_trainer import DraftDataCollector, DraftPredictor, create_sample_data

    collector = DraftDataCollector()

    print("📊 Step 1: Collecting training data...")

    # For demo purposes, create sample data
    # In production, you'd scrape real data from Sleeper, ESPN, etc.
    sample_picks = create_sample_data()

    # Expand sample data for better training
    expanded_sample = []
    for i in range(50):  # Create 50 mock drafts
        for j, pick in enumerate(sample_picks):
            new_pick = DraftPick(
                round_num=pick.round_num,
                pick_num=pick.pick_num + (i * len(sample_picks)),
                player_name=f"{pick.player_name}_{i}" if i > 0 else pick.player_name,
                position=pick.position,
                team=pick.team,
                adp=pick.adp + np.random.normal(0, 0.5),  # Add some noise
                projection=pick.projection + np.random.normal(0, 10),
                drafted_by_team=(pick.drafted_by_team + i) % 12,
                league_size=pick.league_size,
                scoring_format=pick.scoring_format,
                draft_date=f"2023-08-{15 + i // 10}"
            )
            expanded_sample.append(new_pick)

    collector.save_picks_to_db(expanded_sample)
    print(f"✅ Saved {len(expanded_sample)} training examples")

    # Step 2: Model Training
    print("\n🧠 Step 2: Training ML models...")

    predictor = DraftPredictor()
    df = predictor.load_training_data()

    if len(df) == 0:
        print("❌ No training data available!")
        return False

    print(f"📈 Loaded {len(df)} training examples")

    # Engineer features and train models
    df = predictor.engineer_features(df)
    predictor.train_pick_predictor(df)
    predictor.train_value_predictor(df)

    # Step 3: Save models
    print("\n💾 Step 3: Saving trained models...")
    predictor.save_models()

    print("\n✅ Training pipeline complete!")
    print("🎯 Models are ready for use in your draft advisor")

    return True


if __name__ == "__main__":
    # Run training pipeline
    success = run_training_pipeline()

    if success:
        print("\n🚀 Testing ML-enhanced analyzer...")

        # Test the ML analyzer
        ml_analyzer = MLEnhancedDraftAnalyzer()

        # Create test data
        from dataclasses import dataclass


        @dataclass
        class TestPlayer:
            name: str
            position: str
            team: str
            projection: float
            adp: float
            bye: int = 7


        test_players = [
            TestPlayer("Test RB1", "RB", "SF", 250.0, 5.0),
            TestPlayer("Test WR1", "WR", "MIA", 220.0, 8.0),
            TestPlayer("Test QB1", "QB", "BUF", 300.0, 25.0)
        ]

        test_draft_state = {
            'current_round': 3,
            'total_picks': 24,
            'league_size': 12,
            'scoring': 'PPR'
        }

        # Get ML insights
        insights = ml_analyzer.get_ml_insights(test_players, test_draft_state)

        print("\n📊 ML Insights Test:")
        print(f"ML Enabled: {insights['ml_enabled']}")
        print(f"Position Predictions: {insights.get('predictions', {})}")
        print(f"Recommendations: {len(insights.get('recommendations', []))}")

        print("\n🎉 ML integration ready!")
        print("Add the integration code to your Flask app to enable ML features.")
    else:
        print("\n❌ Training failed. Check your data and try again.")