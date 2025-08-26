#!/usr/bin/env python3
"""
Fantasy Football Draft Advisor
A Flask web application that analyzes available players and provides draft recommendations
based on positional scarcity, roster needs, and value over stream calculations.
"""

from flask import Flask, render_template, request, jsonify, session
import pandas as pd
import numpy as np
import io
import csv
import random
import time
import pickle
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'


@dataclass
class Player:
    name: str
    position: str
    team: str
    projection: float
    bye: int
    adp: float
    vos: float = 0.0
    scarcity: float = 0.0
    need_multiplier: float = 1.0
    total_score: float = 0.0
    recommendation: str = ""
    tier: int = 0
    ceiling: float = 0.0
    floor: float = 0.0

    def __post_init__(self):
        if self.ceiling == 0.0:
            self.ceiling = self.projection * 1.25
        if self.floor == 0.0:
            self.floor = self.projection * 0.75


@dataclass
class LeagueSettings:
    teams: int = 12
    scoring_type: str = 'PPR'
    draft_position: int = 1
    current_round: int = 1
    roster_spots: Dict[str, int] = None

    def __post_init__(self):
        if self.roster_spots is None:
            self.roster_spots = {
                'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1,
                'FLEX': 1, 'K': 1, 'DST': 1, 'BENCH': 6
            }


def get_ml_insights(available_players, draft_context):
    """Get ML insights if models are available."""
    try:
        # Try to load ML models
        with open("models/pick_predictor.pkl", "rb") as f:
            pick_predictor = pickle.load(f)

        # Create simple features for prediction
        features = [
            draft_context.get('round_num', 1),  # round number
            draft_context.get('total_picks', 0),  # picks so far
            15.0, 200.0, 12,  # defaults
            draft_context.get('qb_taken', 0),  # positions taken
            draft_context.get('rb_taken', 0),
            draft_context.get('wr_taken', 0),
            draft_context.get('te_taken', 0),
            0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0  # other features
        ]

        # Get position predictions
        probs = pick_predictor.predict_proba([features])[0]
        classes = pick_predictor.classes_
        position_demand = dict(zip(classes, probs))

        # Find hot positions
        hot_positions = [pos for pos, prob in position_demand.items() if prob > 0.25]

        return {
            'ml_enabled': True,
            'position_demand': position_demand,
            'hot_positions': hot_positions
        }

    except Exception as e:
        # If ML models don't exist, return empty insights
        return {'ml_enabled': False, 'hot_positions': []}


class DraftAnalyzer:
    """Core logic for analyzing draft situations and generating recommendations."""

    POSITION_SCARCITY = {
        'QB': 0.3,
        'RB': 1.2,
        'WR': 0.8,
        'TE': 1.0,
        'K': 0.1,
        'DST': 0.1
    }

    TYPICAL_NEEDS = {
        'QB': 1.5,
        'RB': 3.5,
        'WR': 4.0,
        'TE': 1.5,
        'K': 1.0,
        'DST': 1.0
    }

    def __init__(self, settings: LeagueSettings):
        self.settings = settings

    def calculate_tier_cliffs(self, players: List[Player], position: str) -> List[int]:
        """Find where significant value drops occur for a position."""
        pos_players = [p for p in players if p.position == position]
        pos_players.sort(key=lambda x: x.projection, reverse=True)

        if len(pos_players) < 3:
            return []

        gaps = []
        for i in range(len(pos_players) - 1):
            gap = pos_players[i].projection - pos_players[i + 1].projection
            gaps.append((i, gap))

        if not gaps:
            return []

        avg_gap = sum(g[1] for g in gaps) / len(gaps)
        cliff_threshold = avg_gap * 1.5

        return [i for i, gap in gaps if gap > cliff_threshold]

    def calculate_vos(self, player: Player, all_players: List[Player]) -> float:
        """Calculate Value Over Stream - how much better than replacement level."""
        pos_players = [p for p in all_players if p.position == player.position]
        pos_players.sort(key=lambda x: x.projection, reverse=True)

        # Stream level = worst starter + bench depth
        stream_index = min(self.settings.teams * 2, len(pos_players) - 1)
        stream_level = pos_players[stream_index].projection if pos_players else 0

        return player.projection - stream_level

    def calculate_scarcity_urgency(self, player: Player, all_players: List[Player]) -> float:
        """Calculate how urgent it is to draft this position now."""
        position = player.position
        pos_players = [p for p in all_players if p.position == position]
        cliffs = self.calculate_tier_cliffs(all_players, position)

        # Base urgency from position scarcity
        urgency = self.POSITION_SCARCITY.get(position, 0.5)

        # Find where this player ranks in their position
        pos_players.sort(key=lambda x: x.projection, reverse=True)
        try:
            player_index = next(i for i, p in enumerate(pos_players) if p.name == player.name)
        except StopIteration:
            player_index = len(pos_players)

        # Boost if approaching a cliff
        for cliff in cliffs[:2]:  # Check next 2 cliffs
            if abs(cliff - player_index) <= 2:
                urgency *= 1.8
                break

        # Raw scarcity adjustment
        typical_need = self.TYPICAL_NEEDS.get(position, 1.0)
        remaining_teams = self.settings.teams
        if len(pos_players) < typical_need * remaining_teams * 1.5:
            urgency *= 1.3

        # Draft stage adjustment
        if self.settings.current_round <= 5:
            urgency *= 0.8  # Early: value over scarcity
        elif self.settings.current_round >= 11:
            urgency *= 1.3  # Late: scarcity matters more

        return min(urgency, 2.5)  # Cap the urgency

    def get_roster_needs(self, my_roster: List[Player]) -> Dict[str, int]:
        """Calculate what positions are most needed."""
        position_counts = {}
        for player in my_roster:
            position_counts[player.position] = position_counts.get(player.position, 0) + 1

        needs = {}
        for pos, required in self.settings.roster_spots.items():
            if pos in ['BENCH', 'FLEX']:  # Skip these for need calculation
                continue
            current = position_counts.get(pos, 0)
            needs[pos] = max(0, required - current)

        return needs

    def apply_scoring_bonus(self, player: Player) -> float:
        """Apply PPR/Half-PPR bonuses."""
        if player.position not in ['WR', 'RB', 'TE']:
            return 0

        if self.settings.scoring_type == 'PPR':
            bonuses = {'WR': 15, 'RB': 8, 'TE': 12}
        elif self.settings.scoring_type == 'Half PPR':
            bonuses = {'WR': 8, 'RB': 4, 'TE': 6}
        else:
            bonuses = {'WR': 0, 'RB': 0, 'TE': 0}

        return bonuses.get(player.position, 0)

    def generate_recommendation(self, player: Player, scarcity: float,
                                need_multiplier: float, roster_needs: Dict[str, int]) -> str:
        """Generate human-readable recommendation text."""
        position_need = roster_needs.get(player.position, 0)

        if position_need > 1:
            reason = f"Need {player.position} - fills starter spot"
        elif position_need == 1:
            reason = f"Last {player.position} needed for starting lineup"
        elif scarcity > 1.5:
            reason = f"Position tier about to drop - grab now"
        elif need_multiplier < 1:
            reason = f"Best available value despite position depth"
        else:
            reason = f"Strong value with decent positional need"

        # Add urgency context
        if scarcity > 1.8:
            reason += " - URGENT"
        elif scarcity > 1.4:
            reason += " - high priority"

        return reason

    def analyze_players(self, available_players: List[Player],
                        my_roster: List[Player]) -> List[Player]:
        """Main analysis function that scores and ranks all available players."""
        if not available_players:
            return []

        roster_needs = self.get_roster_needs(my_roster)
        analyzed_players = []

        for player in available_players:
            # Calculate core metrics
            player.vos = self.calculate_vos(player, available_players)
            player.scarcity = self.calculate_scarcity_urgency(player, available_players)

            # Position need multiplier
            position_need = roster_needs.get(player.position, 0)
            if position_need > 0:
                player.need_multiplier = 1 + (position_need * 0.3)
            elif position_need == 0:
                player.need_multiplier = 0.7  # Already filled starter spots
            else:
                player.need_multiplier = 1.0

            # Scoring system bonus
            scoring_bonus = self.apply_scoring_bonus(player)

            # Calculate total score
            player.total_score = (player.vos + scoring_bonus) * player.need_multiplier * player.scarcity

            # Generate recommendation
            player.recommendation = self.generate_recommendation(
                player, player.scarcity, player.need_multiplier, roster_needs
            )

            analyzed_players.append(player)

        # Sort by total score
        analyzed_players.sort(key=lambda x: x.total_score, reverse=True)
        return analyzed_players[:10]  # Return top 10


def parse_csv_data(csv_text: str) -> List[Player]:
    """Parse CSV text into Player objects."""
    try:
        df = pd.read_csv(io.StringIO(csv_text))

        # Clean column names
        df.columns = df.columns.str.strip()

        players = []
        for _, row in df.iterrows():
            try:
                player = Player(
                    name=str(row.get('Name', '')).strip(),
                    position=str(row.get('Position', '')).strip().upper(),
                    team=str(row.get('Team', '')).strip().upper(),
                    projection=float(row.get('Projection', 0)),
                    bye=int(row.get('Bye', 0)),
                    adp=float(row.get('ADP', 999))
                )
                if player.name and player.position:
                    players.append(player)
            except (ValueError, TypeError):
                continue  # Skip malformed rows

        return players
    except Exception as e:
        print(f"Error parsing CSV: {e}")
        return []


def get_sample_data() -> str:
    """Return sample CSV data for demo purposes."""
    return """Name,Position,Team,Projection,Bye,ADP
Christian McCaffrey,RB,SF,285.5,9,1.2
Tyreek Hill,WR,MIA,245.8,11,2.1
Austin Ekeler,RB,LAC,268.3,5,3.4
Cooper Kupp,WR,LAR,235.6,7,4.2
Derrick Henry,RB,TEN,255.1,7,5.8
Davante Adams,WR,LV,228.4,13,6.1
Nick Chubb,RB,CLE,248.7,7,7.3
Stefon Diggs,WR,BUF,222.9,7,8.2
Travis Kelce,TE,KC,198.5,10,9.1
Tony Pollard,RB,DAL,195.4,7,10.5
Ja'Marr Chase,WR,CIN,215.7,7,11.2
Saquon Barkley,RB,NYG,210.3,11,12.8
CeeDee Lamb,WR,DAL,208.9,7,13.5
Josh Jacobs,RB,LV,205.1,13,14.2
A.J. Brown,WR,PHI,201.8,10,15.3
Joe Mixon,RB,CIN,198.2,7,16.4
Amon-Ra St. Brown,WR,DET,195.6,9,17.1
Kenneth Walker III,RB,SEA,190.8,5,18.7
Mike Evans,WR,TB,188.3,11,19.2
Jaylen Waddle,WR,MIA,185.9,11,20.1
Josh Allen,QB,BUF,310.5,7,25.3
Patrick Mahomes,QB,KC,305.2,10,28.1
Lamar Jackson,QB,BAL,298.7,8,32.4
Jalen Hurts,QB,PHI,290.3,10,35.6
Joe Burrow,QB,CIN,275.8,7,45.2
Mark Andrews,TE,BAL,165.4,8,55.3
George Kittle,TE,SF,158.7,9,62.1
T.J. Hockenson,TE,MIN,145.2,13,78.4"""


# Flask Routes
@app.route('/')
def index():
    """Main page with the draft advisor interface."""
    return render_template('index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze_draft():
    """API endpoint to analyze current draft situation."""
    try:
        data = request.get_json()

        # Parse league settings
        settings = LeagueSettings(
            teams=data.get('teams', 12),
            scoring_type=data.get('scoring_type', 'PPR'),
            draft_position=data.get('draft_position', 1),
            current_round=data.get('current_round', 1)
        )

        # Parse players
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

        # Add ML insights
        draft_context = {
            'round_num': settings.current_round,
            'total_picks': len(my_roster),
            'qb_taken': len([p for p in my_roster if p.position == 'QB']),
            'rb_taken': len([p for p in my_roster if p.position == 'RB']),
            'wr_taken': len([p for p in my_roster if p.position == 'WR']),
            'te_taken': len([p for p in my_roster if p.position == 'TE'])
        }
        ml_insights = get_ml_insights(available_players, draft_context)

        # Analyze
        analyzer = DraftAnalyzer(settings)
        recommendations = analyzer.analyze_players(available_players, my_roster)

        # Format response
        result = []
        for player in recommendations:
            urgency_level = "can-wait"
            if player.scarcity > 1.8:
                urgency_level = "urgent"
            elif player.scarcity > 1.4:
                urgency_level = "high-priority"
            elif player.scarcity > 1.0:
                urgency_level = "consider-soon"

            # Check if this position is in high demand (ML)
            is_hot = player.position in ml_insights.get('hot_positions', [])
            recommendation = player.recommendation
            if is_hot:
                recommendation += " - 🔥 HIGH DEMAND"

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
                'recommendation': recommendation,
                'urgency_level': urgency_level,
                'ml_hot': is_hot
            })

        return jsonify({
            'success': True,
            'recommendations': result,
            'ml_insights': ml_insights
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/api/parse_csv', methods=['POST'])
def parse_csv():
    """Parse CSV data and return structured player data."""
    try:
        data = request.get_json()
        csv_text = data.get('csv_text', '')

        if not csv_text.strip():
            return jsonify({
                'success': False,
                'error': 'No CSV data provided'
            }), 400

        players = parse_csv_data(csv_text)

        result = []
        for player in players:
            result.append({
                'name': player.name,
                'position': player.position,
                'team': player.team,
                'projection': player.projection,
                'bye': player.bye,
                'adp': player.adp
            })

        return jsonify({
            'success': True,
            'players': result,
            'count': len(result)
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/api/sample_data')
def sample_data():
    """Return sample CSV data."""
    return jsonify({
        'success': True,
        'csv_data': get_sample_data()
    })


# HTML Template (save as templates/index.html)
HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fantasy Football Draft Advisor</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .urgency-urgent { @apply text-red-600 bg-red-100; }
        .urgency-high-priority { @apply text-orange-600 bg-orange-100; }
        .urgency-consider-soon { @apply text-yellow-600 bg-yellow-100; }
        .urgency-can-wait { @apply text-green-600 bg-green-100; }
    </style>
</head>
<body class="bg-gray-50 min-h-screen">
    <div class="max-w-7xl mx-auto p-6" id="app">
        <!-- Header -->
        <div class="bg-white rounded-lg shadow-sm border p-6 mb-6">
            <h1 class="text-3xl font-bold text-gray-900 mb-2">🎯 Fantasy Draft Advisor</h1>
            <p class="text-gray-600">Upload your draft board and get smart recommendations with ML insights</p>
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-4 gap-6">
            <!-- Settings Panel -->
            <div class="bg-white rounded-lg shadow-sm border p-6">
                <h2 class="text-xl font-semibold mb-4">⚙️ League Settings</h2>

                <div class="space-y-4">
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-1">Teams</label>
                        <input type="number" id="teams" value="12" min="8" max="16" 
                               class="w-full border border-gray-300 rounded-md px-3 py-2">
                    </div>

                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-1">Scoring</label>
                        <select id="scoring_type" class="w-full border border-gray-300 rounded-md px-3 py-2">
                            <option value="Standard">Standard</option>
                            <option value="Half PPR">Half PPR</option>
                            <option value="PPR" selected>PPR</option>
                        </select>
                    </div>

                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-1">Draft Position</label>
                        <input type="number" id="draft_position" value="6" min="1" max="16"
                               class="w-full border border-gray-300 rounded-md px-3 py-2">
                    </div>

                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-1">Current Round</label>
                        <input type="number" id="current_round" value="1" min="1" max="20"
                               class="w-full border border-gray-300 rounded-md px-3 py-2">
                    </div>
                </div>

                <!-- CSV Input -->
                <div class="mt-6">
                    <h3 class="text-lg font-semibold mb-2">📁 Player Data</h3>
                    <button onclick="loadSampleData()" 
                            class="mb-3 bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 text-sm w-full">
                        Load Sample Data
                    </button>
                    <textarea id="csv_input" placeholder="Paste CSV: Name,Position,Team,Projection,Bye,ADP"
                              class="w-full border border-gray-300 rounded-md px-3 py-2 h-32 text-sm font-mono"></textarea>
                    <button onclick="processCsv()" 
                            class="mt-2 bg-green-600 text-white px-4 py-2 rounded-md hover:bg-green-700 w-full">
                        Process Players
                    </button>
                    <div id="player_count" class="mt-2 text-sm text-gray-600"></div>
                </div>
            </div>

            <!-- My Roster -->
            <div class="bg-white rounded-lg shadow-sm border p-6">
                <h2 class="text-xl font-semibold mb-4">👥 My Roster (<span id="roster_count">0</span>)</h2>
                <div id="my_roster" class="space-y-2">
                    <p class="text-gray-500 italic">No players drafted yet</p>
                </div>
            </div>

            <!-- Recommendations -->
            <div class="lg:col-span-2 bg-white rounded-lg shadow-sm border p-6">
                <h2 class="text-xl font-semibold mb-4">📈 Recommendations</h2>
                <div id="recommendations" class="space-y-3">
                    <p class="text-gray-500 italic">Load player data to see recommendations</p>
                </div>
            </div>
        </div>
    </div>

    <script>
        let availablePlayers = [];
        let myRoster = [];

        async function loadSampleData() {
            try {
                const response = await fetch('/api/sample_data');
                const data = await response.json();
                if (data.success) {
                    document.getElementById('csv_input').value = data.csv_data;
                    await processCsv();
                }
            } catch (error) {
                console.error('Error loading sample data:', error);
            }
        }

        async function processCsv() {
            const csvText = document.getElementById('csv_input').value;
            if (!csvText.trim()) return;

            try {
                const response = await fetch('/api/parse_csv', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ csv_text: csvText })
                });

                const data = await response.json();
                if (data.success) {
                    availablePlayers = data.players;
                    document.getElementById('player_count').textContent = 
                        `Loaded ${data.count} players`;
                    await analyzeAndUpdate();
                } else {
                    alert('Error parsing CSV: ' + data.error);
                }
            } catch (error) {
                console.error('Error processing CSV:', error);
                alert('Error processing CSV data');
            }
        }

        async function analyzeAndUpdate() {
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
                const response = await fetch('/api/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(settings)
                });

                const data = await response.json();
                if (data.success) {
                    updateRecommendations(data.recommendations, data.ml_insights);
                } else {
                    console.error('Analysis error:', data.error);
                }
            } catch (error) {
                console.error('Error analyzing draft:', error);
            }
        }

        function updateRecommendations(recommendations, mlInsights) {
            mlInsights = mlInsights || {};
            const hotPositions = mlInsights.hot_positions || [];

            const container = document.getElementById('recommendations');
            if (recommendations.length === 0) {
                container.innerHTML = '<p class="text-gray-500 italic">No recommendations available</p>';
                return;
            }

            // Add ML insights header if available
            let insightsHtml = '';
            if (mlInsights.ml_enabled && hotPositions.length > 0) {
                insightsHtml = `
                    <div class="bg-purple-50 border border-purple-200 rounded-lg p-4 mb-4">
                        <h3 class="font-semibold text-purple-800 mb-2">🤖 ML Market Insights</h3>
                        <p class="text-sm text-purple-700">
                            High Demand Positions: <strong>${hotPositions.join(', ')}</strong>
                        </p>
                    </div>
                `;
            }

            container.innerHTML = insightsHtml + recommendations.map((player, index) => `
                <div class="border border-gray-200 rounded-lg p-4 ${index === 0 ? 'ring-2 ring-blue-500' : ''}
                            ${player.ml_hot ? 'bg-red-50 border-red-300' : ''}">
                    <div class="flex justify-between items-start mb-2">
                        <div>
                            <h3 class="font-semibold text-lg">${player.name}
                                ${player.ml_hot ? ' 🔥' : ''}
                            </h3>
                            <p class="text-sm text-gray-600">${player.position} - ${player.team}</p>
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

        function getUrgencyText(urgencyLevel) {
            const texts = {
                'urgent': 'URGENT',
                'high-priority': 'High Priority',
                'consider-soon': 'Consider Soon',
                'can-wait': 'Can Wait'
            };
            return texts[urgencyLevel] || 'Unknown';
        }

        function draftPlayer(playerName) {
            const player = availablePlayers.find(p => p.name === playerName);
            if (player) {
                myRoster.push(player);
                availablePlayers = availablePlayers.filter(p => p.name !== playerName);
                updateRosterDisplay();
                analyzeAndUpdate();
            }
        }

        function undraftPlayer(playerName) {
            const player = myRoster.find(p => p.name === playerName);
            if (player) {
                availablePlayers.push(player);
                availablePlayers.sort((a, b) => a.adp - b.adp);
                myRoster = myRoster.filter(p => p.name !== playerName);
                updateRosterDisplay();
                analyzeAndUpdate();
            }
        }

        function updateRosterDisplay() {
            const container = document.getElementById('my_roster');
            const countEl = document.getElementById('roster_count');

            countEl.textContent = myRoster.length;

            if (myRoster.length === 0) {
                container.innerHTML = '<p class="text-gray-500 italic">No players drafted yet</p>';
                return;
            }

            container.innerHTML = myRoster.map(player => `
                <div class="flex justify-between items-center p-2 bg-gray-50 rounded">
                    <div>
                        <span class="font-medium">${player.name}</span>
                        <span class="text-sm text-gray-500 ml-2">${player.position} - ${player.team}</span>
                    </div>
                    <button onclick="undraftPlayer('${player.name}')" 
                            class="text-red-500 hover:text-red-700 text-sm">
                        Remove
                    </button>
                </div>
            `).join('');
        }

        // Auto-update when settings change
        ['teams', 'scoring_type', 'draft_position', 'current_round'].forEach(id => {
            document.getElementById(id).addEventListener('change', analyzeAndUpdate);
        });
    </script>
</body>
</html>'''

# Create templates directory and file
import os

if not os.path.exists('templates'):
    os.makedirs('templates')

with open('templates/index.html', 'w') as f:
    f.write(HTML_TEMPLATE)

if __name__ == '__main__':
    print("Fantasy Football Draft Advisor starting...")
    print("Visit http://localhost:8080 to use the app")
    print("\nTo run this app:")
    print("1. pip install flask pandas numpy scikit-learn")
    print("2. python simple_ml_trainer.py  # Train ML models first")
    print("3. python draft_advisor.py      # Run the app")
    print("4. Open http://localhost:8080 in your browser")

    app.run(debug=True, host='0.0.0.0', port=8080)