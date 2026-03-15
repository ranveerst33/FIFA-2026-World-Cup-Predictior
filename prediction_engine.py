"""
FIFA 2026 World Cup Prediction - Prediction Engine
====================================================
High-level prediction logic: group stage simulation, player stats lookup,
team predictions, custom Playing XI analysis, and injury impact assessment.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")


class PredictionEngine:
    """Prediction engine for FIFA 2026 World Cup."""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_cols = None
        self.data_processor = None
        self._loaded = False

    def load(self):
        """Load trained model, scaler, and data processor."""
        if self._loaded:
            return self

        from data_processor import DataProcessor

        print("Loading prediction engine...")

        # Load the ensemble model (best model)
        model_path = os.path.join(MODELS_DIR, "voting_ensemble.pkl")
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            print("  [OK] Loaded Voting Ensemble model")
        else:
            # Fallback to random forest
            model_path = os.path.join(MODELS_DIR, "random_forest.pkl")
            self.model = joblib.load(model_path)
            print("  [OK] Loaded Random Forest model (fallback)")

        self.scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))

        with open(os.path.join(MODELS_DIR, "feature_cols.json"), "r") as f:
            self.feature_cols = json.load(f)

        # Load data processor for features
        print("  Loading data for feature computation...")
        self.dp = DataProcessor()
        self.dp.load_all_data()
        self.dp.compute_elo_ratings()
        self.dp.compute_team_stats()
        self.dp.compute_squad_features()

        self._loaded = True
        print("  [OK] Prediction engine ready!")
        return self

    # ------------------------------------------------------------------
    # TEAM NAME MAPPING (FIFA 2026 names ↔ historical names)
    # ------------------------------------------------------------------
    _NAME_MAP = {
        "Korea Republic": "South Korea",
        "IR Iran": "Iran",
        "Côte d'Ivoire": "Ivory Coast",
        "Curaçao": "Curacao",
        "United States": "USA",
        "Cabo Verde": "Cape Verde",
    }

    def _resolve_team(self, name):
        """Resolve a team name to the historical name used in ELO/stats."""
        if name in self.dp.elo_ratings:
            return name
        if name in self._NAME_MAP and self._NAME_MAP[name] in self.dp.elo_ratings:
            return self._NAME_MAP[name]
        # Reverse map
        for k, v in self._NAME_MAP.items():
            if v == name and k in self.dp.elo_ratings:
                return k
        # Fuzzy
        for key in self.dp.elo_ratings:
            if name.lower() in key.lower() or key.lower() in name.lower():
                return key
        return name

    # ------------------------------------------------------------------
    # MATCH PREDICTION
    # ------------------------------------------------------------------
    def predict_match(self, home_team, away_team, stage_importance=1, home_injuries=None, away_injuries=None):
        """Predict a single match outcome with probabilities, optionally accounting for injuries."""
        home_r = self._resolve_team(home_team)
        away_r = self._resolve_team(away_team)

        features = {}

        features["elo_home"] = self.dp._get_elo(home_r)
        features["elo_away"] = self.dp._get_elo(away_r)
        features["elo_diff"] = features["elo_home"] - features["elo_away"]

        for prefix, team in [("home", home_r), ("away", away_r)]:
            features[f"{prefix}_win_rate"] = self.dp._get_team_feature(team, "win_rate")
            features[f"{prefix}_goals_per_match"] = self.dp._get_team_feature(team, "goals_per_match")
            features[f"{prefix}_goals_conceded_pm"] = self.dp._get_team_feature(team, "goals_conceded_per_match")
            features[f"{prefix}_goal_diff_pm"] = self.dp._get_team_feature(team, "goal_diff_per_match")
            features[f"{prefix}_wc_appearances"] = self.dp._get_team_feature(team, "wc_appearances")
            features[f"{prefix}_knockout_ratio"] = self.dp._get_team_feature(team, "knockout_ratio")
            features[f"{prefix}_total_matches"] = self.dp._get_team_feature(team, "total_matches")

        features["home_attack_strength"] = features["home_goals_per_match"] - features["away_goals_conceded_pm"]
        features["away_attack_strength"] = features["away_goals_per_match"] - features["home_goals_conceded_pm"]
        features["home_defense_strength"] = features["away_goals_per_match"] - features["home_goals_conceded_pm"]
        features["away_defense_strength"] = features["home_goals_per_match"] - features["away_goals_conceded_pm"]
        features["wc_experience_diff"] = features["home_wc_appearances"] - features["away_wc_appearances"]
        features["win_rate_diff"] = features["home_win_rate"] - features["away_win_rate"]
        features["tournament_importance"] = stage_importance

        X = pd.DataFrame([features])[self.feature_cols]
        X = X.fillna(0)

        proba = self.model.predict_proba(X)[0]
        
        # --- INJURY ADJUSTMENT ---
        # If injuries are provided, we penalize the affected team's win probability
        # based on the relative impact (market value + caps lost)
        home_penalty = 0
        away_penalty = 0

        if home_injuries:
            impact = self._calculate_injury_penalty(home_team, home_injuries)
            home_penalty = impact
        
        if away_injuries:
            impact = self._calculate_injury_penalty(away_team, away_injuries)
            away_penalty = impact

        # Apply net adjustment (max shift of +/- 0.15)
        net_adj = np.clip(away_penalty - home_penalty, -0.15, 0.15)
        
        # Result labels: 0=Home Win, 1=Draw, 2=Away Win
        adj_proba = proba.copy()
        adj_proba[0] = np.clip(adj_proba[0] + net_adj, 0.02, 0.95)
        adj_proba[2] = np.clip(adj_proba[2] - net_adj, 0.02, 0.95)
        
        # Normalize
        adj_proba = adj_proba / adj_proba.sum()
        
        # Determine prediction from adjusted proba
        pred = np.argmax(adj_proba)

        result_labels = {0: "Home Win", 1: "Draw", 2: "Away Win"}

        return {
            "home_team": home_team,
            "away_team": away_team,
            "prediction": result_labels[pred],
            "probabilities": {
                f"{home_team} Win": float(adj_proba[0]),
                "Draw": float(adj_proba[1]),
                f"{away_team} Win": float(adj_proba[2]),
            },
            "elo_home": features["elo_home"],
            "elo_away": features["elo_away"],
            "confidence": float(max(adj_proba)),
            "injury_impact": net_adj,
        }

    def _calculate_injury_penalty(self, team_name, injured_player_ids):
        """Calculate a penalty score (0.0 to 0.1) based on importance of injured players."""
        players = self.get_team_players(team_name)
        if not players: return 0
        
        total_val = sum(p['market_value'] for p in players[:23])
        total_caps = sum(p['caps'] for p in players[:23])
        
        p_val = 0
        p_caps = 0
        for pid in injured_player_ids:
            p = next((x for x in players if x['player_id'] == pid), None)
            if p:
                p_val += p['market_value']
                p_caps += p['caps']
        
        # Impact factors
        val_impact = (p_val / total_val) if total_val > 0 else 0
        exp_impact = (p_caps / total_caps) if total_caps > 0 else 0
        
        return (val_impact * 0.5 + exp_impact * 0.5) * 0.15 # Max adjustment factor

    # ------------------------------------------------------------------
    # GROUP STAGE SIMULATION
    # ------------------------------------------------------------------
    def predict_group(self, group_name, injuries_dict=None):
        """
        Simulate all matches in a group and return standings.
        injuries_dict: mapping of team_name -> list of player_ids
        """
        groups = self.dp.get_2026_groups()

        if group_name not in groups:
            return {"error": f"Group {group_name} not found. Available: {list(groups.keys())}"}

        teams = groups[group_name]
        if len(teams) < 2:
            return {"error": f"Not enough confirmed teams in {group_name}"}

        # Get group matches from fixture list
        group_fixtures = self.dp.fifa2026[self.dp.fifa2026["Group"] == group_name].copy()

        standings = {t: {"points": 0, "gf": 0, "ga": 0, "gd": 0, "w": 0, "d": 0, "l": 0} for t in teams}
        match_results = []

        # Predict each match
        for _, row in group_fixtures.iterrows():
            home = row["Home Team"]
            away = row["Away Team"]

            # Skip placeholder teams
            if "/" in str(home) or "/" in str(away):
                continue
            if home not in teams or away not in teams:
                continue

            # Pass injuries if relevant
            home_inj = injuries_dict.get(home) if injuries_dict else None
            away_inj = injuries_dict.get(away) if injuries_dict else None

            result = self.predict_match(home, away, stage_importance=1, home_injuries=home_inj, away_injuries=away_inj)
            match_results.append(result)

            # Determine winner and update standings
            proba = result["probabilities"]
            home_win_p = proba[f"{home} Win"]
            draw_p = proba["Draw"]
            away_win_p = proba[f"{away} Win"]

            # Use expected points based on probabilities for more nuanced standings
            standings[home]["points"] += home_win_p * 3 + draw_p * 1
            standings[away]["points"] += away_win_p * 3 + draw_p * 1

            # Estimate goals based on team stats
            home_r = self._resolve_team(home)
            away_r = self._resolve_team(away)
            home_gpm = self.dp._get_team_feature(home_r, "goals_per_match", 1.0)
            away_gpm = self.dp._get_team_feature(away_r, "goals_per_match", 1.0)

            # Apply subtle goal adjustment based on win probability shift
            home_goal_adj = 1.0 + (home_win_p - 0.4) * 0.5
            away_goal_adj = 1.0 + (away_win_p - 0.4) * 0.5

            standings[home]["gf"] += home_gpm * home_goal_adj
            standings[home]["ga"] += away_gpm * away_goal_adj
            standings[away]["gf"] += away_gpm * away_goal_adj
            standings[away]["ga"] += home_gpm * home_goal_adj

            if result["prediction"] == "Home Win":
                standings[home]["w"] += 1
                standings[away]["l"] += 1
            elif result["prediction"] == "Draw":
                standings[home]["d"] += 1
                standings[away]["d"] += 1
            else:
                standings[away]["w"] += 1
                standings[home]["l"] += 1

        # Calculate GD
        for team in standings:
            standings[team]["gd"] = standings[team]["gf"] - standings[team]["ga"]

        # Sort by points, then GD, then GF
        sorted_teams = sorted(
            standings.items(),
            key=lambda x: (x[1]["points"], x[1]["gd"], x[1]["gf"]),
            reverse=True,
        )

        return {
            "group": group_name,
            "standings": [{
                "position": i + 1,
                "team": team,
                "points": round(standings[team]["points"], 2),
                "w": standings[team]["w"],
                "d": standings[team]["d"],
                "l": standings[team]["l"],
                "gf": round(standings[team]["gf"], 1),
                "ga": round(standings[team]["ga"], 1),
                "gd": round(standings[team]["gd"], 1),
                "qualifies": i < 2,  # Top 2 qualify (+ best 3rd places)
            } for i, (team, _) in enumerate(sorted_teams)],
            "matches": match_results,
            "qualified_teams": [t for t, _ in sorted_teams[:2]],
        }

    def predict_all_groups(self):
        """Simulate all 12 groups."""
        groups = self.dp.get_2026_groups()
        results = {}
        for group_name in sorted(groups.keys()):
            results[group_name] = self.predict_group(group_name)
        return results

    # ------------------------------------------------------------------
    # TOURNAMENT BRACKET SIMULATION
    # ------------------------------------------------------------------
    def simulate_tournament(self):
        """Simulate the entire tournament including group stage and knockouts."""
        group_results = self.predict_all_groups()
        
        # Collect all teams for Round of 32 (Top 2 from each group + 8 best 3rd placed)
        qualified_top2 = []
        third_places = []
        
        for grp, res in group_results.items():
            if "standings" in res:
                standings = res["standings"]
                if len(standings) >= 2:
                    qualified_top2.append(standings[0]["team"])
                    qualified_top2.append(standings[1]["team"])
                if len(standings) >= 3:
                    third_places.append(standings[2])
                    
        # Sort third places by points, then gd, then gf
        third_places.sort(key=lambda x: (x["points"], x["gd"], x["gf"]), reverse=True)
        qualified_3rd = [x["team"] for x in third_places[:8]]
        
        round_of_32_teams = qualified_top2 + qualified_3rd
        
        # Seed the 32 teams based on their ELO for deterministic bracket placement
        seeded_teams = []
        for team in round_of_32_teams:
            seeded_teams.append((team, self.dp._get_elo(self._resolve_team(team))))
            
        seeded_teams.sort(key=lambda x: x[1], reverse=True)
        teams_only = [t[0] for t in seeded_teams]
        
        # Seed order for standard 32-team single elimination
        seed_order = [0, 31, 15, 16, 8, 23, 7, 24, 3, 28, 12, 19, 11, 20, 4, 27, 
                      1, 30, 14, 17, 9, 22, 6, 25, 2, 29, 13, 18, 10, 21, 5, 26]
        
        current_round_teams = []
        for i in seed_order:
            if i < len(teams_only):
                current_round_teams.append(teams_only[i])
                
        # Fallback if somehow < 32 teams or mapping fails
        if len(current_round_teams) != 32:
            current_round_teams = teams_only 
            
        bracket = {
            "Round of 32": [],
            "Round of 16": [],
            "Quarterfinals": [],
            "Semifinals": [],
            "Final": [],
            "Winner": None
        }
        
        rounds = ["Round of 32", "Round of 16", "Quarterfinals", "Semifinals", "Final"]
        stage_importance_map = {
            "Round of 32": 1.5, "Round of 16": 2.0, 
            "Quarterfinals": 2.5, "Semifinals": 3.0, "Final": 4.0
        }
        
        # Simulate rounds
        for r_name in rounds:
            next_round_teams = []
            for i in range(0, len(current_round_teams), 2):
                if i+1 < len(current_round_teams):
                    t1 = current_round_teams[i]
                    t2 = current_round_teams[i+1]
                    
                    match_stats = self.simulate_knockout_match(t1, t2, stage_importance_map[r_name])
                    bracket[r_name].append(match_stats)
                    next_round_teams.append(match_stats["winner"])
                else:
                    next_round_teams.append(current_round_teams[i])
                    
            current_round_teams = next_round_teams
            if len(current_round_teams) == 1:
                bracket["Winner"] = current_round_teams[0]
                break
                
        return {
            "group_stage": group_results,
            "knockout_bracket": bracket
        }
        
    def simulate_knockout_match(self, home_team, away_team, stage_importance):
        """Play a knockout match, determine winner, generate detailed mock stats."""
        pred = self.predict_match(home_team, away_team, stage_importance=stage_importance)
        
        proba = pred["probabilities"]
        home_prob = proba.get(f"{home_team} Win", 0.33)
        draw_prob = proba.get("Draw", 0.33)
        away_prob = proba.get(f"{away_team} Win", 0.33)
        
        home_elo = pred["elo_home"]
        away_elo = pred["elo_away"]
        
        # No draws in knockout, allocate draw prob
        home_win_prob = home_prob + draw_prob/2
        away_win_prob = away_prob + draw_prob/2
        
        if home_win_prob > away_win_prob:
            winner, loser = home_team, away_team
            winner_prob = home_win_prob
        elif away_win_prob > home_win_prob:
            winner, loser = away_team, home_team
            winner_prob = away_win_prob
        else:
            if home_elo >= away_elo:
                winner, loser = home_team, away_team
                winner_prob = home_win_prob
            else:
                winner, loser = away_team, home_team
                winner_prob = away_win_prob
                
        # Generate stats deterministically
        import hashlib
        seed_str = f"{home_team}_{away_team}_{stage_importance}"
        seed = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2**32)
        rng = np.random.RandomState(seed)
        
        home_poss = int(np.clip(rng.normal(50 + (home_elo - away_elo) * 0.05, 8), 30, 70))
        away_poss = 100 - home_poss
        
        winner_goals = int(np.clip(rng.poisson(1.5 + (winner_prob - 0.5)*2), 1, 5))
        loser_goals = int(np.clip(rng.poisson(0.8), 0, winner_goals - 1))
        
        home_goals = winner_goals if winner == home_team else loser_goals
        away_goals = loser_goals if winner == home_team else winner_goals
        
        def get_scorers(team, num_goals):
            players = self.get_team_players(team)
            if not players or num_goals == 0: return []
            
            attackers = [p for p in players if p.get("position", "") in ["Attack", "Midfield", "Forward", "Attacker", "Right Winger", "Left Winger", "Centre-Forward", "Attacking Midfield"]]
            if not attackers: attackers = players[:11]
            
            weights = [p.get("market_value", 0) + 100000 for p in attackers]
            total_w = sum(weights)
            probs = [w/total_w for w in weights]
            
            picked = rng.choice(len(attackers), size=num_goals, p=probs, replace=True)
            return [str(attackers[i].get("name", "Unknown")).split(" (")[0] for i in picked]
            
        home_scorers = get_scorers(home_team, home_goals)
        away_scorers = get_scorers(away_team, away_goals)
        
        home_s_str = [f"{name} {rng.randint(1, 120)}'" for name in home_scorers]
        away_s_str = [f"{name} {rng.randint(1, 120)}'" for name in away_scorers]
        
        return {
            "home": home_team, "away": away_team,
            "winner": winner, "loser": loser,
            "score": f"{home_goals} - {away_goals}",
            "home_goals": home_goals, "away_goals": away_goals,
            "stats": {
                "possession": {"home": home_poss, "away": away_poss},
                "shots": {"home": home_goals * rng.randint(2, 5) + rng.randint(2, 8), 
                          "away": away_goals * rng.randint(2, 5) + rng.randint(2, 8)},
                "yellow_cards": {"home": int(rng.poisson(1.5)), "away": int(rng.poisson(1.5))},
                "red_cards": {"home": int(rng.choice([0, 1], p=[0.95, 0.05])), 
                              "away": int(rng.choice([0, 1], p=[0.95, 0.05]))},
                "scorers": {"home": home_s_str, "away": away_s_str}
            }
        }

    # ------------------------------------------------------------------
    # PLAYER STATS LOOKUP
    # ------------------------------------------------------------------
    def get_player_stats(self, player_name):
        """Look up a player's stats, team, and team predictions."""
        profiles = self.dp.profiles
        nat_perf = self.dp.national_perf
        injuries = self.dp.injuries
        market_vals = self.dp.market_values

        # Search for player (case-insensitive partial match)
        mask = profiles["player_name"].str.contains(player_name, case=False, na=False)
        matches = profiles[mask]

        if len(matches) == 0:
            return {"error": f"Player '{player_name}' not found"}

        # Take the first match (most relevant)
        player = matches.iloc[0]
        pid = player["player_id"]

        # National team performance
        nat = nat_perf[nat_perf["player_id"] == pid]
        national_stats = []
        for _, row in nat.iterrows():
            national_stats.append({
                "team_id": int(row["team_id"]),
                "matches": int(row["matches"]),
                "goals": int(row["goals"]),
                "career_state": row["career_state"],
            })

        # Injuries
        player_injuries = injuries[injuries["player_id"] == pid]
        injury_list = []
        for _, row in player_injuries.head(10).iterrows():
            injury_list.append({
                "season": row["season_name"],
                "injury": row["injury_reason"],
                "days_missed": int(row["days_missed"]) if pd.notna(row["days_missed"]) else 0,
                "games_missed": int(row["games_missed"]) if pd.notna(row["games_missed"]) else 0,
            })

        # Market value
        mv = market_vals[market_vals["player_id"] == pid]
        current_value = float(mv["value"].iloc[0]) if len(mv) > 0 else 0

        # Calculate age
        dob = pd.to_datetime(player["date_of_birth"], errors="coerce")
        age = None
        if pd.notna(dob):
            age = round((pd.Timestamp("2026-06-01") - dob).days / 365.25, 1)

        # Find the national team name
        citizenship = player.get("citizenship", "Unknown")

        result = {
            "player_id": int(pid),
            "name": player["player_name"],
            "date_of_birth": str(player["date_of_birth"]),
            "age_at_wc2026": age,
            "nationality": citizenship,
            "position": player.get("position", "Unknown"),
            "main_position": player.get("main_position", "Unknown"),
            "foot": player.get("foot", "Unknown"),
            "height": float(player["height"]) if pd.notna(player["height"]) else None,
            "current_club": player.get("current_club_name", "Unknown"),
            "market_value": current_value,
            "national_team_stats": national_stats,
            "recent_injuries": injury_list,
            "total_injuries": len(player_injuries),
            "total_days_missed": int(player_injuries["days_missed"].sum()) if len(player_injuries) > 0 else 0,
        }

        return result

    def search_players(self, query, limit=20):
        """Search players by name."""
        profiles = self.dp.profiles
        mask = profiles["player_name"].str.contains(query, case=False, na=False)

        # Prioritize current national players
        matching = profiles[mask].copy()
        nat_current = self.dp.national_perf[
            self.dp.national_perf["career_state"] == "CURRENT_NATIONAL_PLAYER"
        ]["player_id"].unique()

        matching["is_current_national"] = matching["player_id"].isin(nat_current)
        matching = matching.sort_values("is_current_national", ascending=False)

        results = []
        for _, row in matching.head(limit).iterrows():
            results.append({
                "player_id": int(row["player_id"]),
                "name": row["player_name"],
                "position": row.get("main_position", "Unknown"),
                "nationality": row.get("citizenship", "Unknown"),
                "club": row.get("current_club_name", "Unknown"),
                "is_current_national": bool(row["is_current_national"]),
            })

        return results

    def get_top_players(self, limit=3000):
        """Get the top players by market value for searchable dropdowns."""
        if not hasattr(self, '_top_players_cache'):
            profiles = self.dp.profiles.merge(
                self.dp.market_values[["player_id", "value"]], on="player_id", how="left"
            )
            profiles = profiles.sort_values("value", ascending=False).head(limit)
            
            results = []
            for _, row in profiles.iterrows():
                results.append({
                    "player_id": int(row["player_id"]),
                    "name": row["player_name"],
                    "position": row.get("main_position", "Unknown"),
                    "nationality": row.get("citizenship", "Unknown"),
                    "club": row.get("current_club_name", "Unknown"),
                })
            self._top_players_cache = results
            
        return self._top_players_cache

    def get_team_players(self, team_name):
        """Get all players for a country (inclusive of current and former)."""
        # Resolve team name to potential citizenship strings
        search_terms = [team_name]
        if team_name == "South Korea": search_terms.append("Korea, South")
        if team_name == "USA": search_terms.append("United States")
        
        # Get profiles matching citizenship
        mask = self.dp.profiles["citizenship"].str.contains("|".join(search_terms), case=False, na=False)
        team_players = self.dp.profiles[mask].copy()

        # Add national performance data
        team_players = team_players.merge(
            self.dp.national_perf[["player_id", "matches", "goals", "career_state"]],
            on="player_id",
            how="left"
        )
        
        # Fill missing national stats
        team_players["matches"] = team_players["matches"].fillna(0)
        team_players["goals"] = team_players["goals"].fillna(0)

        # Add market values
        team_players = team_players.merge(
            self.dp.market_values[["player_id", "value"]],
            on="player_id",
            how="left",
        )

        # Calculate age
        team_players["dob"] = pd.to_datetime(team_players["date_of_birth"], errors="coerce")
        team_players["age"] = ((pd.Timestamp("2026-06-01") - team_players["dob"]).dt.days / 365.25).round(1)

        # Sort by: Current National Player first, then Market Value, then Caps
        team_players["is_current"] = team_players["career_state"] == "CURRENT_NATIONAL_PLAYER"
        team_players = team_players.sort_values(["is_current", "value", "matches"], ascending=[False, False, False])

        players = []
        for _, row in team_players.iterrows():
            players.append({
                "player_id": int(row["player_id"]),
                "name": row["player_name"],
                "position": row.get("main_position", "Unknown"),
                "club": row.get("current_club_name", "Unknown"),
                "caps": int(row["matches"]),
                "goals": int(row["goals"]),
                "market_value": float(row["value"]) if pd.notna(row["value"]) else 0,
                "age": float(row["age"]) if pd.notna(row["age"]) else None,
                "status": row["career_state"]
            })

        return players

    # ------------------------------------------------------------------
    # CUSTOM PLAYING XI ANALYSIS
    # ------------------------------------------------------------------
    def analyze_custom_xi(self, player_ids, opponent_team):
        """Analyze a custom Playing XI against an opponent team."""
        if len(player_ids) != 11:
            return {"error": "Please select exactly 11 players"}

        # Get player data for selected XI
        selected = self.dp.profiles[self.dp.profiles["player_id"].isin(player_ids)].copy()

        # Get national performance
        nat = self.dp.national_perf[self.dp.national_perf["player_id"].isin(player_ids)]

        # Get market values
        vals = self.dp.market_values[self.dp.market_values["player_id"].isin(player_ids)]

        # Compute XI strength metrics
        total_caps = nat["matches"].sum()
        total_goals = nat["goals"].sum()
        avg_caps = nat["matches"].mean()
        avg_goals = nat["goals"].mean()

        total_value = vals["value"].sum() if len(vals) > 0 else 0
        avg_value = vals["value"].mean() if len(vals) > 0 else 0

        # Position balance
        positions = selected["main_position"].value_counts().to_dict()

        # Get the team these players belong to (most common citizenship)
        team = selected["citizenship"].mode().iloc[0] if len(selected) > 0 else "Unknown"

        # Compare with opponent
        opp_players = self.get_team_players(opponent_team)
        opp_total_value = sum(p["market_value"] for p in opp_players[:11])
        opp_avg_caps = np.mean([p["caps"] for p in opp_players[:11]]) if opp_players else 0
        opp_total_goals = sum(p["goals"] for p in opp_players[:11])

        # Compute strength score (normalized)
        xi_strength = (
            (total_value / 1e6 if total_value > 0 else 0) * 0.3 +
            total_caps * 0.3 +
            total_goals * 0.2 +
            len(positions) * 10 * 0.2  # Squad diversity bonus
        )

        opp_strength = (
            (opp_total_value / 1e6 if opp_total_value > 0 else 0) * 0.3 +
            opp_avg_caps * 11 * 0.3 +
            opp_total_goals * 0.2 +
            30 * 0.2
        )

        # Predict match using team-level features + adjustment
        match_pred = self.predict_match(team, opponent_team)

        # Adjust probabilities based on squad strength ratio
        strength_ratio = xi_strength / opp_strength if opp_strength > 0 else 1.0
        adjustment = np.clip((strength_ratio - 1.0) * 0.15, -0.2, 0.2)

        adjusted_proba = match_pred["probabilities"].copy()
        keys = list(adjusted_proba.keys())
        adjusted_proba[keys[0]] = np.clip(adjusted_proba[keys[0]] + adjustment, 0.05, 0.9)
        adjusted_proba[keys[2]] = np.clip(adjusted_proba[keys[2]] - adjustment, 0.05, 0.9)
        # Normalize
        total = sum(adjusted_proba.values())
        adjusted_proba = {k: v / total for k, v in adjusted_proba.items()}

        return {
            "your_team": team,
            "opponent": opponent_team,
            "xi_stats": {
                "total_caps": int(total_caps),
                "total_goals": int(total_goals),
                "average_caps": round(float(avg_caps), 1),
                "average_goals": round(float(avg_goals), 1),
                "total_market_value": float(total_value),
                "positions": positions,
            },
            "prediction": adjusted_proba,
            "base_prediction": match_pred["probabilities"],
            "strength_ratio": round(float(strength_ratio), 2),
            "selected_players": [
                {
                    "name": row["player_name"],
                    "position": row.get("main_position", "Unknown"),
                }
                for _, row in selected.iterrows()
            ],
        }

    # ------------------------------------------------------------------
    # INJURY IMPACT ANALYSIS
    # ------------------------------------------------------------------
    def analyze_injury_impact(self, team_name, injured_player_ids):
        """Analyze the impact of player injuries on a team's predictions."""
        # Get full squad
        all_players = self.get_team_players(team_name)
        if not all_players:
            return {"error": f"No players found for {team_name}"}

        # Identify injured players
        injured = [p for p in all_players if p["player_id"] in injured_player_ids]
        healthy = [p for p in all_players if p["player_id"] not in injured_player_ids]

        # Compute squad metrics with and without injured players
        full_value = sum(p["market_value"] for p in all_players[:23])
        full_caps = sum(p["caps"] for p in all_players[:23])
        full_goals = sum(p["goals"] for p in all_players[:23])

        injured_value = sum(p["market_value"] for p in injured)
        injured_caps = sum(p["caps"] for p in injured)
        injured_goals = sum(p["goals"] for p in injured)

        reduced_value = max(0, full_value - injured_value)
        value_loss_pct = (min(100, injured_value / full_value * 100)) if full_value > 0 else 0

        # Get team's group and predict with/without injuries
        groups = self.dp.get_2026_groups()
        team_group = None
        for grp, teams in groups.items():
            found = False
            for t in teams:
                if team_name.lower() in t.lower() or t.lower() in team_name.lower():
                    team_group = grp
                    found = True
                    break
            if found: break

        group_predictions = None
        if team_group:
            # Pass the injuries to the group simulation!
            injuries_dict = {team_name: injured_player_ids}
            group_predictions = self.predict_group(team_group, injuries_dict=injuries_dict)

        return {
            "team": team_name,
            "group": team_group,
            "injured_players": injured,
            "squad_impact": {
                "full_squad_value": full_value,
                "reduced_squad_value": reduced_value,
                "value_loss": injured_value,
                "value_loss_percentage": round(value_loss_pct, 1),
                "caps_lost": int(injured_caps),
                "goals_lost": int(injured_goals),
            },
            "healthy_squad_size": len(healthy),
            "group_predictions": group_predictions,
        }

    # ------------------------------------------------------------------
    # HEAD-TO-HEAD
    # ------------------------------------------------------------------
    def get_head_to_head(self, team1, team2):
        """Get historical head-to-head World Cup record."""
        df = self.dp.matches

        h2h_home = df[
            (df["Home Team Name"].str.contains(team1, case=False, na=False)) &
            (df["Away Team Name"].str.contains(team2, case=False, na=False))
        ]
        h2h_away = df[
            (df["Home Team Name"].str.contains(team2, case=False, na=False)) &
            (df["Away Team Name"].str.contains(team1, case=False, na=False))
        ]

        matches = []
        team1_wins = 0
        team2_wins = 0
        draws = 0

        for _, row in pd.concat([h2h_home, h2h_away]).iterrows():
            result = row["Result"]
            home = row["Home Team Name"]

            if result == "home team win":
                winner = home
            elif result == "away team win":
                winner = row["Away Team Name"]
            else:
                winner = "Draw"

            if team1.lower() in winner.lower():
                team1_wins += 1
            elif team2.lower() in winner.lower():
                team2_wins += 1
            else:
                draws += 1

            matches.append({
                "tournament": row["tournament Name"],
                "date": row["Match Date"],
                "home": home,
                "away": row["Away Team Name"],
                "score": row["Score"],
                "result": result,
            })

        return {
            "team1": team1,
            "team2": team2,
            "total_matches": len(matches),
            f"{team1}_wins": team1_wins,
            f"{team2}_wins": team2_wins,
            "draws": draws,
            "matches": matches,
        }


# ------------------------------------------------------------------
# Standalone test
# ------------------------------------------------------------------
if __name__ == "__main__":
    engine = PredictionEngine()
    engine.load()

    print("\n" + "=" * 60)
    print("TEST: Predict Brazil vs Germany")
    print("=" * 60)
    result = engine.predict_match("Brazil", "Germany")
    for k, v in result.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("TEST: Group A Simulation")
    print("=" * 60)
    group = engine.predict_group("Group A")
    if "standings" in group:
        for s in group["standings"]:
            print(f"  {s['position']}. {s['team']} - {s['points']} pts (W{s['w']} D{s['d']} L{s['l']})")

    print("\n" + "=" * 60)
    print("TEST: Player Stats - Cristiano Ronaldo")
    print("=" * 60)
    player = engine.get_player_stats("Cristiano Ronaldo")
    for k, v in player.items():
        if k not in ["recent_injuries", "national_team_stats"]:
            print(f"  {k}: {v}")
