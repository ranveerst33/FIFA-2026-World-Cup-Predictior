"""
FIFA 2026 World Cup Prediction - Data Processing & Feature Engineering
======================================================================
Loads all available datasets, computes ELO ratings, builds team/player features,
and produces a clean training dataset for ML models.
"""

import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class DataProcessor:
    """End-to-end data processing pipeline for FIFA World Cup prediction."""

    def __init__(self):
        self.base_dir = BASE_DIR
        self.elo_ratings = {}
        self.team_stats = {}
        self.player_data = {}
        self.matches = None
        self.fifa2026 = None
        self.profiles = None
        self.national_perf = None
        self.injuries = None
        self.market_values = None
        self.team_details = None
        self.team_seasons = None

    # ------------------------------------------------------------------
    # DATA LOADING
    # ------------------------------------------------------------------
    def load_all_data(self):
        """Load every usable CSV into memory."""
        print("[1/7] Loading match history...")
        self.matches = pd.read_csv(
            os.path.join(self.base_dir, "allfifamatches.csv"),
            encoding="latin-1",
        )

        print("[2/7] Loading FIFA 2026 fixtures...")
        self.fifa2026 = pd.read_csv(
            os.path.join(self.base_dir, "fifa2026.csv")
        )

        print("[3/7] Loading player profiles...")
        self.profiles = pd.read_csv(
            os.path.join(self.base_dir, "player_profiles", "player_profiles.csv"),
            encoding="latin-1",
            low_memory=False,
        )

        print("[4/7] Loading national team performances...")
        self.national_perf = pd.read_csv(
            os.path.join(self.base_dir, "player_national_performances", "player_national_performances.csv"),
        )

        print("[5/7] Loading injury records...")
        self.injuries = pd.read_csv(
            os.path.join(self.base_dir, "player_injuries", "player_injuries.csv"),
        )

        print("[6/7] Loading market values...")
        self.market_values = pd.read_csv(
            os.path.join(self.base_dir, "player_latest_market_value", "player_latest_market_value.csv"),
        )

        print("[7/7] Loading team season stats...")
        self.team_seasons = pd.read_csv(
            os.path.join(self.base_dir, "team_competitions_seasons", "team_competitions_seasons.csv"),
            encoding="latin-1",
        )

        self.team_details = pd.read_csv(
            os.path.join(self.base_dir, "team_details", "team_details.csv"),
            encoding="latin-1",
        )

        print(f"[OK] All data loaded. Matches: {len(self.matches)}, Players: {len(self.profiles)}")
        return self

    # ------------------------------------------------------------------
    # ELO RATING SYSTEM
    # ------------------------------------------------------------------
    def _expected_score(self, elo_a, elo_b):
        return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

    def _update_elo(self, elo_a, elo_b, score_a, k=32):
        """Update ELO rating. score_a: 1=win, 0.5=draw, 0=loss."""
        expected = self._expected_score(elo_a, elo_b)
        new_elo_a = elo_a + k * (score_a - expected)
        new_elo_b = elo_b + k * ((1 - score_a) - (1 - expected))
        return new_elo_a, new_elo_b

    def compute_elo_ratings(self):
        """Compute ELO ratings from all historical World Cup matches."""
        print("Computing ELO ratings from 1930-2022...")
        self.elo_ratings = {}
        elo_history = []

        df = self.matches.sort_values("Match Date").reset_index(drop=True)

        for _, row in df.iterrows():
            home = row["Home Team Name"]
            away = row["Away Team Name"]
            result = row["Result"]

            if home not in self.elo_ratings:
                self.elo_ratings[home] = 1500
            if away not in self.elo_ratings:
                self.elo_ratings[away] = 1500

            if result == "home team win":
                score_home = 1.0
            elif result == "draw":
                score_home = 0.5
            else:
                score_home = 0.0

            # Tournament stage importance factor for K
            stage = str(row.get("Stage Name", "group stage")).lower()
            if "final" in stage and "quarter" not in stage and "semi" not in stage:
                k = 60
            elif "semi" in stage:
                k = 50
            elif "quarter" in stage:
                k = 45
            elif "round of" in stage:
                k = 40
            else:
                k = 32

            elo_before_home = self.elo_ratings[home]
            elo_before_away = self.elo_ratings[away]

            new_home, new_away = self._update_elo(
                self.elo_ratings[home], self.elo_ratings[away], score_home, k=k
            )
            self.elo_ratings[home] = new_home
            self.elo_ratings[away] = new_away

            elo_history.append({
                "home_team": home,
                "away_team": away,
                "elo_home_before": elo_before_home,
                "elo_away_before": elo_before_away,
                "elo_home_after": new_home,
                "elo_away_after": new_away,
            })

        self.elo_history = pd.DataFrame(elo_history)

        # Sort teams by current ELO
        sorted_elo = sorted(self.elo_ratings.items(), key=lambda x: x[1], reverse=True)
        print(f"[OK] ELO computed for {len(self.elo_ratings)} teams")
        print("  Top 10 teams by ELO:")
        for i, (team, elo) in enumerate(sorted_elo[:10]):
            print(f"    {i+1}. {team}: {elo:.0f}")

        return self

    # ------------------------------------------------------------------
    # TEAM STATISTICS (rolling from historical matches)
    # ------------------------------------------------------------------
    def compute_team_stats(self):
        """Build per-team rolling statistics from match history."""
        print("Computing team historical statistics...")
        df = self.matches.copy()

        team_stats = {}
        for team in set(df["Home Team Name"].unique()) | set(df["Away Team Name"].unique()):
            home_games = df[df["Home Team Name"] == team]
            away_games = df[df["Away Team Name"] == team]

            total_games = len(home_games) + len(away_games)
            if total_games == 0:
                continue

            home_wins = (home_games["Result"] == "home team win").sum()
            away_wins = (away_games["Result"] == "away team win").sum()
            total_wins = home_wins + away_wins

            home_draws = (home_games["Result"] == "draw").sum()
            away_draws = (away_games["Result"] == "draw").sum()
            total_draws = home_draws + away_draws

            total_losses = total_games - total_wins - total_draws

            goals_for = (
                home_games["Home Team Score"].sum() + away_games["Away Team Score"].sum()
            )
            goals_against = (
                home_games["Away Team Score"].sum() + away_games["Home Team Score"].sum()
            )

            # World Cup appearances (unique tournaments)
            tournaments_home = set(home_games["tournament Name"].unique())
            tournaments_away = set(away_games["tournament Name"].unique())
            wc_appearances = len(tournaments_home | tournaments_away)

            # Group stage vs knockout performance
            group_games = len(home_games[home_games["Group Stage"] == 1]) + len(
                away_games[away_games["Group Stage"] == 1]
            )
            knockout_games = len(home_games[home_games["Knockout Stage"] == 1]) + len(
                away_games[away_games["Knockout Stage"] == 1]
            )

            team_stats[team] = {
                "total_matches": total_games,
                "wins": total_wins,
                "draws": total_draws,
                "losses": total_losses,
                "win_rate": total_wins / total_games if total_games > 0 else 0,
                "draw_rate": total_draws / total_games if total_games > 0 else 0,
                "goals_for": goals_for,
                "goals_against": goals_against,
                "goals_per_match": goals_for / total_games if total_games > 0 else 0,
                "goals_conceded_per_match": goals_against / total_games if total_games > 0 else 0,
                "goal_diff_per_match": (goals_for - goals_against) / total_games if total_games > 0 else 0,
                "wc_appearances": wc_appearances,
                "group_stage_matches": group_games,
                "knockout_matches": knockout_games,
                "knockout_ratio": knockout_games / total_games if total_games > 0 else 0,
            }

        self.team_stats = team_stats
        print(f"[OK] Statistics computed for {len(team_stats)} teams")
        return self

    # ------------------------------------------------------------------
    # PLAYER-LEVEL FEATURES (aggregated per national team)
    # ------------------------------------------------------------------
    def compute_squad_features(self):
        """Aggregate player data to build squad-level features per national team."""
        print("Computing squad-level features...")

        # Merge profiles with national performance
        current_players = self.national_perf[
            self.national_perf["career_state"] == "CURRENT_NATIONAL_PLAYER"
        ].copy()

        merged = current_players.merge(
            self.profiles[["player_id", "player_name", "position", "main_position",
                           "citizenship", "current_club_name", "height", "date_of_birth", "foot"]],
            on="player_id",
            how="left",
        )

        # Add market values
        merged = merged.merge(
            self.market_values[["player_id", "value"]],
            on="player_id",
            how="left",
        )

        # Add injury info (count of injuries per player)
        injury_counts = (
            self.injuries.groupby("player_id")
            .agg(
                injury_count=("injury_reason", "count"),
                total_days_missed=("days_missed", "sum"),
                total_games_missed=("games_missed", "sum"),
            )
            .reset_index()
        )
        merged = merged.merge(injury_counts, on="player_id", how="left")
        merged["injury_count"] = merged["injury_count"].fillna(0)
        merged["total_days_missed"] = merged["total_days_missed"].fillna(0)
        merged["total_games_missed"] = merged["total_games_missed"].fillna(0)

        self.player_data = merged

        # ----- Map team_id to country/team name -----
        # We use the team mapping from player profiles (citizenship → national team)
        # Build a team_id → team_name lookup from the data itself
        # group by team_id to build squad features
        squad_features = {}
        for team_id, group in merged.groupby("team_id"):
            # Determine the team name from the most common citizenship
            citizenships = group["citizenship"].dropna()
            if len(citizenships) > 0:
                team_name = citizenships.mode().iloc[0]
            else:
                team_name = f"Team_{team_id}"

            squad_features[team_name] = {
                "squad_size": len(group),
                "avg_caps": group["matches"].mean(),
                "total_caps": group["matches"].sum(),
                "avg_goals": group["goals"].mean(),
                "total_goals": group["goals"].sum(),
                "avg_market_value": group["value"].mean(),
                "total_market_value": group["value"].sum(),
                "max_market_value": group["value"].max(),
                "avg_height": group["height"].mean(),
                "avg_injury_count": group["injury_count"].mean(),
                "total_injuries": group["injury_count"].sum(),
                "avg_days_missed": group["total_days_missed"].mean(),
                "attackers": len(group[group["main_position"] == "Attack"]),
                "midfielders": len(group[group["main_position"] == "Midfield"]),
                "defenders": len(group[group["main_position"] == "Defender"]),
                "goalkeepers": len(group[group["main_position"] == "Goalkeeper"]),
            }

            # Calculate player age
            dob = pd.to_datetime(group["date_of_birth"], errors="coerce")
            ages = (pd.Timestamp("2026-06-01") - dob).dt.days / 365.25
            squad_features[team_name]["avg_age"] = ages.mean()
            squad_features[team_name]["min_age"] = ages.min()
            squad_features[team_name]["max_age"] = ages.max()

        self.squad_features = squad_features
        print(f"[OK] Squad features computed for {len(squad_features)} national teams")
        return self

    # ------------------------------------------------------------------
    # BUILD TRAINING DATASET
    # ------------------------------------------------------------------
    def _get_team_feature(self, team_name, feature, default=0):
        """Safely get a team stat with fuzzy matching."""
        if team_name in self.team_stats:
            return self.team_stats[team_name].get(feature, default)
        # Fuzzy match
        for key in self.team_stats:
            if team_name.lower() in key.lower() or key.lower() in team_name.lower():
                return self.team_stats[key].get(feature, default)
        return default

    def _get_squad_feature(self, team_name, feature, default=0):
        """Safely get a squad feature with fuzzy matching."""
        if team_name in self.squad_features:
            return self.squad_features[team_name].get(feature, default)
        for key in self.squad_features:
            if team_name.lower() in key.lower() or key.lower() in team_name.lower():
                return self.squad_features[key].get(feature, default)
        return default

    def _get_elo(self, team_name, default=1500):
        """Get ELO for a team with fuzzy matching."""
        if team_name in self.elo_ratings:
            return self.elo_ratings[team_name]
        for key in self.elo_ratings:
            if team_name.lower() in key.lower() or key.lower() in team_name.lower():
                return self.elo_ratings[key]
        return default

    def build_training_data(self):
        """Build the final feature matrix for model training."""
        print("\nBuilding training dataset...")
        self.load_all_data()
        self.compute_elo_ratings()
        self.compute_team_stats()
        self.compute_squad_features()

        df = self.matches.copy()

        # --- Target ---
        target_map = {"home team win": 0, "draw": 1, "away team win": 2}
        df["target"] = df["Result"].map(target_map)
        df = df.dropna(subset=["target"])
        df["target"] = df["target"].astype(int)

        # --- Tournament stage importance ---
        def get_stage_importance(row):
            stage = str(row.get("Stage Name", "")).lower()
            if "final" in stage and "quarter" not in stage and "semi" not in stage:
                return 5
            elif "semi" in stage:
                return 4
            elif "quarter" in stage:
                return 3
            elif "round of" in stage:
                return 2
            return 1

        df["tournament_importance"] = df.apply(get_stage_importance, axis=1)

        # --- ELO features (using snapshot before each match via chronological order) ---
        elo_home_list, elo_away_list = [], []
        temp_elo = {}

        for _, row in df.sort_values("Match Date").iterrows():
            home = row["Home Team Name"]
            away = row["Away Team Name"]
            if home not in temp_elo:
                temp_elo[home] = 1500
            if away not in temp_elo:
                temp_elo[away] = 1500

            elo_home_list.append(temp_elo[home])
            elo_away_list.append(temp_elo[away])

            result = row["Result"]
            if result == "home team win":
                s = 1.0
            elif result == "draw":
                s = 0.5
            else:
                s = 0.0

            new_h, new_a = self._update_elo(temp_elo[home], temp_elo[away], s)
            temp_elo[home] = new_h
            temp_elo[away] = new_a

        df = df.sort_values("Match Date").reset_index(drop=True)
        df["elo_home"] = elo_home_list
        df["elo_away"] = elo_away_list
        df["elo_diff"] = df["elo_home"] - df["elo_away"]

        # --- Team historical features ---
        for prefix, team_col in [("home", "Home Team Name"), ("away", "Away Team Name")]:
            df[f"{prefix}_win_rate"] = df[team_col].apply(
                lambda t: self._get_team_feature(t, "win_rate")
            )
            df[f"{prefix}_goals_per_match"] = df[team_col].apply(
                lambda t: self._get_team_feature(t, "goals_per_match")
            )
            df[f"{prefix}_goals_conceded_pm"] = df[team_col].apply(
                lambda t: self._get_team_feature(t, "goals_conceded_per_match")
            )
            df[f"{prefix}_goal_diff_pm"] = df[team_col].apply(
                lambda t: self._get_team_feature(t, "goal_diff_per_match")
            )
            df[f"{prefix}_wc_appearances"] = df[team_col].apply(
                lambda t: self._get_team_feature(t, "wc_appearances")
            )
            df[f"{prefix}_knockout_ratio"] = df[team_col].apply(
                lambda t: self._get_team_feature(t, "knockout_ratio")
            )
            df[f"{prefix}_total_matches"] = df[team_col].apply(
                lambda t: self._get_team_feature(t, "total_matches")
            )

        # --- Attack/Defense strength ---
        df["home_attack_strength"] = df["home_goals_per_match"] - df["away_goals_conceded_pm"]
        df["away_attack_strength"] = df["away_goals_per_match"] - df["home_goals_conceded_pm"]
        df["home_defense_strength"] = df["away_goals_per_match"] - df["home_goals_conceded_pm"]
        df["away_defense_strength"] = df["home_goals_per_match"] - df["away_goals_conceded_pm"]

        # --- Experience diff ---
        df["wc_experience_diff"] = df["home_wc_appearances"] - df["away_wc_appearances"]
        df["win_rate_diff"] = df["home_win_rate"] - df["away_win_rate"]

        # Select final feature columns
        feature_cols = [
            "elo_home", "elo_away", "elo_diff",
            "home_win_rate", "away_win_rate", "win_rate_diff",
            "home_goals_per_match", "away_goals_per_match",
            "home_goals_conceded_pm", "away_goals_conceded_pm",
            "home_goal_diff_pm", "away_goal_diff_pm",
            "home_attack_strength", "away_attack_strength",
            "home_defense_strength", "away_defense_strength",
            "home_wc_appearances", "away_wc_appearances", "wc_experience_diff",
            "home_knockout_ratio", "away_knockout_ratio",
            "home_total_matches", "away_total_matches",
            "tournament_importance",
        ]

        X = df[feature_cols].copy()
        y = df["target"].copy()

        # Handle any remaining NaN
        X = X.fillna(0)

        print(f"[OK] Training dataset built: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"  Target distribution: {dict(y.value_counts().sort_index())}")
        print(f"  Features: {feature_cols}")

        self.X = X
        self.y = y
        self.feature_cols = feature_cols
        self.training_df = df

        return X, y, feature_cols

    # ------------------------------------------------------------------
    # FIFA 2026 TEAMS MAPPING
    # ------------------------------------------------------------------
    def get_2026_teams(self):
        """Extract confirmed teams for FIFA 2026."""
        df = self.fifa2026.copy()
        group_matches = df[df["Group"].notna()].copy()

        teams = set()
        for col in ["Home Team", "Away Team"]:
            for t in group_matches[col].unique():
                # Skip placeholder teams like "DEN/MKD/CZE/IRL"
                if "/" not in str(t) and "To be" not in str(t):
                    teams.add(t)

        return sorted(teams)

    def get_2026_groups(self):
        """Get group assignments for FIFA 2026."""
        df = self.fifa2026[self.fifa2026["Group"].notna()].copy()
        groups = {}
        for _, row in df.iterrows():
            grp = row["Group"]
            if grp not in groups:
                groups[grp] = set()
            for col in ["Home Team", "Away Team"]:
                team = row[col]
                if "/" not in str(team) and "To be" not in str(team):
                    groups[grp].add(team)

        return {k: sorted(v) for k, v in sorted(groups.items())}


# ------------------------------------------------------------------
# Standalone execution
# ------------------------------------------------------------------
if __name__ == "__main__":
    dp = DataProcessor()
    X, y, features = dp.build_training_data()

    print("\n" + "=" * 60)
    print("2026 CONFIRMED TEAMS:")
    print("=" * 60)
    teams_2026 = dp.get_2026_teams()
    for t in teams_2026:
        elo = dp._get_elo(t)
        print(f"  {t}: ELO={elo:.0f}")

    print("\n" + "=" * 60)
    print("2026 GROUPS:")
    print("=" * 60)
    groups = dp.get_2026_groups()
    for grp, teams in groups.items():
        print(f"  {grp}: {', '.join(teams)}")
