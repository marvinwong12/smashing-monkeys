import pandas as pd
import soccerdata as sd


class PremierLeagueScoutEngine:

    def __init__(self, season="2025-2026"):
        """Initializes the FBref scraper for the English Premier League."""
        print(f"Initializing FBref Scraper for Premier League, Season: {season}...")
        self.fbref = sd.FBref(leagues="ENG-Premier League", seasons=season)

    def _flatten_columns(self, df):
        """FBref data has MultiIndex columns.

        This flattens them to 'Category_Metric' strings.
        """
        df.columns = [
            f"{col[0]}_{col[1]}".strip("_") if isinstance(col, tuple) else col
            for col in df.columns
        ]
        return df

    def get_market_data(self):
        """Fetches and merges player data across standard, passing, and defensive metrics."""
        print("Fetching player seasonal stats from FBref...")

        # 1. Fetch distinct stat types
        std_stats = self.fbref.read_player_season_stats(stat_type="standard")
        passing_stats = self.fbref.read_player_season_stats(
            stat_type="passing"
        )
        defense_stats = self.fbref.read_player_season_stats(stat_type="defense")

        # 2. Flatten MultiIndex columns to make manipulation simple
        std_stats = self._flatten_columns(std_stats)
        passing_stats = self._flatten_columns(passing_stats)
        defense_stats = self._flatten_columns(defense_stats)

        # 3. Combine DataFrames on the Index (League, Season, Team, Player)
        # We only take unique columns from subsequent frames to avoid duplicate metadata columns
        merged_df = std_stats.copy()

        passing_cols_to_use = passing_stats.columns.difference(
            merged_df.columns
        )
        merged_df = merged_df.join(passing_stats[passing_cols_to_use])

        defense_cols_to_use = defense_stats.columns.difference(
            merged_df.columns
        )
        merged_df = merged_df.join(defense_stats[defense_cols_to_use])

        # Reset the multi-index to make filtering easier
        return merged_df.reset_index()

    def find_undervalued_gems(self, min_90s=5.0, position="MF"):
        """Filters the master dataframe based on targeted 'Moneyball' metrics."""
        df = self.get_market_data()

        # Print columns dynamically if you need to debug available metrics
        # print(df.columns.tolist())

        # Filter by position and minimum minutes played (to filter out noise/small samples)
        # Positions usually look like: 'MF', 'DF', 'FW', 'GK', 'MF,FW' etc.
        df_filtered = df[
            (df["Prd_Pos"].str.contains(position, na=False))
            & (df["Playing Time_90s"] >= min_90s)
        ].copy()

        # --- THE MONEYBALL FORMULA ---
        # Instead of raw numbers, we look at performance normalized Per 90 Minutes.
        # Let's track Progressive Passes and Passes into the Penalty Area.
        df_filtered["Progressive_Passes_Per90"] = (
            df_filtered["Total_PrgP"] / df_filtered["Playing Time_90s"]
        )
        df_filtered["Passes_Into_Box_Per90"] = (
            df_filtered["Total_PPA"] / df_filtered["Playing Time_90s"]
        )

        # Sort by elite progressive passing profiles
        results = df_filtered.sort_values(
            by="Progressive_Passes_Per90", ascending=False
        )

        # Select a clean subset of columns to hand off to the next Agent
        output_cols = [
            "player",
            "team",
            "Prd_Pos",
            "Prd_Age",
            "Playing Time_90s",
            "Progressive_Passes_Per90",
            "Passes_Into_Box_Per90",
            "Expected_xAG",  # Expected Assisted Goals
        ]

        return results[output_cols].head(10)


# --- Quick Test Execution ---
if __name__ == "__main__":
    scout = PremierLeagueScoutEngine(season="2024-2025")
    # Let's search for Midfielders who excel at breaking low-blocks
    gem_list = scout.find_undervalued_gems(min_90s=10.0, position="MF")

    print("\n--- TOP 10 RECRUITMENT TARGETS FOUND ---")
    print(gem_list.to_string(index=False))