# src/ingestion/structured/compile_master.py
import os
import sqlite3 
import pandas as pd
from src.engine.aliases_config import MANUAL_ALIASES
from src.helper_functions import normalize_name
import difflib
from typing import List, Dict, Optional

def compile_master_dataset(cache_dir: str, understat_season: str = "2025", fbref_season: str = "2526") -> str:
    """
    Ingests raw performance, valuation, and tactical attributes, 
    resolves player identities, standardizes metrics, and saves a strictly-typed 
    master file to an SQLite database.
    """
    os.makedirs(cache_dir, exist_ok=True)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    
    # Define .db PATH
    db_path = os.path.join(project_root, "data", "scout_cache", "scout_platform.db") 
    
    sofifa_path = os.path.join(cache_dir, "sofifa_fc26_ratings.csv")
    
    attacking_file = os.path.join(cache_dir, f"understat_attacking_big5_{understat_season}.csv")
    defensive_file = os.path.join(cache_dir, f"fbref_defensive_big5_{fbref_season}.csv")
    valuation_file = os.path.join(cache_dir, "player_valuations_master.csv")

    # ==========================================
    # A. VERIFY & LOAD ALL DATA SOURCES
    # ==========================================
    required_files = [attacking_file, defensive_file, valuation_file, sofifa_path]
    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing required scouting data file: {file_path}")

    print("📖 Loading raw tracking, valuation, and rating matrices...")
    df_attack = pd.read_csv(attacking_file)
    df_defense = pd.read_csv(defensive_file)
    df_val = pd.read_csv(valuation_file)
    df_sofifa = pd.read_csv(sofifa_path)

    # Standardize ALL columns to lowercase to prevent casing mismatches across files
    df_attack.columns = df_attack.columns.str.lower()
    df_defense.columns = df_defense.columns.str.lower()
    df_val.columns = df_val.columns.str.lower()
    df_sofifa.columns = df_sofifa.columns.str.lower()

    # Drop colliding metadata columns from secondary files to avoid duplicate clutter (_x / _y)
    df_defense = df_defense.drop(columns=['team', 'league', 'age', 'nation'], errors='ignore')
    df_val = df_val.drop(columns=['team', 'league', 'position'], errors='ignore')
    df_sofifa = df_sofifa.drop(columns=['team', 'league', 'position', 'age', 'nation'], errors='ignore')

    # ==========================================
    # B. INITIALIZE TEMPORARY ALIGNMENT VECTORS
    # ==========================================
    # We use 'match_name' purely in memory as a normalized scratchpad for high-accuracy merges.
    # We will safely scrub it from the final dataframe before exporting to keep the schema clean.
    df_attack['match_name'] = df_attack['player'].apply(normalize_name)
    df_defense['match_name'] = df_defense['player'].apply(normalize_name)
    df_val['match_name'] = df_val['player'].apply(normalize_name)
    if 'match_name' in df_sofifa.columns:
        df_sofifa['match_name'] = df_sofifa['match_name'].apply(normalize_name)
    elif 'player' in df_sofifa.columns:
        df_sofifa['match_name'] = df_sofifa['player'].apply(normalize_name)

    print("Initiating offline ETL pipeline for Chief Scout Agent...")

    # ==========================================
    # C. FUZZY MATCH PERFORMANCE DATA (ANCHOR TO DEFENSE)
    # ==========================================
    print("Aligning mismatched player names between datasets (Anchoring to Defense)...")
    attack_names = df_attack['match_name'].unique()
    defense_names = df_defense['match_name'].unique()

    unmatched_attack = set(attack_names) - set(defense_names)

    def clean_pure_text(name_str):
        return "".join([c for c in str(name_str).lower() if c.isalnum()])

    name_corrections = {}
    for atk_name in unmatched_attack:
        pure_atk = clean_pure_text(atk_name)
        subset_matches = [
            def_name for def_name in defense_names 
            if pure_atk in clean_pure_text(def_name) or clean_pure_text(def_name) in pure_atk
        ]
        
        if subset_matches:
            name_corrections[atk_name] = subset_matches[0]
            continue

        matches = difflib.get_close_matches(atk_name, defense_names, n=1, cutoff=0.75)
        if matches:
            name_corrections[atk_name] = matches[0]

    df_attack['match_name'] = df_attack['match_name'].replace(name_corrections)

    # De-duplicate individual file structures prior to fusion
    df_attack = df_attack.drop_duplicates(subset=['match_name'], keep='first')
    df_defense = df_defense.drop_duplicates(subset=['match_name'], keep='first')
    
    print("🔀 Merging Attack & Defense tracking data...")
    master_df = pd.merge(df_attack, df_defense, on="match_name", how="outer", suffixes=('_atk', '_def'))
    
    # Consolidate string identity to clean defensive baseline name (keeps original spelling casing)
    master_df['player'] = master_df['player_def'].fillna(master_df['player_atk'])
    master_df = master_df.drop(columns=['player_def', 'player_atk'], errors='ignore')
    
    # Consolidate position mappings
    if 'position_x' in master_df.columns and 'position_y' in master_df.columns:
        master_df['position'] = master_df['position_x'].fillna(master_df['position_y']).fillna('UNKNOWN')
        master_df = master_df.drop(columns=['position_x', 'position_y'])

    # Enforce valid playing time samples
    master_df['minutes'] = pd.to_numeric(master_df['minutes'], errors='coerce').fillna(0)
    min_minutes_threshold = 270 
    master_df = master_df[master_df['minutes'] >= min_minutes_threshold]

    # ==========================================
    # D. INTEGRATE FINANCIALS
    # ==========================================
    print("Integrating Financial & Contract constraints...")
    df_val = df_val.drop_duplicates(subset=['match_name'], keep='first')
    
    # Merge the financial data
    master_df = pd.merge(master_df, df_val.drop(columns=['player'], errors='ignore'), on='match_name', how='left')
    
    # 1. Define our safe fallback values
    financial_defaults = {
        'annual_wage_eur': 0,
        'market_value_mln': 0.0,
        'age': 25,
        'nation': 'Unknown' # Using 'Unknown' to match our Section F JSON safety rules!
    }

    # 2. Safely create missing columns and fill NaNs generated by the merge
    for col, default_val in financial_defaults.items():
        if col not in master_df.columns:
            master_df[col] = default_val
        master_df[col] = master_df[col].fillna(default_val)
        
    # 3. Enforce strict data types
    master_df['annual_wage_eur'] = master_df['annual_wage_eur'].astype(int)
    master_df['market_value_mln'] = master_df['market_value_mln'].astype(float)
    master_df['age'] = master_df['age'].astype(int)
    master_df['nation'] = master_df['nation'].astype(str)

    # ==========================================
    # E. INTEGRATE SOFIFA (Tactics & Biometrics)
    # ==========================================
    print("🎮 Resolving Tactical & Biometric profiles via SoFIFA...")
    df_sofifa = df_sofifa.drop_duplicates(subset=['match_name'], keep='first')

    # Fallback to internal manual dictionary mapping logic if available, otherwise match clean name
    manual_aliases = getattr(df_sofifa, 'manual_aliases', {}) if 'manual_aliases' in dir() else {}
    master_df['sofifa_lookup_name'] = master_df['match_name'].map(manual_aliases).fillna(master_df['match_name'])
    
    master_df = pd.merge(master_df, df_sofifa.drop(columns=['player'], errors='ignore'), left_on='sofifa_lookup_name', right_on='match_name', how='left', suffixes=('', '_sofifa'))
    if 'match_name_sofifa' in master_df.columns:
        master_df = master_df.drop(columns=['match_name_sofifa'])
        
    # Token Fallback Matcher
    missing_sofifa = master_df['overall'].isna() | (master_df['overall'] == 0)
    valid_lookup = master_df['sofifa_lookup_name'].astype(str).str.strip().str.len() > 0
    unjoined = master_df[missing_sofifa & valid_lookup]
    
    if not unjoined.empty:
        print(f"Running token fallback resolution for {len(unjoined)} players...")
        sofifa_records = [{'tokens': set(str(row['match_name']).split()), 'data': row.to_dict()} for _, row in df_sofifa.iterrows()]
        sofifa_cols = [c for c in df_sofifa.columns if c not in ['match_name', 'player']]
        
        resolved = 0
        for idx, row in unjoined.iterrows():
            m_tokens = set(str(row['sofifa_lookup_name']).split())
            if len(m_tokens) < 2: 
                continue
            match = next((s['data'] for s in sofifa_records if m_tokens.issubset(s['tokens'])), None)
            if match:
                for col in sofifa_cols:
                    master_df.at[idx, col] = match[col]
                resolved += 1
        print(f"Token fallback resolved {resolved} aliases.")

        
    master_df = master_df.drop(columns=['sofifa_lookup_name'])

    # ==========================================
    # F. AGENT DATA STANDARDIZATION & EXPORT
    # ==========================================
    print("Finalizing data typings and schema formatting...")
    if 'preferred_foot' in master_df.columns:
        master_df['preferred_foot'] = master_df['preferred_foot'].astype(str).str.strip().str.capitalize()
        master_df['preferred_foot'] = master_df['preferred_foot'].replace({'Nan': 'Unknown', 'None': 'Unknown'})
    
    num_cols = master_df.select_dtypes(include=['number']).columns
    master_df[num_cols] = master_df[num_cols].fillna(0)

    obj_cols = master_df.select_dtypes(include=['object']).columns
    master_df[obj_cols] = master_df[obj_cols].fillna("Unknown")
    
    int_cols = ['matches', 'height_cm', 'weight_kg', 'weak_foot', 'skill_moves', 'overall', 'potential', 'yellow_cards', 'red_cards']
    for col in int_cols:
        if col in master_df.columns:
            master_df[col] = master_df[col].astype(int)

    # Drop 'match_name' here to maintain a single 'player' identity column
    if 'match_name' in master_df.columns:
        master_df = master_df.drop(columns=['match_name'])

    # ==========================================
    # CALCULATE PER-90 VOLUME METRICS
    # ==========================================
    print("Generating Per-90 metrics and stripping raw volume columns...")
    volume_metrics = [
        'goals', 'np_goals', 'xg', 'np_xg', 'assists', 'xa', 'key_passes', 'shots', 
        'xg_chain', 'xg_buildup', 'yellow_cards', 'red_cards', 
        'performance_tklw', 'performance_int', 'performance_fls', 
        'performance_crdy', 'ball_recoveries'
    ]
    
    cols_to_drop = []
    for col in volume_metrics:
        if col in master_df.columns:
            per90_name = f"{col}_per90"
            master_df[col] = pd.to_numeric(master_df[col], errors='coerce').fillna(0)
            # Calculate per90 based on minutes played
            master_df[per90_name] = ((master_df[col] / master_df['minutes'].replace(0, float('inf'))) * 90.0).round(2)
            cols_to_drop.append(col)
            
    # Drop the raw volume metrics to strictly enforce Per-90 analysis
    master_df = master_df.drop(columns=cols_to_drop)

    with sqlite3.connect(db_path) as conn:
        master_df.to_sql("players", conn, if_exists="replace", index=False)

    print(f"Master dataset successfully compiled to SQL! Target: {db_path} (Shape: {master_df.shape})")
    return db_path

if __name__ == "__main__":
    # Allows developers to run ingestion on demand via command line
    import argparse
    parser = argparse.ArgumentParser(description="Run Structured Data Matrix Compilations")
    parser.add_argument("--dir", type=str, default="./data/raw", help="Path to raw components storage folder")
    args = parser.parse_args()
    compile_master_dataset(cache_dir=args.dir)