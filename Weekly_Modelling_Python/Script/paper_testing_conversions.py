import pandas as pd
import os

DATE = "23-06-2026" # dd-mm-yyyy format, change to run calculations for a different file

# --- Config ---
TOURS = {
    "Euro": f"../Output/Paper_Testing/Euro_{DATE}.xlsx",
    "PGA": f"../Output/Paper_Testing/PGA_{DATE}.xlsx",
}

MARKETS = {
    "Winner": ("Winner_Market", 1000),
    "Top 5": ("Top5_Market", 200),
    "Top 10": ("Top10_Market", 100),
    "Top 20": ("Top20_Market", 50),
}

# --- Load ---
data = {}

for tour, path in TOURS.items():
    if not os.path.exists(path):
        print(f"WARNING: {tour} file not found at {path}, skipping.")
        continue
    data[tour] = {
        market: pd.read_excel(path, sheet_name=sheet)
        for market, (sheet, _) in MARKETS.items()
    }

if not data:
    raise FileNotFoundError(f"No input files found for date {DATE}.")

# --- Functions ---
def get_market_profits(df):
    return df.columns[19], df.columns[21]  # keep your existing logic here

def calculate_fixed_stake(df, max_liability):
    return round(max_liability / max(df["Normalised_Model_Odds"]), 2)

def calculate_market_profits(df, stake_amount):
    lay = df[df["Bet"] == "LAY"]
    us_profit = round((lay["Profit"] / lay["Stake"]).sum(), 2)
    fs_profit = round(((lay["Profit"] / lay["Stake"]) * stake_amount).sum(), 2)
    return us_profit, fs_profit

# --- Compute ---
results = {}

for tour, markets in data.items():
    rows = []
    for market, df in markets.items():
        max_liability = MARKETS[market][1]
        back_profit, fl_lay_profit = get_market_profits(df)
        fixed_stake = calculate_fixed_stake(df, max_liability)
        us_profit, fs_profit = calculate_market_profits(df, fixed_stake)

        rows.append({
            "Market": market,
            "Back Profit": back_profit,
            "FL Lay Profit": fl_lay_profit,
            "£1 Stake Profit": us_profit,
            "FS Lay Profit": fs_profit,
        })

    summary_df = pd.DataFrame(rows)
    totals = summary_df.select_dtypes("number").sum().rename("Total")
    summary_df = pd.concat(
        [summary_df, totals.to_frame().T.assign(Market="Total")],
        ignore_index=True
    )
    results[tour] = summary_df

# --- Display ---
for tour, df in results.items():
    print(f"\n{tour}")
    print(df)