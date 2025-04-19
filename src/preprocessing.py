def add_features(df):
    df["is_weekend"] = df["weekday"].isin(["Saturday", "Sunday"]).astype(int)
    
    # Hour bins: morning (5–11), afternoon (12–17), evening (18–23), night (0–4)
    def bin_hour(h):
        if 5 <= h <= 11:
            return "morning"
        elif 12 <= h <= 17:
            return "afternoon"
        elif 18 <= h <= 23:
            return "evening"
        else:
            return "night"
    df["hour_bin"] = df["hour"].apply(bin_hour)

    # Bin past purchases: low (0), medium (1–3), high (4+)
    def bin_purchases(p):
        if p == 0:
            return "none"
        elif 1 <= p <= 3:
            return "medium"
        else:
            return "high"
    df["purchase_bin"] = df["user_past_purchases"].apply(bin_purchases)

    return df
