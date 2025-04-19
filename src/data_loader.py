import pandas as pd

def load_data(email_path, opened_path, clicked_path):
    email_df = pd.read_csv(email_path)
    opened_df = pd.read_csv(opened_path)
    clicked_df = pd.read_csv(clicked_path)

    
    opened_df['opened'] = 1
    clicked_df['clicked'] = 1

    # Merge
    df = email_df.merge(opened_df, on='email_id', how='left')
    df = df.merge(clicked_df, on='email_id', how='left')

    # Fill NaNs
    df['opened'] = df['opened'].fillna(0).astype(int)
    df['clicked'] = df['clicked'].fillna(0).astype(int)

    return df
