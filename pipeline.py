import pandas as pd

# Load the raw survey data
filename = "export_1.csv"
path = f"raw_data/{filename}"
df = pd.read_csv(path, header=0, skiprows=[1,2])
print(f"Total responses: {len(df)}")
print(f"Columns: {df.columns.tolist()}")

MIN_DURATION = 240
MAX_DURATION = 1200

ATTENTION_CHECK_COLUMN = "Q_attention"
ATTENTION_CHECK_CORRECT = "Strongly disagree"

PROLIFIC_ID_COL = "PROLIFIC_PID"
DURATION_COL = "duration"

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(r"[^\w_]", "", regex=True)
    )
    
    df = df[df['finished'] == True]
    print(f"Finished responses: {len(df)}")
    #[df[ATTENTION_CHECK_COLUMN] == ATTENTION_CHECK_CORRECT]
    
    # df.drop_duplicates(subset=PROLIFIC_ID_COL)
    duration = df[DURATION_COL].astype(int)
    df[
        (duration >= MIN_DURATION) &
        (duration <= MAX_DURATION)
    ]
    
    # for col in scale_cols:
    #     if col in df.columns:
    #         df[col] = pd.to_numeric(df[col], errors="coerce")
            
    # for col in text_cols:
    #     if col in df.columns:
    #         df[col] = (
    #             df[col]
    #             .astype(str)
    #             .str.strip()
    #             .str.replace(r"\s+", " ", regex=True)
    #         )

    return df

df = clean_data(df)
print(f"Cleaned responses: {len(df)}")
output_path = f"cleaned_data/{filename}"
df.to_csv(output_path, index=False)