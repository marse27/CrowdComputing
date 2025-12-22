import pandas as pd
from pathlib import Path

# Load the raw survey data
filename = "export_friends_2.csv"
path = Path(f"raw_data/{filename}")
df = pd.read_csv(path, header=0, skiprows=[1,2])
total_responses = len(df)
print(f"Total responses: {total_responses}")

MIN_DURATION = 0 * 60 
MAX_DURATION = 30 * 60

ATTENTION_CHECK_COLUMN = "Q_attention"
ATTENTION_CHECK_CORRECT = "Strongly disagree"

PROLIFIC_ID_COL = "q1"
DURATION_COL = "duration"

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()
    df_clean.columns = (
        df_clean.columns
        .str.strip()
        .str.lower()
        .str.replace(" (in seconds)", "", regex=False)
        .str.replace(" ", "_")
        .str.replace(r"[^\w_]", "", regex=True)
    )
    
    df_clean = df_clean[df_clean['finished'] == True]
    finished_responses = len(df_clean)
    print(f"Unfinished responses: {total_responses - finished_responses}")
    
    #[df_clean[ATTENTION_CHECK_COLUMN] == ATTENTION_CHECK_CORRECT]
    attention_checked_responses = len(df_clean)
    print(f"Amount that failed attention check: {finished_responses - attention_checked_responses}")
    
    if PROLIFIC_ID_COL in df_clean.columns.to_list():
        df_clean.drop_duplicates(subset=PROLIFIC_ID_COL)
        deduplicated_responses = len(df_clean)
        print(f"Duplicate responses removed: {attention_checked_responses - deduplicated_responses}")
    else:
        deduplicated_responses = attention_checked_responses
    
    duration = df_clean[DURATION_COL].astype(int)
    df_clean = df_clean[
        (duration >= MIN_DURATION) &
        (duration <= MAX_DURATION)
    ]
    duration_filtered_responses = len(df_clean)
    print(f"Responses filtered by duration: {deduplicated_responses - duration_filtered_responses}")

    return df_clean


force_data_cleaning = False
output_path = Path(f"cleaned_data/{filename}")
if output_path.exists() and not force_data_cleaning:
	print(f"Cleaned data already exists at {output_path}, skipping cleaning step.")
	df_clean = pd.read_csv(output_path)
else:
	df_clean = clean_data(df)
	print(f"Cleaned responses: {len(df_clean)}")
	df_clean.to_csv(output_path, index=False)

def clean_demographic_text(series: pd.Series) -> pd.Series:
    return (
        series
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .replace({"nan": None})
    )

import matplotlib.pyplot as plt

def plot_demographic_distribution(
    df: pd.DataFrame,
    column: str,
    title: str,
    min_count: int = 0
):
    """
    Create a bar plot showing the distribution of a text-based demographic variable.
    """
    data = clean_demographic_text(df[column]).dropna()

    counts = data.value_counts()
    counts = counts[counts >= min_count]

    plt.figure()
    counts.plot(kind="bar")
    plt.title(title)
    plt.xlabel("")
    plt.ylabel("Number of participants")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

DEMOGRAPHIC_LABELS = {
    "q2": "Age",
    "q3_1": "Country of Residence",
    "q4_1": "Country of Birth",
    "q5_1": "Native Language",
    "q6": "Highest Education Level",
    "q7": "Field of Study / Work",
    # "q7_8_text": "Specific Field of Study / Work",
    "q8": "Employment Status",
    "q9": "Financial Situation",
}

for col, label in DEMOGRAPHIC_LABELS.items():
    if col in df_clean.columns:
        plot_demographic_distribution(
            df_clean,
            column=col,
            title=f"Distribution of {label}",
            min_count=1   # suppress singletons if desired
        )