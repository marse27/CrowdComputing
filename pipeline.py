import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

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

def clean_multiple_choice_text(series: pd.Series) -> pd.Series:
    return (
        series
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.slice(0, 10)
        .replace({"nan": None})
    )

def plot_multiple_choice_distribution(
    df: pd.DataFrame,
    column: str,
    title: str,
    ax,
    min_count: int = 0,
    bin_numeric: bool = False
):
    """
    Create a bar plot showing the distribution of a text-based demographic variable.
    """
    data = df[column].dropna()

    # If numeric and should be binned (e.g., q18)
    if bin_numeric:
        data = pd.to_numeric(data, errors="coerce").dropna()
        data = bin_1_to_100(data)
    else:
        data = clean_multiple_choice_text(data)

    counts = data.value_counts()
    counts = counts[counts >= min_count]

    counts.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("Participants")
    ax.tick_params(axis="x", rotation=45)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

def bin_1_to_100(series: pd.Series):
    bins = list(range(0, 101, 10))
    labels = [f"{i+1}-{i+10}" for i in range(0, 100, 10)]
    return pd.cut(series, bins=bins, labels=labels, include_lowest=True)

DEMOGRAPHIC_QUESTIONS = {
    "q2": "Age",
    "q3_1": "Country of Residence",
    "q4_1": "Country of Birth",
    "q5_1": "Native Language",
    "q6": "Highest Education Level",
    "q7": "Field of Study / Work",
    "q8": "Employment Status",
    "q9": "Financial Situation",
}
mask = (df_clean["q7"] == "Other:") & (df_clean["q7_8_text"].notna())
df_clean.loc[mask, "q7"] = df_clean.loc[mask, "q7_8_text"]

questions = [
    (col, label) for col, label in DEMOGRAPHIC_QUESTIONS.items()
    if col in df_clean.columns
]

fig, axes = plt.subplots(nrows=2, ncols=4)
axes = axes.flatten()  # makes indexing easier

for ax, (col, label) in zip(axes, questions):
    plot_multiple_choice_distribution(
        df_clean,
        column=col,
        title=label,
        ax=ax,
        min_count=1,
    )

# Hide unused subplots (if fewer than 8 questions)
for ax in axes[len(questions):]:
    ax.axis("off")

plt.tight_layout()
plt.show()

GENERAL_AI_QUESTIONS = {
    "q10": "Harmfulness of AI",
    "q11": "Can AI become conscious",
    "q12": "What type of consciousness",
    "q13": "Does AI make mistakes",
    #"q14": "Mistake explanation",
    "q15": "Importance of AI detection",
    "q16": "Can you detect AI-generated content",
    "q17_1": "Familiarity with generative AI",
    "q18_1": "Familiarity with ChatGPT",
    "q18_2": "Familiarity with DALL·E",
    "q18_3": "Familiarity with Sora",
}

questions = [
    (col, label) for col, label in GENERAL_AI_QUESTIONS.items()
    if col in df_clean.columns
]

fig, axes = plt.subplots(nrows=2, ncols=5)
axes = axes.flatten()  # makes indexing easier

for ax, (col, label) in zip(axes, questions):
    plot_multiple_choice_distribution(
        df_clean,
        column=col,
        title=label,
        ax=ax,
        min_count=1,
        bin_numeric=col.startswith("q18")
    )

# Hide unused subplots (if fewer than 8 questions)
for ax in axes[len(questions):]:
    ax.axis("off")

plt.tight_layout()
plt.show()