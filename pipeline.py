import re
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from collections import Counter
from sentence_transformers import SentenceTransformer
import umap
import hdbscan

MIN_DURATION = 0 * 60 
MAX_DURATION = 30 * 60

ATTENTION_CHECK_COLUMN = "Q_attention"
ATTENTION_CHECK_CORRECT = "Strongly disagree"

PROLIFIC_ID_COL = "q1"
DURATION_COL = "duration"

def load_data(filename: str) -> pd.DataFrame:
    path = Path(f"raw_data/{filename}")
    df = pd.read_csv(path, header=0, skiprows=[1,2])
    print(f"Total responses: {len(df)}")
    return df

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
    remaining_responses = len(df_clean)
    
    df_clean = df_clean[df_clean['finished'] == True]
    finished_responses = len(df_clean)
    print(f"Unfinished responses: {remaining_responses - finished_responses}")
    remaining_responses = finished_responses
    
    if ATTENTION_CHECK_COLUMN in df_clean.columns.to_list():
        df_clean = df_clean[df_clean[ATTENTION_CHECK_COLUMN] == ATTENTION_CHECK_CORRECT]
        attention_checked_responses = len(df_clean)
        print(f"Amount that failed attention check: {remaining_responses - attention_checked_responses}")
        remaining_responses = attention_checked_responses
    
    if PROLIFIC_ID_COL in df_clean.columns.to_list():
        df_clean.drop_duplicates(subset=PROLIFIC_ID_COL)
        deduplicated_responses = len(df_clean)
        print(f"Duplicate responses removed: {remaining_responses - deduplicated_responses}")
        remaining_responses = attention_checked_responses
    
    duration = df_clean[DURATION_COL].astype(int)
    df_clean = df_clean[
        (duration >= MIN_DURATION) &
        (duration <= MAX_DURATION)
    ]
    duration_filtered_responses = len(df_clean)
    print(f"Responses filtered by duration: {remaining_responses - duration_filtered_responses}")

    return df_clean

def clean_multiple_choice_text(series: pd.Series) -> pd.Series:
    return (
        series
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.slice(0, 15)
        .replace({"nan": None})
    )

def bin_1_to_100(series: pd.Series):
    bins = list(range(0, 101, 20))
    labels = [f"{i+1}-{i+20}" for i in range(0, 100, 20)]
    return pd.cut(
        series, 
        bins=bins, 
        labels=labels, 
        include_lowest=True
        )

def plot_multiple_choice_distribution(
    df: pd.DataFrame,
    column: str,
    title: str,
    ax,
    min_count: int = 0,
    bin_numeric: bool = False
):
    data = df[column].dropna()

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
    ax.tick_params(axis="x")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

def create_bar_plot_screen(
    df: pd.DataFrame,
    columns: list,
    title: str,
    bin_numeric_list: tuple = (),
):
    questions = [
        (col, label) for col, label in columns.items()
        if col in df.columns
    ]

    _, axes = plt.subplots(nrows=2, ncols=((len(questions) + 1) // 2), figsize=(20, 8))
    axes = axes.flatten()

    for ax, (col, label) in zip(axes, questions):
        plot_multiple_choice_distribution(
            df,
            column=col,
            title=label,
            ax=ax,
            min_count=1,
            bin_numeric=col.startswith(bin_numeric_list)
        )

    for ax in axes[len(questions):]:
        ax.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f"plots/{title.replace(' ', '_').lower()}.png")
    print(f"Saved plot for {title}")
    # plt.show()

def process_open_questions(df: pd.DataFrame, column: str) -> pd.Series:
    df[f"{column}_clean"] = df[column].fillna("Empty").apply(preprocess_open_text)

    categorize_open_ended_responses(df, f"{column}_clean")

    print(f"Sample categorizations for {column}:")
    max_width = 30  
    for o, c, l in zip(
        df[column], 
        df[f"{column}_clean"], 
        df[f"{column}_clean_label"]
    ): print(f"{str(o)[:max_width]:<{max_width}} → {str(c)[:max_width]:<{max_width}} → {str(l)[:max_width]:<{max_width}}"
    )

def preprocess_open_text(text: str):
    text = text.strip()  # Remove leading/trailing whitespace
    text = text.lower() # Lowercase

    # import emoji
    # from bs4 import BeautifulSoup
    # text = BeautifulSoup(text, "html.parser").get_text() # Remove HTML tags 
    # text = emoji.replace_emoji(text, replace="") # Remove emojis
    
    words = re.findall(r'\b\w+\b', text) # Tokenization using regex
    words = [w for w in words if len(w) > 3]  # Remove short words
    
    return " ".join(words)


def categorize_open_ended_responses(df: pd.DataFrame, column: str):
    # 1. Embed responses
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(df[column].dropna().tolist())

    # 2. Dimensionality reduction
    umap_embeddings = umap.UMAP(n_neighbors=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)

    # 3. Cluster
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric='euclidean')
    clusters = clusterer.fit_predict(umap_embeddings)

    # 4. Assign cluster labels to df
    df.loc[df[column].notna(), f"{column}_cluster"] = clusters

    # 5. Create human-readable labels for clusters
    cluster_labels = {}
    for cluster_id in df[f"{column}_cluster"].dropna().unique():
        texts = df.loc[df[f"{column}_cluster"] == cluster_id, f"{column}"]
        
        all_words = []
        for text in texts:
            all_words.extend(text.split())
        
        word_counts = Counter(all_words)
        most_common = [w for w, _ in word_counts.most_common(2)]
        cluster_labels[cluster_id] = ", ".join(most_common)

    df[f"{column}_label"] = df[f"{column}_cluster"].map(cluster_labels).fillna("Empty")


filename = "export_friends_4.csv"
force_data_cleaning = False
output_path = Path(f"cleaned_data/{filename}")
if output_path.exists() and not force_data_cleaning:
	print(f"Cleaned data already exists at {output_path}, skipping cleaning step.")
	df_clean = pd.read_csv(output_path)
else:
    df = load_data(filename)
    df_clean = clean_data(df)
    print(f"Cleaned responses: {len(df_clean)}")
    df_clean.to_csv(output_path, index=False)

DEMOGRAPHIC_QUESTIONS = {
    "q2": "Age",
    "q3_1": "Country of Residence",
    "q4_1": "Country of Birth",
    "q5_1": "AI Interaction Language",
    "q6": "Highest Education Level",
    "q7": "Field of Study / Work",
    "q8": "Employment Status",
    "q9": "Financial Situation",
}
mask = (df_clean["q7"] == "Other:") & (df_clean["q7_8_text"].notna())
df_clean.loc[mask, "q7"] = df_clean.loc[mask, "q7_8_text"]
create_bar_plot_screen(
    df_clean,
    DEMOGRAPHIC_QUESTIONS,
    title="Demographic Information",
)

GENERAL_AI_OPEN_QUESTIONS = {
    "q14": "Mistake explanation",
}

GENERAL_AI_MC_QUESTIONS = {
    "q10": "Harmfulness of AI",
    "q11": "Can AI become conscious",
    "q12": "What type of consciousness",
    "q13": "Does AI make mistakes",
    "q15": "Importance of AI detection",
    "q16": "Can you detect AI-generated content",
    "q17_1": "Familiarity with generative AI",
    "q18_1": "Familiarity with ChatGPT",
    "q18_2": "Familiarity with DALLE",
    "q18_3": "Familiarity with Sora",
}
create_bar_plot_screen(
    df_clean,
    GENERAL_AI_MC_QUESTIONS,
    title="Opinions about General AI",
    bin_numeric_list=("q18",)
)

CHATGPT_OPEN_QUESTIONS = {
    "q22": "What is ChatGPT",
    "q23": "How does ChatGPT work",
    "q24": "Strengths of ChatGPT",
    "q25": "Weaknesses of ChatGPT",
}

CHATGPT_MC_QUESTIONS = {
    "q21": "How often use ChatGPT",
    "q26_1": "Reuses information",
    "q26_2": "Can combine in new ways",
    "q26_3": "Can access information from the web",
    "q26_4": "Stores personal data",
    "q26_5": "Changes based on user",
    "q26_6": "Just follows rules",
    "q26_7": "Is influenced by the culture of the user",
    "q26_8": "Could produce harmful information",
    "q27_1": "Double check ChatGPT info",
}
create_bar_plot_screen(
    df_clean,
    CHATGPT_MC_QUESTIONS,
    title="Opinions about ChatGPT",
)

DALLE_OPEN_QUESTIONS = {
    "q31": "How does DALLE work",
}

DALLE_MC_QUESTIONS = {
    "q32_1": "Reuses information",
    "q32_2": "Can combine in new ways",
    "q32_3": "Can recreate artists' styles",
    "q32_4": "Uses content without permission",
    "q32_5": "Is a tool and not an creator",
    "q32_6": "Stores copies of generated images",
    "q32_7": "Creates culturally biased images",
}
create_bar_plot_screen(
    df_clean,
    DALLE_MC_QUESTIONS,
    title="Opinions about DALLE",
)

SORA_MC_QUESTIONS = {
    "q41_1": "Creates videos close to reality",
    "q41_2": "Mainly reuses content",
    "q41_3": "Makes faking videos easy",
    "q41_4": "Uses content without permission",
    "q41_5": "Will make it harder to trust videos",
}
create_bar_plot_screen(
    df_clean,
    SORA_MC_QUESTIONS,
    title="Opinions about Sora",
)

FEEDBACK_OPEN_QUESTIONS = {
    "q55": "Any suggestions for improvement",
}

FEEDBACK_MC_QUESTIONS = {
    "q51": "Any confusing questions",
    "q52_1": "Most understandable question",
    "q52_4": "Least understandable question",
    "q53": "How long did the survey feel",
    "q54_1": "How intuitive was the survey",
    "q54_2": "How easy was the navigation",
    "q54_3": "How interesting were the questions",
}
create_bar_plot_screen(
    df_clean,
    FEEDBACK_MC_QUESTIONS,
    title="Survey Feedback",
    bin_numeric_list=("q54",),
)

OPEN_QUESTIONS = GENERAL_AI_OPEN_QUESTIONS | CHATGPT_OPEN_QUESTIONS | DALLE_OPEN_QUESTIONS | FEEDBACK_OPEN_QUESTIONS

for col, label in OPEN_QUESTIONS.items():
    process_open_questions(df_clean, col)

OPEN_QUESTIONS_LABELED = {f"{col}_clean_label": label for col, label in OPEN_QUESTIONS.items()}

create_bar_plot_screen(
    df_clean,
    OPEN_QUESTIONS_LABELED,
    title=f"Categorized responses for open questions",
)


def cramers_v(chi2, contingency_table):
    n = contingency_table.values.sum()
    r, k = contingency_table.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))


def chi_square_all_pairings(
    df: pd.DataFrame,
    demographic_vars: list,
    question_vars: list,
    min_cell_count: int = 0,
    min_group_size: int = 0
) -> pd.DataFrame:
    """
    Runs Chi-square tests + Cramér's V for all demographic x question pairings.
    
    Parameters:
        df: cleaned DataFrame
        demographic_vars: list of demographic column names
        question_vars: list of question column names
        min_cell_count: minimum expected count per cell (for validity)
        min_group_size: minimum category size to keep
        
    Returns:
        DataFrame with test results
    """
    results = []

    for demo in demographic_vars:
        # Drop rare demographic categories
        demo_counts = df[demo].value_counts()
        valid_demo_values = demo_counts[demo_counts >= min_group_size].index
        df_demo = df[df[demo].isin(valid_demo_values)]

        for question in question_vars:
            # Build contingency table
            contingency = pd.crosstab(df_demo[demo], df_demo[question])

            # Skip degenerate tables
            if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                continue

            # Chi-square test
            chi2, p, dof, expected = chi2_contingency(contingency)

            # Skip if expected counts too small
            if (expected < min_cell_count).any():
                continue

            v = cramers_v(chi2, contingency)

            results.append({
                "demographic": demo,
                "question": question,
                "chi2": chi2,
                "dof": dof,
                "p_value": p,
                "cramers_v": v,
                "n": contingency.values.sum()
            })

    return pd.DataFrame(results)

def create_heatmap_of_associations(df: pd.DataFrame):
    heatmap_df = results_df.pivot(
        index="demographic",
        columns="question",
        values="p_value"
    )

    plt.figure(figsize=(20, 8))
    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".2f",
        cmap="viridis"
    )
    title = "Associations between Demographics and AI Beliefs"
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"plots/{title.replace(' ', '_').lower()}.png")
    print(f"Saved plot for {title}")
    # plt.show()

DEMOGRAPHIC_COLS = list(DEMOGRAPHIC_QUESTIONS.keys())

ALL_COLS = list((GENERAL_AI_MC_QUESTIONS | CHATGPT_MC_QUESTIONS | DALLE_MC_QUESTIONS | SORA_MC_QUESTIONS | OPEN_QUESTIONS_LABELED).keys())
results_df = chi_square_all_pairings(
    df_clean,
    demographic_vars=DEMOGRAPHIC_COLS,
    question_vars=ALL_COLS
)
print(results_df.sort_values("p_value").head(10))
create_heatmap_of_associations(results_df)