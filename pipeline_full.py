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
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch
import math

MIN_DURATION = 0 * 60 
MAX_DURATION = 30 * 60

ATTENTION_CHECK_1_COLUMN = "q21"
ATTENTION_CHECK_1_CORRECT = "Agree"

ATTENTION_CHECK_2_COLUMN = "q38"
ATTENTION_CHECK_2_CORRECT = "Apple"

CONSENT_CHECK_COLUMN = "q40"
CONSENT_CHECK_CORRECT = "Yes"

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
    
    if ATTENTION_CHECK_1_COLUMN in df_clean.columns.to_list():
        df_clean = df_clean[df_clean[ATTENTION_CHECK_1_COLUMN] == ATTENTION_CHECK_1_CORRECT]
        attention_checked_1_responses = len(df_clean)
        print(f"Amount that failed attention check 1: {remaining_responses - attention_checked_1_responses}")
        remaining_responses = attention_checked_1_responses

    if ATTENTION_CHECK_2_COLUMN in df_clean.columns.to_list():
        df_clean = df_clean[df_clean[ATTENTION_CHECK_2_COLUMN] == ATTENTION_CHECK_2_CORRECT]
        attention_checked_2_responses = len(df_clean)
        print(f"Amount that failed attention check 2: {remaining_responses - attention_checked_2_responses}")
        remaining_responses = attention_checked_2_responses
    
    if CONSENT_CHECK_COLUMN in df_clean.columns.to_list():
        df_clean = df_clean[df_clean[CONSENT_CHECK_COLUMN].str.startswith(CONSENT_CHECK_CORRECT)]
        consent_checked_responses = len(df_clean)
        print(f"Amount that failed consent check: {remaining_responses - consent_checked_responses}")
        remaining_responses = consent_checked_responses
    
    if PROLIFIC_ID_COL in df_clean.columns.to_list():
        df_clean.drop_duplicates(subset=PROLIFIC_ID_COL)
        deduplicated_responses = len(df_clean)
        print(f"Duplicate responses removed: {remaining_responses - deduplicated_responses}")
        remaining_responses = deduplicated_responses
    
    # duration = df_clean[DURATION_COL].astype(int)
    # df_clean = df_clean[
    #     (duration >= MIN_DURATION) &
    #     (duration <= MAX_DURATION)
    # ]
    # duration_filtered_responses = len(df_clean)
    # print(f"Responses filtered by duration: {remaining_responses - duration_filtered_responses}")

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
    bins = list(range(0, 101, 5))
    labels = [f"{i+1}-{i+5}" for i in range(0, 100, 5)]
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

    counts = counts.sort_index()
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
    plt.savefig(f"plots/full/{title.replace(' ', '_').lower()}.png")
    print(f"Saved plot for {title}")
    # plt.show()

def process_open_questions(df: pd.DataFrame, column: str) -> pd.Series:  

    # === AI-likeness features ===
    df[f"{column}_ppl"] = df[column].fillna("").apply(compute_perplexity)
    df[f"{column}_lexdiv"] = df[column].fillna("").apply(lexical_diversity)
    df[f"{column}_repeat"] = df[column].fillna("").apply(repetition_ratio)
    df[f"{column}_length"] = df[column].fillna("").str.split().apply(len)

    df[f"{column}_ai_score"] = (
        -df[f"{column}_ppl"].rank(pct=True) +     # lower ppl = more AI-like
        -df[f"{column}_lexdiv"].rank(pct=True) +  # lower diversity = more AI-like
        df[f"{column}_repeat"].rank(pct=True)
    ) / 3

    # df["ai_flag"] = df[f"{column}_ai_score"] > df[f"{column}_ai_score"].quantile(0.95)
    df[f"{column}_clean"] = df[column].fillna("Empty").apply(preprocess_open_text)

    categorize_open_ended_responses(df, f"{column}_clean")

    print(f"Sample categorizations for {column}:")
    max_width = 50  
    for o, c, l, s in zip(
        df[column], 
        df[f"{column}_clean"], 
        df[f"{column}_clean_label"],
        df[f"{column}_ai_score"]
    ): print(
            f"{str(o)[:max_width]:<{max_width}} → "
            f"{str(c)[:max_width]:<{max_width}} → "
            f"{str(l)[:max_width]:<{max_width}} | AI-score: {s:.2f}"
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

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

def compute_perplexity(text: str) -> float:
    if not text or len(text.split()) < 5:
        return float("nan")

    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        loss = model(**enc, labels=enc["input_ids"]).loss
    return math.exp(loss.item())

def lexical_diversity(text: str) -> float:
    tokens = text.split()
    if len(tokens) == 0:
        return float("nan")
    return len(set(tokens)) / len(tokens)

def repetition_ratio(text: str) -> float:
    tokens = text.split()
    if len(tokens) == 0:
        return float("nan")
    counts = Counter(tokens)
    return max(counts.values()) / len(tokens)


filename = "export_full_2.csv"
force_data_cleaning = True
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
    "q1": "Age",
    "q2_1": "Country of Residence",
    "q3_1": "Country of Birth",
    "q4_1": "AI Interaction Language",
    "q5": "Highest Education Level",
    "q6": "Living Area",
    "q7": "Field of Study / Work",
    "q8": "Employment Status",
    "q9": "Financial Situation",
}
mask = (df_clean["q7"] == "Other:") & (df_clean["q7_8_text"].notna())
df_clean.loc[mask, "q7"] = df_clean.loc[mask, "q7_8_text"]
df_clean["q1"] = 2025 - df_clean["q1"]
create_bar_plot_screen(
    df_clean,
    DEMOGRAPHIC_QUESTIONS,
    title="Demographic Information",
    bin_numeric_list=("q1",)
)

GENERAL_AI_OPEN_QUESTIONS = {
    "q16": "Mistake explanation",
    "q17": "No mistake explanation",
}

MAILS_QUESTIONS = {
    "q22_1": "Difference between AI and logic",
    "q22_2": "Create AI yourself",
    "q23_1": "Keep up-to-date with AI",
    "q23_2": "Handle frustration with AI",
    "q24_1": "Weigh consequences of AI use",
    "q24_2": "Design new AI applications",
    "q25_1": "Meaningful use of AI",
    "q25_2": "Difficulty of tasks",
    "q26_1": "Prevent AI from influencing you",
    "q26_2": "Access pros and cons of AI",
}
df_clean["mails_mean"] = (
    df_clean[list(MAILS_QUESTIONS.keys())]
    .apply(pd.to_numeric, errors="coerce")
    .mean(axis=1)
    .round(0)
)

GENERAL_AI_MC_QUESTIONS = {
    "q10": "Control over AI",
    "q11": "Harmfulness of AI",
    "q12": "Types of harm",
    "q13": "Can AI become conscious",
    "q14": "Type of consciousness",
    "q15": "Does AI make mistakes",
    "q18": "Frequency of mistakes",
    "q19": "Importance of AI detection",
    "q20": "Can you detect AI-generated content",
    "q27_1": "Familiarity with ChatGPT",
    "q27_2": "Familiarity with DALLE",
    "q27_3": "Familiarity with Sora",
} | { "mails_mean": "MAILS Score"}
create_bar_plot_screen(
    df_clean,
    GENERAL_AI_MC_QUESTIONS,
    title="Opinions about General AI",
)

CHATGPT_OPEN_QUESTIONS = {
    "q32": "How does ChatGPT work",
    "q33": "Why can ChatGPT answer",
    "q34": "What does ChatGPT learn from conversations",
    "q35": "What determines quality",
}

CHATGPT_MC_QUESTIONS = {
    "q30": "How often use ChatGPT",
    "q31": "Learns from interactions",
    "q36_1": "Reuses information",
    "q36_2": "Can combine in new ways",
    "q36_3": "Can access information from the web",
    "q36_4": "Stores personal data",
    "q36_5": "Changes based on user",
    "q36_6": "Just follows rules",
    "q36_7": "Is influenced by the culture of the user",
    "q36_8": "Could produce harmful information",
    "q37": "Double check ChatGPT info",
}
create_bar_plot_screen(
    df_clean,
    CHATGPT_MC_QUESTIONS,
    title="Opinions about ChatGPT",
)

DALLE_OPEN_QUESTIONS = {
    "q41": "How does DALLE work",
}

DALLE_MC_QUESTIONS = {
    "q42_1": "Reuses information",
    "q42_2": "Can combine in new ways",
    "q42_3": "Can recreate artists' styles",
    "q42_4": "Uses content without permission",
    "q42_5": "Is a tool and not an creator",
    "q42_6": "Stores copies of generated images",
    "q42_7": "Creates culturally biased images",
}
create_bar_plot_screen(
    df_clean,
    DALLE_MC_QUESTIONS,
    title="Opinions about DALLE",
)

SORA_MC_QUESTIONS = {
    "q51_1": "Creates videos close to reality",
    "q51_2": "Mainly reuses content",
    "q51_3": "Makes faking videos easy",
    "q51_4": "Uses content without permission",
    "q51_5": "Will make it harder to trust videos",
}
create_bar_plot_screen(
    df_clean,
    SORA_MC_QUESTIONS,
    title="Opinions about Sora",
)

OPEN_QUESTIONS = GENERAL_AI_OPEN_QUESTIONS | CHATGPT_OPEN_QUESTIONS | DALLE_OPEN_QUESTIONS

for col, label in OPEN_QUESTIONS.items():
    process_open_questions(df_clean, col)

OPEN_QUESTIONS_LABELED = {f"{col}_clean_label": label for col, label in OPEN_QUESTIONS.items()}

create_bar_plot_screen(
    df_clean,
    OPEN_QUESTIONS_LABELED,
    title=f"Categorized responses for open questions",
)

AI_SCORE_COLS = [
    col for col in df_clean.columns
    if col.endswith("_ai_score")
]
df_clean["ai_score_mean"] = (
    df_clean[AI_SCORE_COLS]
    .mean(axis=1)
    .round(3)
)
df_clean["ai_score_mean_z"] = (
    df_clean["ai_score_mean"] - df_clean["ai_score_mean"].mean()
) / df_clean["ai_score_mean"].std()
print(df_clean[["ai_score_mean", "ai_score_mean_z"]])



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
    plt.savefig(f"plots/full/{title.replace(' ', '_').lower()}.png")
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