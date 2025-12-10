import os
import json
import re
from collections import Counter
from tqdm import tqdm
from pathlib import Path
import subprocess
import textwrap
from pyserini.search.lucene import LuceneSearcher
from pyserini.search.lucene import LuceneImpactSearcher
from pyserini.search.lucene import LuceneHnswDenseSearcher
from pyserini.index.lucene import LuceneIndexReader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

# Base directory = a folder under the user's home directory 
base_dir = Path.home() 
base_dir.mkdir(parents=True, exist_ok=True)

data_dir = base_dir / "data"
data_dir.mkdir(parents=True, exist_ok=True)

csv_path = data_dir / "complaints_data.csv"
tsv_path = data_dir / "search_corpus.tsv"
output_dir = data_dir / "preprocessed_corpus"
index_dir = data_dir / "complaints_index"
input_dir = output_dir  # for Pyserini indexer

# Load & normalize data
def load_and_clean(path: Path):
    
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find complaints_data.csv at:\n  {path}\n"
            f"Please place the file there or update csv_path in the script."
        )

    data = pd.read_csv(path)
    # convert date received to proper datetime format
    data['Date received'] = data['Date received'].astype(str).str.strip()
    data['Date received'] = pd.to_datetime(data['Date received'], errors='coerce')
    data = data.dropna(subset=['Consumer complaint narrative'])
    return data


# normalize dataset and mark censored data with [REDACTED]
def normalize_text(df_in: pd.DataFrame) -> pd.DataFrame:
    df_in['Consumer complaint narrative'] = df_in['Consumer complaint narrative'].str.replace(
        r"\bX{2,10}\b", "[REDACTED]", regex=True
    )
    df_in['Consumer complaint narrative'] = df_in['Consumer complaint narrative'].str.lower()
    return df_in

# Preprocess corpus into json files
def preprocess_corpus(input_file: Path, output_dir: Path):
    os.makedirs(output_dir, exist_ok=True)
    with input_file.open('r') as f:
        for i, line in enumerate(tqdm(f, desc="Preprocessing corpus")):
            try:
                doc_id, contents = line.rstrip("\n").split("\t", 1)
            except ValueError:
                continue  # skip malformed lines

            doc = {"id": doc_id, "contents": contents}
            with (output_dir / f"doc{i}.json").open('w') as out:
                json.dump(doc, out)


# Build index using pyserini
def build_index(input_dir: Path, index_dir: Path):
    if index_dir.exists() and any(index_dir.iterdir()):
        print(f"Index already exists at {index_dir}. Skipping index building.")
        return

    cmd = [
        "python", "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        "--input", str(input_dir),
        "--index", str(index_dir),
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", "1",
        "--storePositions", "--storeDocvectors", "--storeRaw"
    ]
    print("Building index with command:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

# Load data & build search corpus 
df = load_and_clean(csv_path)
df = normalize_text(df)
df = df.reset_index(inplace=False)
df["id"] = df.index.astype(str)

# Save search corpus TSV (for indexing)
search_df = df[["id", "Consumer complaint narrative"]]
search_df.to_csv(
    path_or_buf=str(tsv_path),
    index=False,
    header=False,
    sep='\t'
)

input_file = tsv_path

if not output_dir.exists() or not any(output_dir.iterdir()):
    preprocess_corpus(input_file, output_dir)
else:
    print(f"Preprocessed corpus already exists at {output_dir}. Skipping preprocessing.")

build_index(input_dir, index_dir)

# Build global token DF stats - count number of documents each token appears
def build_global_token_df(df_in: pd.DataFrame):
    df_counts = Counter()
    for text in df_in["Consumer complaint narrative"]:
        text = str(text).lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        tokens = set(text.split())   
        df_counts.update(tokens)
    return df_counts


GLOBAL_DF = build_global_token_df(df)
TOTAL_DOCS = len(df)

# Token / phrase filter lists
AUX_OR_FUNCTION_WORDS = {
    "am", "is", "are", "was", "were",
    "be", "been", "being",
    "have", "has", "had",
    "do", "does", "did",
    "can", "could", "will", "would",
    "shall", "should", "may", "might", "must", "when", "any",
    "use", "went", "what", "but",
    "why", "years", "ago", "told",
    "writing", "whom", "attached", "listed",
    "matter"
}

DETERMINER_WORDS = {"this", "that", "these", "those"}

PRONOUN_OR_POLITE_WORDS = {
    "you", "your", "yours",
    "thank", "thanks", "unfair"
}

BANK_TOKENS = {
    "navy", "federal", "union",
    "american", "express",
    "boa", "chase", "wells", "fargo",
    "citi", "citibank", "discover",
    "capital", "one", "bank", "america",
    "cfpb"
}

GENERIC_NOUNS = {
    "complaint", "complaints",
    "customer", "service",
    "investigation", "file", "files",
    "funds", "practice", "practices",
    "multiple", "business", "days", "reason", "action",
    "report", "reporting"
}

PREPOSITION_WORDS = {
    "about", "against", "into", "under", "over",
    "regarding", "concerning", "within", "without",
    "despite", "where"
}

BORING_START_WORDS = {
    "despite", "were", "having", "similar",
    "regarding", "where", "believe", "even", "though",
    "more", "than", "would", "like"
}

LEGAL_POLICY_WORDS = {
    "act", "consumer", "protection", "regulation",
    "equal", "opportunity", "faith", "securities",
    "fair", "truth"
}


# Pyserini searchers 
rm3_searcher = LuceneSearcher(str(index_dir))
rm3_searcher.set_bm25(k1=1.2, b=0.75)
rm3_searcher.set_rm3(
    fb_terms=10,            # number of feedback terms
    fb_docs=10,             # number of top docs used as pseudo-relevant
    original_query_weight=0.5
)

# RM3 Retrieval
def search_rm3(df_in, query: str, top_k: int = 500):
    hits = rm3_searcher.search(query, k=top_k)

    hit_ids = [h.docid for h in hits]
    hit_scores = [h.score for h in hits]

    subset = df_in[df_in["id"].isin(hit_ids)].copy()

    score_map = dict(zip(hit_ids, hit_scores))
    subset["rm3_score"] = subset["id"].map(score_map)

    subset = subset.sort_values("rm3_score", ascending=False)
    return subset

# Tokenization / n-gram utilities
def tokenize(text: str):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()

    cleaned = []
    for t in tokens:
        if t.isdigit():
            continue
        if len(t) <= 2:
            continue
        if t in DETERMINER_WORDS:
            continue
        if t in PRONOUN_OR_POLITE_WORDS:
            continue
        if t in BANK_TOKENS:
            continue
        if t in GENERIC_NOUNS:
            continue
        if t in PREPOSITION_WORDS:
            continue
        if t in BORING_START_WORDS:
            continue
        if t in LEGAL_POLICY_WORDS:
            continue
        if t in AUX_OR_FUNCTION_WORDS:
            continue

        dfreq = GLOBAL_DF.get(t, 0)
        if dfreq >= 0.40 * TOTAL_DOCS:
            continue
        if dfreq < 3:
            continue

        cleaned.append(t)

    return cleaned


def generate_ngrams(tokens, n):
    return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

# True if the phrase is basically just the query words, with at most 1 extra word.
def phrase_is_query_heavy(phrase: str, query_terms: set, max_non_query_words: int = 1):
    tokens = phrase.split()
    non_query_count = sum(1 for t in tokens if t not in query_terms)
    return non_query_count <= max_non_query_words

# Drops phrases that are substrings of already-kept phrases (e.g., drop “personal” if “personal information” already kept).
def deduplicate_phrases(phrases_with_counts, max_phrases: int = 20):
    sorted_phrases = sorted(
        phrases_with_counts,
        key=lambda x: (-x[1], -len(x[0]))
    )

    kept = []
    for phrase, count in sorted_phrases:
        if any(phrase in kp for kp, _ in kept):
            continue
        kept.append((phrase, count))
        if len(kept) >= max_phrases:
            break

    return kept

# Part 1: query-based word association
def get_top_phrases_from_results(
    results_df: pd.DataFrame,
    query: str,
    top_k_phrases: int = 20,
    min_ngram: int = 2,
    max_ngram: int = 3,
):
    query_terms = set(query.lower().split())
    phrase_counts = Counter()

    for text in results_df["Consumer complaint narrative"]:
        tokens = tokenize(text)

        doc_phrases = set()
        for n in range(min_ngram, max_ngram + 1):
            for p in generate_ngrams(tokens, n):
                if phrase_is_query_heavy(p, query_terms):
                    continue
                tokens_p = p.split()
                if all(tok in AUX_OR_FUNCTION_WORDS for tok in tokens_p):
                    continue
                doc_phrases.add(p)

        phrase_counts.update(doc_phrases)

    raw_top = phrase_counts.most_common(100)
    clean_top = deduplicate_phrases(raw_top, max_phrases=top_k_phrases)
    return clean_top

# Part 2: topic-like phrases & monthly trends
def is_boring_phrase(phrase: str) -> bool:
    toks = phrase.split()
    if not toks:
        return True

    if toks[0] in BORING_START_WORDS:
        return True

    strong_tokens = [
        t for t in toks
        if t not in AUX_OR_FUNCTION_WORDS
        and t not in PRONOUN_OR_POLITE_WORDS
        and t not in BANK_TOKENS
        and t not in GENERIC_NOUNS
        and t not in PREPOSITION_WORDS
        and t not in LEGAL_POLICY_WORDS
        and t not in BORING_START_WORDS
        and t not in DETERMINER_WORDS
    ]

    return not bool(strong_tokens)


def collect_global_topic_phrases(
    df_in: pd.DataFrame,
    start_date: str = None,
    end_date: str = None,
    company: str = None,
    min_ngram: int = 2,
    max_ngram: int = 3,
    top_k_global: int = 30,
):
    date_col = pd.to_datetime(df_in["Date received"])
    mask = pd.Series(True, index=df_in.index)

    if start_date is not None:
        mask &= date_col >= pd.to_datetime(start_date)
    if end_date is not None:
        mask &= date_col <= pd.to_datetime(end_date)

    if company is not None and "Company" in df_in.columns:
        mask &= df_in["Company"].str.contains(company, case=False, na=False)

    sub_df = df_in.loc[mask].dropna(subset=["Consumer complaint narrative"])
    print(f"[collect_global_topic_phrases] Using {len(sub_df)} complaints")

    phrase_counts = Counter()
    for text in sub_df["Consumer complaint narrative"]:
        tokens = tokenize(text)
        doc_phrases = set()
        for n in range(min_ngram, max_ngram + 1):
            for p in generate_ngrams(tokens, n):
                if is_boring_phrase(p):
                    continue
                doc_phrases.add(p)
        phrase_counts.update(doc_phrases)

    raw_top = phrase_counts.most_common(200)
    deduped = deduplicate_phrases(raw_top, max_phrases=top_k_global)

    return deduped


def monthly_phrase_trends(
    df_in: pd.DataFrame,
    topic_phrases,
    start_date: str = None,
    end_date: str = None,
    company: str = None,
    min_ngram: int = 2,
    max_ngram: int = 3,
):
    phrase_set = set(topic_phrases)

    date_col = pd.to_datetime(df_in["Date received"])
    mask = pd.Series(True, index=df_in.index)

    if start_date is not None:
        mask &= date_col >= pd.to_datetime(start_date)
    if end_date is not None:
        mask &= date_col <= pd.to_datetime(end_date)

    if company is not None and "Company" in df_in.columns:
        mask &= df_in["Company"].str.contains(company, case=False, na=False)

    sub_df = df_in.loc[mask].dropna(subset=["Consumer complaint narrative"]).copy()
    sub_df["month"] = pd.to_datetime(sub_df["Date received"]).dt.to_period("M")

    print(f"[monthly_phrase_trends] Using {len(sub_df)} complaints after filters")

    counts = Counter()
    for _, row in sub_df.iterrows():
        month = row["month"]
        tokens = tokenize(row["Consumer complaint narrative"])
        doc_phrases = set()
        for n in range(min_ngram, max_ngram + 1):
            for p in generate_ngrams(tokens, n):
                if p in phrase_set and not is_boring_phrase(p):
                    doc_phrases.add(p)

        for p in doc_phrases:
            counts[(month, p)] += 1

    if not counts:
        print("No phrase counts found for these filters.")
        return pd.DataFrame()

    months = sorted({m for (m, _) in counts.keys()})
    months_str = [str(m) for m in months]

    data = {p: [0] * len(months_str) for p in topic_phrases}
    month_index = {m: i for i, m in enumerate(months)}

    for (m, p), c in counts.items():
        i = month_index[m]
        data[p][i] = c

    trends_df = pd.DataFrame(data, index=months_str)
    trends_df.index.name = "month"

    return trends_df


def plot_top_topic_trends(trends_df: pd.DataFrame, top_n: int = 5, title: str = None):
    if trends_df.empty:
        print("Trend table is empty; nothing to plot.")
        return

    totals = trends_df.sum(axis=0).sort_values(ascending=False)
    top_phrases = list(totals.head(top_n).index)

    subset = trends_df[top_phrases]

    subset.plot(marker="o")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Number of complaints")
    plt.xlabel("Month")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()


# Interactive CLI
RESET   = "\033[0m"
BOLD    = "\033[1m"
CYAN    = "\033[36m"
YELLOW  = "\033[33m"
GREEN   = "\033[32m"
MAGENTA = "\033[35m"
RED     = "\033[31m"


def print_banner(title: str, color=CYAN):
    line = "=" * 70
    print(f"\n{color}{line}{RESET}")
    print(f"{color}{BOLD}{title}{RESET}")
    print(f"{color}{line}{RESET}")


def print_subsection(title: str, color=YELLOW):
    line = "-" * 70
    print(f"\n{color}{line}{RESET}")
    print(f"{color}{title}{RESET}")
    print(f"{color}{line}{RESET}")


def print_top_retrieved_complaints(results_df: pd.DataFrame, k: int = 10):
    if results_df.empty:
        print_subsection("NO RETRIEVED COMPLAINTS TO DISPLAY", color=RED)
        print(f"{RED}Result set is empty, nothing to show.{RESET}")
        return

    k = min(k, len(results_df))

    print_subsection(f"PART 1 – TOP {k} RETRIEVED COMPLAINTS", color=MAGENTA)

    for rank, (_, row) in enumerate(results_df.head(k).iterrows(), start=1):
        issue   = row.get("Issue", "N/A")
        company = row.get("Company", "N/A")
        date    = row.get("Date received", "N/A")
        score   = row.get("rm3_score", None)

        raw_text = str(row.get("Consumer complaint narrative", "")).replace("\r", " ").replace("\n", " ")
        narrative_wrapped = textwrap.fill(raw_text.strip(), width=120)

        print(f"{BOLD}{CYAN}Rank {rank}{RESET}")
        if score is not None:
            print(f"  {YELLOW}RM3 score:{RESET} {score:.4f}")
        print(f"  {YELLOW}Date:      {RESET}{date}")
        print(f"  {YELLOW}Company:   {RESET}{company}")
        print(f"  {YELLOW}Issue:     {RESET}{issue}")
        print(f"  {YELLOW}Narrative:{RESET}")
        print(f"    {narrative_wrapped}")
        print()

# CLI workflows (Part 1 & Part 2)
def run_phrase_association_cli():
    print_banner("PART 1 – QUERY-BASED WORD ASSOCIATION DISCOVERY", color=CYAN)

    query = input("Enter search query (e.g., 'identity theft'): ").strip()
    if not query:
        print_subsection("NO QUERY ENTERED", color=RED)
        print(f"{RED}No query entered; skipping Part 1.{RESET}")
        return

    year_str = input(
        "Enter year (YYYY) to filter results, or press Enter for all years: "
    ).strip()
    year = None
    if year_str:
        try:
            year = int(year_str)
        except ValueError:
            print(f"{YELLOW}Invalid year format; ignoring year filter.{RESET}")
            year = None

    top_k = 500
    results = search_rm3(df, query, top_k=top_k)

    if year is not None:
        mask = pd.to_datetime(results["Date received"]).dt.year == year
        results = results.loc[mask]

    if results.empty:
        print_subsection("NO RESULTS", color=RED)
        if year is None:
            print(f"{RED}No complaints found for this query.{RESET}")
        else:
            print(f"{RED}No complaints found for query '{query}' in year {year}.{RESET}")
        return

    top_phrases = get_top_phrases_from_results(results, query, top_k_phrases=20)

    print_subsection("PART 1 – FILTER SUMMARY", color=YELLOW)
    print(f"{YELLOW}Query:        {RESET}{query}")
    print(f"{YELLOW}Year filter:  {RESET}{year if year is not None else 'All years'}")
    print(f"{YELLOW}Number of complaints: {RESET}{len(results)} retrieved")

    print_subsection("PART 1 – TOP ASSOCIATED PHRASES", color=GREEN)
    if not top_phrases:
        print(f"{YELLOW}No informative phrases found for this query / year filter.{RESET}")
    else:
        for rank, (phrase, count) in enumerate(top_phrases, start=1):
            print(f"{GREEN}{rank:2d}. {phrase}  (appears in {count} complaints){RESET}")

    print_top_retrieved_complaints(results, k=10)
    print()


def run_monthly_topic_trends_cli():
    print_banner("PART 2 – COMPLAINTS TOPIC TRENDS", color=MAGENTA)

    start_date = input("Start date (YYYY-MM-DD), or press Enter for none: ").strip()
    if not start_date:
        start_date = None

    end_date = input("End date (YYYY-MM-DD), or press Enter for none: ").strip()
    if not end_date:
        end_date = None

    company = input(
        "Company filter (e.g., 'NAVY FEDERAL CREDIT UNION'), or press Enter for all: "
    ).strip()
    if not company:
        company = None

    top_n_plot_str = input(
        "How many phrases to plot in the trend chart? (default 3): "
    ).strip()
    top_n_plot = int(top_n_plot_str) if top_n_plot_str else 3

    top_k_global = 20

    deduped_topics = collect_global_topic_phrases(
        df,
        start_date=start_date,
        end_date=end_date,
        company=company,
        min_ngram=2,
        max_ngram=3,
        top_k_global=top_k_global,
    )

    if not deduped_topics:
        print_subsection("NO TOPIC PHRASES FOUND", color=RED)
        print(f"{RED}No topic phrases found for these filters.{RESET}")
        return

    global_topics = [p for p, c in deduped_topics]

    print_subsection("PART 2 – FILTER SUMMARY", color=YELLOW)
    print(f"{YELLOW}Date range:   {RESET}{start_date or 'None'}  ->  {end_date or 'None'}")
    print(f"{YELLOW}Company:      {RESET}{company or 'All companies'}")
    print(f"{YELLOW}Number of topic phrases tracked: {RESET}{len(global_topics)}")

    print_subsection("PART 2 – SELECTED TOPIC PHRASES", color=GREEN)
    for rank, (phrase, count) in enumerate(deduped_topics, start=1):
        print(f"{GREEN}{rank:2d}. {phrase}  (appears in {count} complaints){RESET}")

    trends_df = monthly_phrase_trends(
        df,
        topic_phrases=global_topics,
        start_date=start_date,
        end_date=end_date,
        company=company,
        min_ngram=2,
        max_ngram=3,
    )

    if trends_df.empty:
        print_subsection("NO MONTHLY COUNTS", color=RED)
        print(f"{RED}No phrase counts found for these filters.{RESET}")
        return

    totals = trends_df.sum(axis=0).sort_values(ascending=False)
    top_plot_phrases = list(totals.head(top_n_plot).index)

    print_subsection("PART 2 – MONTHLY TREND TABLE (TOP PHRASES)", color=MAGENTA)
    print(trends_df[top_plot_phrases])

    title = "Top complaint phrase trends by month"
    if company:
        title += f" – {company}"
    plot_top_topic_trends(trends_df, top_n=top_n_plot, title=title)

# Main
if __name__ == "__main__":
    print_banner("FINANCIAL COMPLAINTS EXPLORER", color=CYAN)
    print(f"{CYAN}You will perform a query-based association analysis (Part 1),{RESET}")
    print(f"{CYAN}You will then explore topic trends (Part 2).{RESET}")

    run_phrase_association_cli()
    print("\n" + "=" * 70)
    print("END OF PART 1")
    print("=" * 70 + "\n")

    run_monthly_topic_trends_cli()
    print("\n" + "=" * 70)
    print("END OF PROGRAM")
    print("=" * 70 + "\n")
