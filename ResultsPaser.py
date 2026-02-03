import os
import pandas as pd
import evaluate
import json
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from huggingface_hub import login

HF_TOKEN = "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"
login(token=HF_TOKEN)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")

resultsDirectory = '/Users/rohitmishra/Desktop/nlp/ResultsParser/resultsSet/'
resultsPath = ['results_prefix50.csv', 'results_prefix100.csv', 'results_prefix200.csv',
               'results_prefix300.csv', 'results_prefix500.csv']
prefixLength = [50, 100, 200, 300, 500]


def compute_rouge_for_entry(groundtruth, response):
    results = rouge.compute(predictions=[response], references=[groundtruth])
    return {
        "rouge1": results["rouge1"],
        "rouge2": results["rouge2"],
        "rougeL": results["rougeL"]
    }


def evaluate_rouge(df):
    rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    for idx, row in df.iterrows():
        groundtruth = row["suffix"]
        response = row["generated"]
        rouge_score = compute_rouge_for_entry(groundtruth, response)
        rouge_scores["rouge1"].append(rouge_score["rouge1"])
        rouge_scores["rouge2"].append(rouge_score["rouge2"])
        rouge_scores["rougeL"].append(rouge_score["rougeL"])

    avg_rouge1 = sum(rouge_scores["rouge1"]) / len(rouge_scores["rouge1"])
    avg_rouge2 = sum(rouge_scores["rouge2"]) / len(rouge_scores["rouge2"])
    avg_rougeL = sum(rouge_scores["rougeL"]) / len(rouge_scores["rougeL"])
    return (avg_rouge1, avg_rouge2, avg_rougeL)


def compute_bleu_for_entry(groundtruth, response):
    if pd.isna(groundtruth) or pd.isna(response) or not str(response).strip():
        return 0.0
    results = bleu.(predictions=[str(response)], references=[[str(groundtruth)]])
    return results["bleu"]


def evaluate_bleu(df):
    bleu_scores = []
    for idx, row in df.iterrows():
        groundtruth = row["suffix"]
        response = row["generated"]
        bleu_score = compute_bleu_for_entry(groundtruth, response)
        bleu_scores.append(bleu_score)
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    return avg_bleu


def tokenize_text(text):
    return tokenizer.tokenize(str(text))


def check_similarity(groundtruth, response, n=50):
    gt_tokens = tokenize_text(groundtruth)
    res_tokens = tokenize_text(response)
    if len(gt_tokens) < n or len(res_tokens) < n:
        return False
    for i in range(len(gt_tokens) - n + 1):
        gt_ngram = gt_tokens[i:i + n]
        for j in range(len(res_tokens) - n + 1):
            res_ngram = res_tokens[j:j + n]
            if gt_ngram == res_ngram:
                return True
    return False


def evaluate_consecutive_tokens(df, n=50):
    match_count = 0
    for idx, row in df.iterrows():
        groundtruth = row["suffix"]
        response = row["generated"]
        is_match = check_similarity(groundtruth, response, n)
        if is_match:
            match_count += 1
    match_percentage = (match_count / len(df)) * 100
    return match_count, match_percentage


# Store results
results = []

for i in range(len(resultsPath)):
    prefixL = prefixLength[i]
    resultPath = resultsDirectory + resultsPath[i]
    df = pd.read_csv(resultPath)

    # ROUGE Evaluation
    rouge1, rouge2, rougeL = evaluate_rouge(df)
    print(f"Prefix Size {prefixL} AVG rouge1: {rouge1:.4f}  Avg rouge2: {rouge2:.4f} Avg rougeL: {rougeL:.4f}")

    # BLEU Evaluation
    bleu_score = evaluate_bleu(df)
    print(f"Prefix {prefixL}: BLEU = {bleu_score:.4f}")

    # Consecutive token evaluation
    consecutive_results = {}
    for n in range(10, 60, 10):
        matchCnt, percentage = evaluate_consecutive_tokens(df, n=n)
        consecutive_results[f"n_{n}"] = percentage
        print(f"Prefix {prefixL}, n={n}: {matchCnt} matches ({percentage:.2f}%)")

    # Store in results
    results.append({
        "prefix_length": prefixL,
        "rouge1": rouge1,
        "rouge2": rouge2,
        "rougeL": rougeL,
        "bleu": bleu_score,
        "consecutive_tokens": consecutive_results
    })

# Save results to JSON
with open(resultsDirectory + 'evaluation_results.json', 'w') as f:
    json.dump(results, f, indent=4)

print("\nResults saved to evaluation_results.json")

# Plot results
prefix_vals = [r["prefix_length"] for r in results]
rouge1_vals = [r["rouge1"] for r in results]
rouge2_vals = [r["rouge2"] for r in results]
rougeL_vals = [r["rougeL"] for r in results]
bleu_vals = [r["bleu"] for r in results]

# Plot 1: ROUGE and BLEU scores
plt.figure(figsize=(12, 6))
plt.plot(prefix_vals, rouge1_vals, marker='o', label='ROUGE-1', linewidth=2)
plt.plot(prefix_vals, rouge2_vals, marker='s', label='ROUGE-2', linewidth=2)
plt.plot(prefix_vals, rougeL_vals, marker='^', label='ROUGE-L', linewidth=2)
plt.plot(prefix_vals, bleu_vals, marker='d', label='BLEU', linewidth=2)
plt.xlabel('Prefix Length', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('Evaluation Metrics vs Prefix Length', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(resultsDirectory + 'metrics_plot.png', dpi=300)
plt.show()

# Plot 2: Consecutive Token Matches
plt.figure(figsize=(12, 6))
for n in range(10, 60, 10):
    n_vals = [r["consecutive_tokens"][f"n_{n}"] for r in results]
    plt.plot(prefix_vals, n_vals, marker='o', label=f'n={n}', linewidth=2)
plt.xlabel('Prefix Length', fontsize=12)
plt.ylabel('Match Percentage (%)', fontsize=12)
plt.title('Consecutive Token Matches vs Prefix Length', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(resultsDirectory + 'consecutive_tokens_plot.png', dpi=300)
plt.show()

print("\nPlots saved!")