import json
import matplotlib.pyplot as plt
import pandas as pd

with open('EncoderResults.json', 'r') as f:
    results = json.load(f)

prefix_vals = [r["prefix_length"] for r in results]
bert_vals = [r["bert_score"] for r in results]
bart_vals = [r["bart_score"] for r in results]

plt.figure(figsize=(10, 6))
plt.plot(prefix_vals, bert_vals, marker='o', linewidth=2, markersize=8, color='blue')
plt.xlabel('Prefix Length', fontsize=12)
plt.ylabel('BERT Score F1', fontsize=12)
plt.title('BERT Score vs Prefix Length', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('bert_score_plot.png', dpi=300)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(prefix_vals, bart_vals, marker='s', linewidth=2, markersize=8, color='red')
plt.xlabel('Prefix Length', fontsize=12)
plt.ylabel('BART Score', fontsize=12)
plt.title('BART Score vs Prefix Length', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('bart_score_plot.png', dpi=300)
plt.show()

fig, ax1 = plt.subplots(figsize=(12, 6))

ax1.set_xlabel('Prefix Length', fontsize=12)
ax1.set_ylabel('BERT Score F1', fontsize=12, color='blue')
ax1.plot(prefix_vals, bert_vals, marker='o', linewidth=2, markersize=8,
         color='blue', label='BERT Score')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
ax2.set_ylabel('BART Score', fontsize=12, color='red')
ax2.plot(prefix_vals, bart_vals, marker='s', linewidth=2, markersize=8,
         color='red', label='BART Score')
ax2.tick_params(axis='y', labelcolor='red')

plt.title('BERT and BART Scores vs Prefix Length', fontsize=14, fontweight='bold')
fig.tight_layout()
plt.savefig('bert_bart_combined_plot.png', dpi=300)
plt.show()

print("\nSUMMARY TABLE")
print("="*60)
summary_df = pd.DataFrame(results)
print(summary_df.to_string(index=False))
print("="*60)

bert_improvement = ((bert_vals[-1] - bert_vals[0]) / bert_vals[0]) * 100
bart_improvement = ((bart_vals[-1] - bart_vals[0]) / bart_vals[0]) * 100

print(f"\nBERT Score improvement: {bert_improvement:.2f}%")
print(f"BART Score improvement: {bart_improvement:.2f}%")