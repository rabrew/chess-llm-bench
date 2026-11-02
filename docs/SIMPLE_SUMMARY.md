# Chess LLM Benchmark — simple summary
The benchmark contains 526,662 saved evaluations of 22 local language models on 4,000 Lichess puzzle positions.

The audit changed how the results can be interpreted. At the two-pawn direction threshold, always answering “equal” scores 93.275%; every model falls below its matched majority baseline. A high raw agreement rate is therefore insufficient evidence of evaluation skill.

Historical move-quality endpoints were not saved, so the earlier reconstructed quality rankings and bimodal percentages are withdrawn. Historical legality mixes uncertain assistance procedures and cannot establish an all-model single-shot rate. Engine provenance also remains unresolved.

The repaired code records more evidence for future runs, compares fine-tuned models on matched samples, protects saved training work, and makes missing measurements explicit. This study cannot establish whether models reason or memorize positions from these results alone.

See [audited findings](FINDINGS.md) for source tables and [repair status](EXPERIMENT_REPAIRS.md) for remaining evidence gaps.
