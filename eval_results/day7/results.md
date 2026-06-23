# Day 7 Eval Results — 2026-06-01 12:18

**Questions evaluated:** 10  |  **Corrective loop fired:** 2/10 (20.0%)  |  **Citations present:** 6/10

## Aggregate Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| **Exact Match** | 10.0% | Gold string found in prediction |
| **Numeric Match** (±1%) | 33.3% | Numeric answers within 1% of gold (n=6) |
| Self-RAG Confidence | 0.88 | LLM judge [0-1] |
| Hallucination Rate | 0.8333 | NLI-based, see note below |
| ROUGE-L | 0.1924 | Low due to verbose vs. short gold |
| BERTScore F1 | 0.2813 | Semantic similarity to gold |
| RAGAS Faithfulness (n≤10) | — | |
| RAGAS Answer Relevancy (n≤10) | — | |

> **Note on Hallucination Rate:** NLI checks sentence entailment against cited quotes.
> High rate on FinanceBench is expected — numeric reasoning sentences ("average is 10.3%")
> are not syntactically entailed by table excerpts even when the answer is correct.
> Use Exact Match and Numeric Match as the primary correctness signals.

## Per-Question Results

| # | Question (truncated) | Prediction | Gold | EM | NM | Conf | H-Rate | RougeL |
|---|---------------------|------------|------|----|----|------|--------|--------|
| 1 | Taking into account the information outlined … | The 3-year average unadjusted operating … | 10.3% | ✅ | ❌ | 1.0 | 1.0 | 0.2105 |
| 2 | Are there any product categories / service ca… | Yes, for fiscal year 2022, Commercial Ai… | Yes. Boeing has product and service categories that represent more than 20% of Boeing's revenue for FY2022. These categories are Commercial Airplanes which comprises 39% of total revenue, Defence which comprises 35% of total revenue and Services which comprises 26% of total revenue. | ❌ | ❌ | 1.0 | 0.5 | 0.3711 |
| 3 | Which of JPM's business segments had the lowe… | I don't know… | Corporate. Its net revenue was -$473 million. | ❌ | — | 1.0 | 1.0 | 0.0 |
| 4 | What is Coca Cola's FY2022 dividend payout ra… | I don't know… | 0.8 | ❌ | — | 0.0 | 1.0 | 0.0 |
| 5 | What are major acquisitions that Best Buy has… | In fiscal year 2022, Best Buy acquired C… | Best Buy closed two acquisitions, both these companies were already partially owned by Best Buy, but Best Buy acquired all outstanding shares of these two companies during FY 2022: (1) Current Health Ltd and (2) Two Peaks, LLC d/b/a Yardbird Furniture | ❌ | ✅ | 0.9 | 0.5 | 0.4839 |
| 6 | What is the amount of the cash proceeds that … | Johnson & Johnson secured $13.2 billion … | JnJ realised $13.2 billion in cash proceeds from the separation of Kenvue. | ❌ | ✅ | 1.0 | 1.0 | 0.5806 |
| 7 | Using the cash flow statement, answer the fol… | I don't know… | $382.00 | ❌ | — | 1.0 | 1.0 | 0.0 |
| 8 | What drove gross margin change as of the FY20… | I don't know. The provided context for A… | Performance is not measured through gross margin | ❌ | — | 0.9 | 1.0 | 0.0923 |
| 9 | Is growth in JnJ's adjusted EPS expected to a… | Yes, Johnson & Johnson's adjusted EPS gr… | No, rate of growth in adjusted EPS is expected to decelerate slightly from 3.6% in FY2022 to 3.5% in FY2023. | ❌ | ❌ | 1.0 | 0.3333333333333333 | 0.186 |
| 10 | What is Amcor's year end FY2020 net AR (in US… | Amcor's net Accounts Receivable for fisc… | $1616.00 | ❌ | ❌ | 1.0 | 1.0 | 0.0 |