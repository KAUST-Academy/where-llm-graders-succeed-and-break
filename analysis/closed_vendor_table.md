# Closed models across vendors under neutral / strict / lenient

MAE [95% CI] bias, against the grader average; CV floor 2.61/35, ML floor 5.13/65. Behaviour per Section 5.1. Gemini rows are the paper's runs (plus F03/IG23); OpenAI and Anthropic rows come from closed_runs/ (Batch API, temperature 0 where the API allows it, reasoning/thinking off).

| vendor | model | persona | CV: MAE [CI] bias | CV n | CV behaviour | ML: MAE [CI] bias | ML n | ML behaviour | run ids | batch cost |
|---|---|---|---|---|---|---|---|---|---|---|
| Anthropic | claude-opus-5 | neutral | 3.54 [3.31, 3.77] -3.24 | 570 | graded | 4.37 [4.13, 4.63] -2.64 | 1038 | graded | N01/IN01 | $159.16 |
| Anthropic | claude-opus-5 | strict | 5.17 [4.90, 5.45] -5.02 | 570 | graded | — | — | — | N02 | $51.29 |
| Google | gemini-3.1-pro-preview | neutral | 1.86 [1.70, 2.02] -0.82 | 570 | graded | 3.40 [3.16, 3.65] +1.18 | 1038 | graded | D02/IG08 | — |
| Google | gemini-3.1-pro-preview | strict | 2.75 [2.51, 2.99] -2.09 | 570 | graded | 3.67 [3.43, 3.94] -1.00 | 1038 | graded | F03/IG23 | — |
| Google | gemini-3.1-pro-preview | lenient | 1.79 [1.62, 1.95] +0.82 | 570 | graded | 4.48 [4.22, 4.75] +3.46 | 1033 | graded | F02/IG14 | — |
| Google | gemini-flash-lite | neutral | 3.34 [3.15, 3.53] -1.78 | 570 | graded | 7.53 [7.20, 7.87] +5.98 | 1013 | graded | A01/IG01 | — |
| Google | gemini-flash-lite | strict | 5.75 [5.46, 6.04] -5.44 | 570 | graded | 9.02 [8.61, 9.43] -7.83 | 1008 | graded | C01/IG05 | — |
| Google | gemini-flash-lite | lenient | 4.49 [4.18, 4.82] +4.01 | 569 | graded | 17.86 [17.30, 18.42] +17.80 | 1020 | collapse | C02/IG06 | — |
| OpenAI | gpt-5.4 | neutral | 4.69 [4.46, 4.92] -4.45 | 570 | graded | 4.30 [4.07, 4.55] -1.95 | 1038 | graded | O11/IO11 | $44.54 |
| OpenAI | gpt-5.4 | strict | 6.90 [6.63, 7.15] -6.81 | 570 | graded | 5.11 [4.85, 5.37] -3.61 | 1038 | graded | O12/IO12 | $44.27 |
| OpenAI | gpt-5.4 | lenient | 2.69 [2.51, 2.89] +1.00 | 570 | graded | 8.30 [7.96, 8.64] +7.94 | 1038 | graded | O13/IO13 | $44.45 |
| OpenAI | gpt-5.5 | neutral | 2.43 [2.28, 2.59] -1.65 | 570 | graded | 3.54 [3.33, 3.77] +1.23 | 1038 | graded | O01/IO01 | $98.05 |
| OpenAI | gpt-5.5 | strict | 4.43 [4.19, 4.68] -4.25 | 570 | graded | 3.70 [3.48, 3.93] -1.38 | 1038 | graded | O02/IO02 | $97.82 |
| OpenAI | gpt-5.5 | lenient | 2.25 [2.08, 2.44] +1.13 | 570 | graded | 6.29 [6.02, 6.57] +5.83 | 1038 | graded | O03/IO03 | $99.15 |

Ratios strict/neutral and lenient/neutral (MAE):

- claude-opus-5: strict/CV ×1.46 (zero-rate 0.7%, r=0.912)
- gemini-3.1-pro-preview: strict/CV ×1.48 (zero-rate 0.4%, r=0.906); strict/ML ×1.08 (zero-rate 1.1%, r=0.928); lenient/CV ×0.96 (zero-rate 0.4%, r=0.938); lenient/ML ×1.32 (zero-rate 0.8%, r=0.935)
- gemini-flash-lite: strict/CV ×1.72 (zero-rate 0.9%, r=0.844); strict/ML ×1.20 (zero-rate 0.2%, r=0.880); lenient/CV ×1.35 (zero-rate 0.2%, r=0.810); lenient/ML ×2.37 (zero-rate 0.0%, r=0.775)
- gpt-5.4: strict/CV ×1.47 (zero-rate 0.4%, r=0.891); strict/ML ×1.19 (zero-rate 0.7%, r=0.926); lenient/CV ×0.57 (zero-rate 0.0%, r=0.918); lenient/ML ×1.93 (zero-rate 0.7%, r=0.906)
- gpt-5.5: strict/CV ×1.82 (zero-rate 0.4%, r=0.912); strict/ML ×1.04 (zero-rate 0.8%, r=0.938); lenient/CV ×0.93 (zero-rate 0.2%, r=0.933); lenient/ML ×1.78 (zero-rate 0.3%, r=0.937)
