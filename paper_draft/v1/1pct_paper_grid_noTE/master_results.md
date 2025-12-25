# Master Results: Paper Grid 1% (no TE, Fold A/B)

- Sweep: `20251225_024117_paper_full_grid_1pct_noTE_s1pct`
- Note: time-agg-only ablation (`--no-te`); no statistical inference computed (no prediction exports).

| Run | Description | Test ROC mean+/-std | Test PR mean+/-std |
| --- | --- | --- | --- |
| A4_event50 | A4 event windows last50 | 0.7349+/-0.0054 | 0.3443+/-0.0046 |
| A3_event50 | A3 event windows last50 | 0.7341+/-0.0030 | 0.3433+/-0.0064 |
| A3_trailing | A3 trailing windows (1, 6, 24, 48, 168) | 0.7339+/-0.0053 | 0.3431+/-0.0038 |
| A4_trailing | A4 trailing windows (1, 2, 4, 8, 16, 24, 48, 96, 168) | 0.7336+/-0.0036 | 0.3438+/-0.0077 |
| A4_gap1 | A4 gap g=1 windows (1, 2, 4, 8, 16, 24, 48, 96, 168) | 0.7336+/-0.0046 | 0.3428+/-0.0062 |
| A2_event50 | A2 event windows last50 | 0.7335+/-0.0041 | 0.3431+/-0.0065 |
| A4_calendar | A4 calendar windows on trailing (1, 2, 4, 8, 16, 24, 48, 96, 168) | 0.7334+/-0.0018 | 0.3437+/-0.0085 |
| A3_calendar | A3 calendar windows on trailing (1, 6, 24, 48, 168) | 0.7331+/-0.0045 | 0.3423+/-0.0047 |
| A3_bucket | A3 bucket edges (1, 6, 24, 48, 168) | 0.7327+/-0.0025 | 0.3407+/-0.0074 |
| A1_event50 | A1 event windows last50 | 0.7316+/-0.0063 | 0.3382+/-0.0004 |
| A4_bucket | A4 bucket edges (1, 2, 4, 8, 16, 24, 48, 96, 168) | 0.7309+/-0.0022 | 0.3387+/-0.0076 |
| A3_gap1 | A3 gap g=1 windows (1, 6, 24, 48, 168) | 0.7308+/-0.0033 | 0.3378+/-0.0054 |
| A1_calendar | A1 calendar windows on trailing (1, 6, 24) | 0.7296+/-0.0039 | 0.3399+/-0.0057 |
| A2_calendar | A2 calendar windows on trailing (1, 3, 6, 12, 24) | 0.7287+/-0.0032 | 0.3387+/-0.0072 |
| A1_trailing | A1 trailing windows (1, 6, 24) | 0.7278+/-0.0030 | 0.3376+/-0.0061 |
| A1_bucket | A1 bucket edges (1, 6, 24) | 0.7257+/-0.0010 | 0.3353+/-0.0079 |
| A2_trailing | A2 trailing windows (1, 3, 6, 12, 24) | 0.7257+/-0.0016 | 0.3329+/-0.0053 |
| A1_gap1 | A1 gap g=1 windows (1, 6, 24) | 0.7256+/-0.0026 | 0.3351+/-0.0072 |
| A2_bucket | A2 bucket edges (1, 3, 6, 12, 24) | 0.7252+/-0.0022 | 0.3346+/-0.0061 |
| A2_gap1 | A2 gap g=1 windows (1, 3, 6, 12, 24) | 0.7241+/-0.0015 | 0.3323+/-0.0070 |
