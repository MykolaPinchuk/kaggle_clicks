# Contrasts vs `trailing` (test, mean over folds)

- Sweep: `20251221_183006_paper_full_grid_10pct_s10pct_tecache`
- Baseline within each length set: `trailing`
- Shapes compared: gap1, bucket, calendar, event50
- PR-AUC bootstrap: not computed

| Length set | Shape | ΔROC-AUC | 95% CI |
| --- | --- | ---: | --- |
| A1 | bucket | -0.001643 | [-0.001981, -0.001304] |
| A2 | bucket | -0.000250 | [-0.000593, 0.000094] |
| A3 | bucket | -0.001247 | [-0.001598, -0.000897] |
| A4 | bucket | -0.000159 | [-0.000644, 0.000325] |
| A1 | calendar | 0.000120 | [-0.000147, 0.000386] |
| A2 | calendar | 0.001618 | [0.001237, 0.001999] |
| A3 | calendar | -0.000078 | [-0.000308, 0.000153] |
| A4 | calendar | 0.000840 | [0.000546, 0.001134] |
| A1 | event50 | 0.001178 | [0.000848, 0.001509] |
| A2 | event50 | 0.002204 | [0.001818, 0.002589] |
| A3 | event50 | 0.000402 | [0.000112, 0.000691] |
| A4 | event50 | 0.000833 | [0.000504, 0.001161] |
| A1 | gap1 | -0.002312 | [-0.002665, -0.001959] |
| A2 | gap1 | -0.000467 | [-0.000817, -0.000116] |
| A3 | gap1 | -0.001901 | [-0.002249, -0.001553] |
| A4 | gap1 | -0.000705 | [-0.001120, -0.000290] |
