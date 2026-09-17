# Categorical Features Don’t Lie: Detecting Poisoning in Gradient-Boosted Tree Ensembles

**Valency Oscar Colaco**, **Buse Atli**, and **Simin Nadjm-Tehrani**
Institutionen för Datavetenskap (IDA) @ Linköping University, Sweden

---

## About

This repository contains the implementation and experimental artifacts for the paper:

> **Categorical Features Don’t Lie: Detecting Poisoning in Gradient-Boosted Tree Ensembles**

**Coming soon.**

The source code, datasets/configuration details, experimental scripts, and additional materials will be made available here.


### Additional metrics

If we consider the detection metrics computed in, for example, `cic-iiot-datasense.ipynb`, the snippet below is taken from that notebook (the *AUROC threshold selection for COMBINED margins* cell) and shows the pattern used throughout all experiments:

```python
# snippet from cic-iiot-datasense.ipynb
from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt

# ── AUROC threshold selection for COMBINED margins ──────────────────────────
# 1 = poisoned, 0 = clean
y_true = is_poisoned_y0.astype(int)

# Use combined margins
margins_for_auc = np.asarray(combined_margins_y0)

# Lower margin = more suspicious/poisoned
scores = -margins_for_auc

# ROC curve
fpr, tpr, thresholds = roc_curve(y_true, scores)
auroc = auc(fpr, tpr)

# Select threshold using Youden's J statistic
j_scores = tpr - fpr
best_idx = np.argmax(j_scores)

best_score_threshold = thresholds[best_idx]

# Convert threshold back to original combined-margin scale
combined_threshold = -best_score_threshold

print(f"Combined AUROC: {auroc:.4f}")
print(f"Selected combined-margin threshold: {combined_threshold:.6f}")
print(f"TPR: {tpr[best_idx]:.4f}")
print(f"FPR: {fpr[best_idx]:.4f}")

# Poison prediction rule:
# combined margin <= threshold means predicted poisoned
predicted_poison = margins_for_auc <= combined_threshold
```

The cell produces three objects that every other metric builds on:

| Object | Meaning |
| --- | --- |
| `y_true` | Ground-truth labels, 1 = poisoned, 0 = clean |
| `scores` | Negated combined margins, so that higher = more suspicious |
| `predicted_poison` | Binary predictions at the selected threshold |

Note the sign convention: a **lower** combined margin indicates a more suspicious sample,
which is why `scores = -margins_for_auc`. The same convention applies to every metric
below.

Threshold-dependent metrics (Recall, Precision, F1) use `predicted_poison`; ranking-based
metrics use `scores`. Append the following to the cell above:

```python
from sklearn.metrics import (
    recall_score, precision_score, f1_score, precision_recall_curve
)

print(f"Recall    : {recall_score(y_true, predicted_poison):.4f}")
print(f"Precision : {precision_score(y_true, predicted_poison):.4f}")
print(f"F1        : {f1_score(y_true, predicted_poison):.4f}")

auprc = average_precision_score(y_true, scores)
print(f"Combined AUPRC: {auprc:.4f}")
```
