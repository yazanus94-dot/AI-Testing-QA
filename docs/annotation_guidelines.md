# Annotation Guidelines (v1.0)

## Part A: Sentiment labeling (`annotations.jsonl`)

Label the **customer's sentiment toward the product** in the text.

| Label | Use when | Example |
|---|---|---|
| `positive` | The writer expresses satisfaction, praise, or a recommendation | "Great value, the laptop is excellent." |
| `negative` | The writer expresses dissatisfaction, a complaint, or a warning | "The phone broke after two days. Terrible." |
| `neutral` | Facts, logistics, or no clear opinion | "The router arrived on Tuesday." |

### Edge cases
| Situation | Decision |
|---|---|
| Mixed opinion ("Camera is great but the battery is awful") | Label the **overall** sentiment; if truly balanced, escalate as `neutral` + flag for adjudication |
| Sarcasm ("Oh great, it broke again") | Label the **intended** sentiment (`negative`) |
| Uncertainty ("Not sure yet about the keyboard") | `neutral` |
| Empty / unreadable text | Do **not** label. Flag the item as a data defect |
| Text in another language you cannot read | Do **not** guess. Skip and flag |

### Rules
1. Use **only** the three labels above (exact, lowercase). Free text is not allowed.
2. Label independently. Do not discuss items with other annotators before submitting.
3. Gold (hidden test) items are mixed into your queue. Accuracy below 85% triggers retraining.
4. If an item has no majority after three labels, it goes to an expert adjudicator.

## Part B: Preference ranking for RLHF (`preferences.jsonl`)

You will see a prompt and two responses (A and B). Choose the better response.

**Order of precedence:** 1) Safety, 2) Factual accuracy, 3) Helpfulness / completeness, 4) Clarity and concision.

| Choice | Use when |
|---|---|
| `A` / `B` | One response is clearly better under the precedence above |
| `tie` | Both are equally good or equally bad. Use sparingly |

### Common biases to avoid
| Bias | What it looks like | What to do |
|---|---|---|
| **Position bias** | Choosing A (or B) because of where it appears | Read both fully before deciding |
| **Verbosity bias** | Choosing the longer answer because it feels thorough | Longer is **not** better. Reward specific, correct content |
| **Confidence bias** | Choosing the answer that sounds more certain | Verify claims; confident but wrong loses to hedged but right |

Calibration pairs (where the shorter or lower-ranked-by-position answer is correct) are inserted regularly to measure these biases.

## Part C: QA metrics tracked per annotator
- Accuracy vs gold items (target >= 85%)
- Mean pairwise Cohen's kappa vs other annotators (target >= 0.60 for labels, >= 0.40 for preferences)
- Position bias (% choosing A; target 35% to 65%)
- Verbosity bias (% choosing the longer response; target <= 70%)
