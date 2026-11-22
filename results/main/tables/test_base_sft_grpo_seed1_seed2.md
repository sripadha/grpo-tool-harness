Families: 48. Scenarios: 192. Episodes per arm: 1536.
Intervals: 95%, by resampling families, 10000 draws, seed 0. Paired bootstrap: the main version of paired_family_bootstrap.

| Arm | Episodes | Harness errors | Pass rate | Interval | grant | approval | already | inactive | Seen styles | Held-out style | Groups with equal rewards | Groups with equal pass or fail | Malformed messages (% of assistant messages) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 1536 | 0 | 41.3 | 32.2 to 50.1 | 49.2 | 74.7 | 24.7 | 16.4 | 42.6 | 37.2 | 26.6 | 29.7 | 4.1 |
| sft | 1536 | 0 | 91.3 | 85.9 to 95.8 | 95.6 | 98.7 | 98.4 | 72.7 | 91.1 | 91.9 | 66.1 | 71.4 | 1.1 |
| grpo_seed1 | 1536 | 0 | 99.5 | 99.0 to 99.9 | 98.4 | 100.0 | 99.7 | 100.0 | 99.6 | 99.5 | 95.8 | 97.4 | 0.1 |
| grpo_seed2 | 1536 | 0 | 99.5 | 99.0 to 99.9 | 98.7 | 100.0 | 99.2 | 100.0 | 99.4 | 99.7 | 93.8 | 95.8 | 0.1 |

| Comparison | Families | Difference in points | Paired bootstrap by family | t-interval |
|---|---|---|---|---|
| sft minus base | 48 | +50.1 | +42.2 to +58.1 | +41.8 to +58.3 |
| grpo_seed1 minus sft | 48 | +8.2 | +3.9 to +13.6 | +3.1 to +13.3 |
| grpo_seed2 minus sft | 48 | +8.1 | +3.7 to +13.5 | +3.0 to +13.3 |

| Arm | Split | Rule set | Tools | Prompt | Text | Config (start of its SHA-256) |
|---|---|---|---|---|---|---|
| base | test | default | five | final | 6e05616ecf4e | 88b3bf6642df |
| sft | test | default | five | final | 6e05616ecf4e | 88b3bf6642df |
| grpo_seed1 | test | default | five | final | 6e05616ecf4e | 88b3bf6642df |
| grpo_seed2 | test | default | five | final | 6e05616ecf4e | 88b3bf6642df |

| Arm | Pass@8 | Pass^1 | Pass^2 | Pass^4 | Pass^8 | All wordings pass | Variance inside families due to wording (if sampling only) | Mean turns | Ended at the turn limit | closed_correctly | closed_wrongly | never_closed | malformed_only | Smallest detectable effect, points |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 78.6 | 41.3 | 26.4 | 15.4 | 8.3 | 15.8 | 9.7 (9.7) | 4.46 | 11.1 | 41.3 | 47.7 | 11.1 | 0.0 | 3.9 (200 splits; one split alone 2.9 to 5.3) |
| sft | 99.0 | 91.3 | 86.6 | 80.0 | 70.3 | 79.5 | 14.5 (9.7) | 3.59 | 0.7 | 91.3 | 7.9 | 0.7 | 0.0 | 2.2 (200 splits; one split alone 1.4 to 3.3) |
| grpo_seed1 | 100.0 | 99.5 | 99.1 | 98.4 | 97.4 | 98.3 | 12.2 (9.7) | 3.51 | 0.1 | 99.5 | 0.3 | 0.1 | 0.0 | 0.6 (200 splits; one split alone 0.4 to 1.1) |
| grpo_seed2 | 100.0 | 99.5 | 99.0 | 97.9 | 95.8 | 98.0 | 7.4 (9.7) | 3.51 | 0.1 | 99.5 | 0.5 | 0.1 | 0.0 | 0.7 (200 splits; one split alone 0.6 to 1.0) |

Episodes per reward value (all valid episodes):
- base: reward 0: 902, reward 77: 1, reward 79: 2, reward 86: 1, reward 87: 2, reward 88: 3, reward 89: 5, reward 91: 5, reward 92: 2, reward 93: 6, reward 94: 3, reward 95: 26, reward 96: 5, reward 98: 7, reward 100: 566
- sft: reward 0: 133, reward 79: 1, reward 81: 2, reward 88: 1, reward 93: 2, reward 95: 9, reward 98: 1, reward 100: 1387
- grpo_seed1: reward 0: 7, reward 93: 1, reward 95: 2, reward 98: 1, reward 100: 1525
- grpo_seed2: reward 0: 8, reward 94: 1, reward 95: 3, reward 100: 1524

Malformed messages: assistant messages that are not exactly one valid tool call, as a share of all assistant messages. A well-formed call that a tool refused (for example an unknown id) is not counted.
Smallest detectable effect: from the arm's own episodes (sampling noise only); a scenario needs 2 samples or more.
