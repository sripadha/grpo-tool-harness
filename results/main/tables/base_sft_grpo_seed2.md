Families: 48. Scenarios: 192. Episodes per arm: 1536.
Intervals: 95%, by resampling families, 10000 draws, seed 0. Paired bootstrap: the main version of paired_family_bootstrap.

| Arm | Episodes | Harness errors | Pass rate | Interval | grant | approval | already | inactive | Seen styles | Held-out style | Groups with equal rewards | Groups with equal pass or fail | Malformed messages (% of assistant messages) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 1536 | 0 | 39.8 | 31.4 to 48.4 | 47.7 | 69.3 | 21.4 | 20.8 | 41.8 | 33.9 | 26.6 | 31.2 | 4.6 |
| sft | 1536 | 0 | 89.4 | 82.1 to 95.4 | 97.1 | 97.7 | 98.4 | 64.3 | 90.2 | 87.0 | 71.9 | 78.1 | 0.7 |
| grpo_seed2 | 1536 | 0 | 97.7 | 93.8 to 99.9 | 92.4 | 99.2 | 99.5 | 99.7 | 97.9 | 97.1 | 95.3 | 96.9 | 0.2 |

| Comparison | Families | Difference in points | Paired bootstrap by family | t-interval |
|---|---|---|---|---|
| sft minus base | 48 | +49.6 | +41.1 to +57.9 | +40.9 to +58.3 |
| grpo_seed2 minus sft | 48 | +8.3 | +1.5 to +16.1 | +0.6 to +16.0 |
| grpo_seed2 minus base | 48 | +57.9 | +48.5 to +67.1 | +48.4 to +67.5 |

| Arm | Split | Rule set | Tools | Prompt | Text | Config (start of its SHA-256) |
|---|---|---|---|---|---|---|
| base | dev | default | five | final | 6e05616ecf4e | 88b3bf6642df |
| sft | dev | default | five | final | 6e05616ecf4e | 88b3bf6642df |
| grpo_seed2 | dev | default | five | final | 6e05616ecf4e | 88b3bf6642df |

| Arm | Pass@8 | Pass^1 | Pass^2 | Pass^4 | Pass^8 | All wordings pass | Variance inside families due to wording (if sampling only) | Mean turns | Ended at the turn limit | closed_correctly | closed_wrongly | never_closed | malformed_only | Smallest detectable effect, points |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 78.6 | 39.8 | 25.4 | 15.6 | 9.9 | 13.4 | 14.7 (9.7) | 4.45 | 10.7 | 39.8 | 49.5 | 10.7 | 0.0 | 3.9 (200 splits; one split alone 2.4 to 5.0) |
| sft | 97.9 | 89.4 | 85.8 | 81.7 | 76.0 | 80.3 | 15.8 (9.7) | 3.59 | 0.5 | 89.4 | 10.2 | 0.5 | 0.0 | 1.9 (200 splits; one split alone 1.1 to 3.1) |
| grpo_seed2 | 98.4 | 97.7 | 97.2 | 96.5 | 95.3 | 96.4 | 18.2 (9.7) | 3.51 | 0.3 | 97.7 | 2.0 | 0.3 | 0.0 | 0.7 (200 splits; one split alone 0.4 to 1.1) |

Episodes per reward value (all valid episodes):
- base: reward 0: 925, reward 79: 1, reward 81: 1, reward 82: 1, reward 84: 3, reward 87: 2, reward 88: 3, reward 91: 4, reward 92: 1, reward 93: 2, reward 94: 8, reward 95: 21, reward 96: 8, reward 98: 3, reward 100: 553
- sft: reward 0: 163, reward 81: 1, reward 88: 1, reward 91: 3, reward 93: 1, reward 94: 1, reward 95: 8, reward 96: 1, reward 98: 1, reward 100: 1356
- grpo_seed2: reward 0: 35, reward 92: 1, reward 95: 2, reward 100: 1498

Malformed messages: assistant messages that are not exactly one valid tool call, as a share of all assistant messages. A well-formed call that a tool refused (for example an unknown id) is not counted.
Smallest detectable effect: from the arm's own episodes (sampling noise only); a scenario needs 2 samples or more.
