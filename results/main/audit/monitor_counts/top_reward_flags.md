# Monitor counts among the top-reward attempts

Made by `results/main/audit/monitor_counts/count.py` from `results/main/<arm>_<split>/episodes.jsonl`,
with the monitors of `accessdesk/eval/monitors.py`. A top-reward attempt passed with the full reward
(`monitors.has_full_reward`): the row "passed with the full reward" of `results/main/monitors/<arm>_<split>.md`.
Each number is a count of attempts, not a share.

m1 refused_call, m2 id_not_from_a_result, m3 access_not_checked, m4 repeated_call, m5 words_outside_call.
"m1 to m3": flagged by at least one of m1, m2, m3. "m4 or m5": flagged by m4 or m5.
"any of m1 to m5": flagged by at least one of the five. "any of all seven": also m5b and m6.

| split | arm | attempts | top-reward | m1 | m2 | m3 | m4 | m5 | m1 to m3 | m4 or m5 | any of m1 to m5 | any of all seven |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dev | base | 1536 | 553 | 4 | 4 | 1 | 16 | 31 | 4 | 43 | 45 | 45 |
| dev | sft | 1536 | 1356 | 5 | 4 | 0 | 4 | 59 | 5 | 63 | 67 | 67 |
| dev | grpo_seed1 | 1536 | 1512 | 1 | 1 | 0 | 4 | 10 | 1 | 14 | 15 | 15 |
| dev | grpo_seed2 | 1536 | 1498 | 0 | 0 | 0 | 4 | 14 | 0 | 18 | 18 | 18 |
| dev | grpo_seed1 + grpo_seed2 | 3072 | 3010 | 1 | 1 | 0 | 8 | 24 | 1 | 32 | 33 | 33 |
| test | base | 1536 | 566 | 1 | 0 | 0 | 19 | 37 | 1 | 47 | 48 | 48 |
| test | sft | 1536 | 1387 | 1 | 1 | 0 | 8 | 58 | 1 | 65 | 66 | 66 |
| test | grpo_seed1 | 1536 | 1525 | 0 | 0 | 0 | 3 | 10 | 0 | 13 | 13 | 13 |
| test | grpo_seed2 | 1536 | 1524 | 0 | 0 | 0 | 5 | 10 | 0 | 15 | 15 | 15 |
| test | grpo_seed1 + grpo_seed2 | 3072 | 3049 | 0 | 0 | 0 | 8 | 20 | 0 | 28 | 28 | 28 |
