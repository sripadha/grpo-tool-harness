# Monitors outside the reward: results/main/grpo_seed2_dev/episodes.jsonl

- Records: split dev; run grpo_seed2_dev, arm grpo_seed2, ruleset default, toolset five, prompt final. Records of other splits left out: 0.
- Families: data/scenarios/dev.jsonl (SHA-256 fea3e07efa22; its SHA-256 is the one that the records name; the verifier gives the saved pass or fail for every valid record).
- Text file: text/model_text.yaml (SHA-256 6e05616ecf4e, the one that the records name).

Episodes: 1536 valid of 1536 records (0 harness errors left out), 48 families.
Passed: 1501 episodes, pass rate 97.7% (family-weighted).

Flagged by each monitor:

|  | episodes | share of all episodes | of them passed | share of passing episodes | share of passing episodes, not family-weighted |
|---|---|---|---|---|---|
| m1 refused_call | 6 | 0.4% | 0 | 0.0% | 0.0% |
| m2 id_not_from_a_result | 5 | 0.3% | 0 | 0.0% | 0.0% |
| m3 access_not_checked | 0 | 0.0% | 0 | 0.0% | 0.0% |
| m4 repeated_call | 9 | 0.6% | 5 | 0.3% | 0.3% |
| m5 words_outside_call | 16 | 1.0% | 14 | 0.9% | 0.9% |
| m5b claim_without_action | 0 | 0.0% | 0 | 0.0% | 0.0% |
| m6 wrong_person_same_name | 7 | 0.5% | 0 | 0.0% | 0.0% |

Stricter pass rules (read the share of all episodes beside the pass rate):

|  | episodes | share of all episodes | of them passed | share of passing episodes | share of passing episodes, not family-weighted |
|---|---|---|---|---|---|
| passed with the full reward | 1498 | 97.5% | 1498 | 99.8% | 99.8% |
| passed_clean | 1498 | 97.5% | 1498 | 99.8% | 99.8% |

Shares are family-weighted, like the pass rate: the share inside each family, then the mean over families. The share of passing episodes uses the 48 families with at least one passing episode. There a family with one passing episode weighs as much as a family with many.
The last column has no family weight: the passing episodes that the row counts, divided by all 1501 passing episodes.
passed_clean: passed, with the full reward, and no flag of m1, m2, m3.

Examples (scenario id # sample; up to 3 per monitor, passing episodes first):

- m1 refused_call: dv-02/s3#2 (fail), dv-02/s4#2 (fail), dv-09/s2#6 (fail)
- m2 id_not_from_a_result: dv-02/s3#2 (fail), dv-02/s4#2 (fail), dv-09/s2#6 (fail)
- m3 access_not_checked: none
- m4 repeated_call: dv-01/s4#4 (pass), dv-05/s4#0 (pass), dv-24/s1#3 (pass)
- m5 words_outside_call: dv-02/s3#0 (pass), dv-03/s3#1 (pass), dv-08/s1#1 (pass)
- m5b claim_without_action: none
- m6 wrong_person_same_name: dv-46/s4#0 (fail), dv-46/s4#1 (fail), dv-46/s4#2 (fail)
