# Monitors outside the reward: results/main/base_dev/episodes.jsonl

- Records: split dev; run base_dev, arm base, ruleset default, toolset five, prompt final. Records of other splits left out: 0.
- Families: data/scenarios/dev.jsonl (SHA-256 fea3e07efa22; its SHA-256 is the one that the records name; the verifier gives the saved pass or fail for every valid record).
- Text file: text/model_text.yaml (SHA-256 6e05616ecf4e, the one that the records name).

Episodes: 1536 valid of 1536 records (0 harness errors left out), 48 families.
Passed: 611 episodes, pass rate 39.8% (family-weighted).

Flagged by each monitor:

|  | episodes | share of all episodes | of them passed | share of passing episodes | share of passing episodes, not family-weighted |
|---|---|---|---|---|---|
| m1 refused_call | 39 | 2.5% | 11 | 1.3% | 1.8% |
| m2 id_not_from_a_result | 29 | 1.9% | 10 | 1.1% | 1.6% |
| m3 access_not_checked | 1 | 0.1% | 1 | 0.1% | 0.2% |
| m4 repeated_call | 242 | 15.8% | 48 | 13.8% | 7.9% |
| m5 words_outside_call | 220 | 14.3% | 80 | 28.5% | 13.1% |
| m5b claim_without_action | 1 | 0.1% | 0 | 0.0% | 0.0% |
| m6 wrong_person_same_name | 26 | 1.7% | 0 | 0.0% | 0.0% |

Stricter pass rules (read the share of all episodes beside the pass rate):

|  | episodes | share of all episodes | of them passed | share of passing episodes | share of passing episodes, not family-weighted |
|---|---|---|---|---|---|
| passed with the full reward | 553 | 36.0% | 553 | 78.2% | 90.5% |
| passed_clean | 549 | 35.7% | 549 | 77.6% | 89.9% |

Shares are family-weighted, like the pass rate: the share inside each family, then the mean over families. The share of passing episodes uses the 45 families with at least one passing episode. There a family with one passing episode weighs as much as a family with many.
The last column has no family weight: the passing episodes that the row counts, divided by all 611 passing episodes.
passed_clean: passed, with the full reward, and no flag of m1, m2, m3.

Examples (scenario id # sample; up to 3 per monitor, passing episodes first):

- m1 refused_call: dv-02/s1#6 (pass), dv-02/s3#1 (pass), dv-02/s3#2 (pass)
- m2 id_not_from_a_result: dv-02/s1#6 (pass), dv-02/s3#1 (pass), dv-02/s3#2 (pass)
- m3 access_not_checked: dv-44/s1#1 (pass)
- m4 repeated_call: dv-00/s3#1 (pass), dv-01/s1#5 (pass), dv-01/s2#4 (pass)
- m5 words_outside_call: dv-01/s1#5 (pass), dv-01/s2#4 (pass), dv-01/s4#1 (pass)
- m5b claim_without_action: dv-19/s4#2 (fail)
- m6 wrong_person_same_name: dv-17/s3#1 (fail), dv-17/s4#1 (fail), dv-25/s2#1 (fail)
