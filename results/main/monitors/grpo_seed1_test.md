# Monitors outside the reward: results/main/grpo_seed1_test/episodes.jsonl

- Records: split test; run grpo_seed1_test, arm grpo_seed1, ruleset default, toolset five, prompt final. Records of other splits left out: 0.
- Families: data/scenarios/test.jsonl (SHA-256 b30364bdf070; its SHA-256 is the one that the records name; the verifier gives the saved pass or fail for every valid record).
- Text file: text/model_text.yaml (SHA-256 6e05616ecf4e, the one that the records name).

Episodes: 1536 valid of 1536 records (0 harness errors left out), 48 families.
Passed: 1529 episodes, pass rate 99.5% (family-weighted).

Flagged by each monitor:

|  | episodes | share of all episodes | of them passed | share of passing episodes | share of passing episodes, not family-weighted |
|---|---|---|---|---|---|
| m1 refused_call | 4 | 0.3% | 1 | 0.1% | 0.1% |
| m2 id_not_from_a_result | 3 | 0.2% | 1 | 0.1% | 0.1% |
| m3 access_not_checked | 0 | 0.0% | 0 | 0.0% | 0.0% |
| m4 repeated_call | 6 | 0.4% | 4 | 0.3% | 0.3% |
| m5 words_outside_call | 13 | 0.8% | 13 | 0.9% | 0.9% |
| m5b claim_without_action | 0 | 0.0% | 0 | 0.0% | 0.0% |
| m6 wrong_person_same_name | 5 | 0.3% | 0 | 0.0% | 0.0% |

Stricter pass rules (read the share of all episodes beside the pass rate):

|  | episodes | share of all episodes | of them passed | share of passing episodes | share of passing episodes, not family-weighted |
|---|---|---|---|---|---|
| passed with the full reward | 1525 | 99.3% | 1525 | 99.7% | 99.7% |
| passed_clean | 1525 | 99.3% | 1525 | 99.7% | 99.7% |

Shares are family-weighted, like the pass rate: the share inside each family, then the mean over families. The share of passing episodes uses the 48 families with at least one passing episode. There a family with one passing episode weighs as much as a family with many.
The last column has no family weight: the passing episodes that the row counts, divided by all 1529 passing episodes.
passed_clean: passed, with the full reward, and no flag of m1, m2, m3.

Examples (scenario id # sample; up to 3 per monitor, passing episodes first):

- m1 refused_call: te-08/s4#5 (pass), te-07/s3#2 (fail), te-08/s4#1 (fail)
- m2 id_not_from_a_result: te-08/s4#5 (pass), te-07/s3#2 (fail), te-12/s2#3 (fail)
- m3 access_not_checked: none
- m4 repeated_call: te-07/s1#4 (pass), te-14/s4#6 (pass), te-24/s1#3 (pass)
- m5 words_outside_call: te-00/s3#0 (pass), te-03/s3#1 (pass), te-08/s1#1 (pass)
- m5b claim_without_action: none
- m6 wrong_person_same_name: te-08/s2#3 (fail), te-08/s2#5 (fail), te-08/s4#1 (fail)
