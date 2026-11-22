"""Family files, scenario ids, and the fixed subsets of the dev split.

A scenario is one family in one wording style. Its id is "{family_id}/s{style}".
"""
from __future__ import annotations

import json
from pathlib import Path

from accessdesk.config import RunConfig
from accessdesk.env.rules import RULESETS
from accessdesk.rng import Rng, derive_seed
from accessdesk.types import OUTCOMES

CURVE_PER_OUTCOME = 5        # curve subset: 20 dev families
EXTRA_TOOL_PER_OUTCOME = 4   # extra-tool set: 16 dev families
N_SHIFT_DIFFER = 11          # policy-shift set: families whose correct end state changes under the new rule
N_SHIFT_CONTROL = 5          # policy-shift set: families whose correct end state stays the same


def save_families(path: str | Path, families: list[dict]) -> None:
    """One JSON object per line, keys in record order, UTF-8, "\\n" line ends."""
    lines = [json.dumps(family, sort_keys=False, ensure_ascii=False) + "\n" for family in families]
    Path(path).write_bytes("".join(lines).encode("utf-8"))


def load_families(path: str | Path) -> list[dict]:
    lines = Path(path).read_bytes().decode("utf-8").split("\n")
    return [json.loads(line) for line in lines if line]


def scenario_ids(family: dict) -> list[str]:
    """The ids of a family's scenarios, one per wording, in style order."""
    return [f"{family['family_id']}/s{wording['style']}" for wording in family["wordings"]]


def split_scenario_id(sid: str) -> tuple[str, int]:
    """"dv-007/s2" -> ("dv-007", 2)."""
    family_id, separator, style = sid.rpartition("/s")
    if not separator or not family_id or not style.isdigit():
        raise ValueError(f"not a scenario id: {sid!r}")
    return family_id, int(style)


def make_sets(dev_families: list[dict], cfg: RunConfig, variant: str = "all_grants_need_approval") -> dict:
    """The fixed subsets of the dev split, chosen by cfg.data_seed. Every list holds family ids, sorted.

    curve_subset:  5 families per outcome (20).
    extra_tool:    4 families per outcome (16).
    policy_shift:  "variant": the rule set, for the record.
                   "differ": 11 families whose expected state under the rule set `variant` differs from the
                   default one.
                   "control": 5 families where it is the same, taken in turn from each outcome that has
                   such families (at least two outcomes), with one more condition: a control must be a
                   family on which the constant policy that fits the changed rule does NOT pass.

    The rule for the controls. A constant policy always acts for one resolution, whatever the facts are,
    and it passes exactly the families whose correct resolution that is. The constant policy that fits the
    changed rule is the one whose resolution is the new correct resolution of the "differ" families: for
    all_grants_need_approval "always ask the manager for approval and close as pending", for
    sensitive_denied "always close as denied". It passes every "differ" family under the new rule without
    reading one fact. If it also passed a control, that control could not tell it from a model that
    follows the new rule. So a control is a family whose correct resolution (the same under both rule
    sets) is another one: for all_grants_need_approval 3 already-held and 2 inactive families, never an
    approval family; for sensitive_denied 3 grant and 2 already-held families, never an inactive one.

    `variant` must be the text bundle's policy_shift_variant. Raises ValueError if the dev split is too small.
    """
    if variant not in RULESETS or variant == "default":
        raise ValueError(f"variant must be a rule set other than 'default', not {variant!r}")
    by_outcome = {outcome: [f for f in dev_families if f["outcome"] == outcome] for outcome in OUTCOMES}
    differ = [f for f in dev_families if f["expected"][variant] != f["expected"]["default"]]
    new_resolutions = {_resolution(f, variant) for f in differ}      # what the fitting constant policy does
    same_by_outcome = {outcome: [f for f in families if f["expected"][variant] == f["expected"]["default"]
                                 and _resolution(f, "default") not in new_resolutions]
                       for outcome, families in by_outcome.items()}
    return {
        "curve_subset": _per_outcome(by_outcome, CURVE_PER_OUTCOME, cfg, "curve_subset"),
        "policy_shift": {
            "variant": variant,
            "differ": _pick(differ, N_SHIFT_DIFFER, cfg, "policy_shift_differ"),
            "control": _in_turn(same_by_outcome, N_SHIFT_CONTROL, cfg, "policy_shift_control"),
        },
        "extra_tool": _per_outcome(by_outcome, EXTRA_TOOL_PER_OUTCOME, cfg, "extra_tool"),
    }


def _resolution(family: dict, ruleset: str) -> str:
    """The resolution that closes the family's request correctly under a rule set."""
    return family["expected"][ruleset]["request"]["resolution"]


def _pick(families: list[dict], k: int, cfg: RunConfig, label: str) -> list[str]:
    """k family ids, chosen by a seed that depends on the label; sorted."""
    if len(families) < k:
        raise ValueError(f"{label}: {k} families are needed, but only {len(families)} are available")
    ids = [family["family_id"] for family in families]
    return sorted(Rng(derive_seed(cfg.data_seed, "sets", label)).sample(ids, k))


def _per_outcome(by_outcome: dict[str, list[dict]], k: int, cfg: RunConfig, label: str) -> list[str]:
    """k family ids of each outcome; sorted."""
    picked = [_pick(by_outcome[outcome], k, cfg, f"{label}/{outcome}") for outcome in OUTCOMES]
    return sorted(family_id for ids in picked for family_id in ids)


def _in_turn(by_outcome: dict[str, list[dict]], k: int, cfg: RunConfig, label: str) -> list[str]:
    """k family ids, taken one at a time from each outcome in turn, so that the outcomes are spread evenly."""
    queues = [Rng(derive_seed(cfg.data_seed, "sets", f"{label}/{outcome}")).shuffle(
                  [family["family_id"] for family in by_outcome[outcome]])
              for outcome in OUTCOMES if by_outcome[outcome]]
    if len(queues) < 2 or sum(len(queue) for queue in queues) < k:
        raise ValueError(f"{label}: {k} families from at least two outcomes are needed")
    chosen: list[str] = []
    while len(chosen) < k:
        for queue in queues:
            if queue and len(chosen) < k:
                chosen.append(queue.pop())
    return sorted(chosen)
