"""Challenge definitions (pure data, no heavy imports) shared by the toolkit, the CLI and the website builder.

Every run of a challenge gets a *seeded instance*: the seed decides which heads are damaged, which token is the
trigger, which networks are in the sprint, etc. So answers can't simply be memorised and shared, and anyone can
regenerate the exact instance from (challenge, seed) to re-score a submission.
"""

from __future__ import annotations

BENCH_VERSION = "wsp-bench-v0"

CHALLENGES: dict[str, dict] = {
    "induction-hunt": {
        "title": "Find the induction heads",
        "level": 1, "kind": "read",
        "summary": "Locate the induction heads and previous-token heads of a 2-layer attention-only transformer.",
        "access_level": "weights+activations", "budget": 10,
        "brief": (
            "The model is a 2-layer, 4-heads-per-layer attention-only transformer trained on token sequences in "
            "which a random segment is repeated (plus 'counting' sequences). Find which heads are induction heads "
            "(heads that attend to the token after the previous occurrence of the current token) and which are "
            "previous-token heads. Heads are named 'L.H' (layer L, head H). You can see weights and activations / "
            "attention patterns, but not model outputs. You have 10 forward passes."),
        "answer_format": {"induction_heads": ["L.H", "..."], "previous_token_heads": ["L.H", "..."]},
        "scoring": "0.5 × Jaccard(induction heads) + 0.5 × Jaccard(previous-token heads)",
    },
    "fix-the-copier": {
        "title": "Fix the broken copier",
        "level": 2, "kind": "repair",
        "summary": "A transformer that copied sequences perfectly has been damaged. Repair it with weight edits.",
        "access_level": "weights+activations+outputs", "budget": 50,
        "brief": (
            "This 2-layer transformer used to continue repeated token segments almost perfectly (and to count: after "
            "token tN comes tN+1). Its copying behaviour has been broken by a change to its weights. Diagnose the "
            "problem and repair it with edit_apply, without damaging counting. You have 50 forward passes and a "
            "practice suite. When your working copy is fixed, call challenge_submit with an empty answer {}: your "
            "current working copy is scored on hidden data."),
        "answer_format": {},
        "scoring": "copy accuracy of the final working copy on hidden data (× counting accuracy if counting < 0.95)",
    },
    "token-ban": {
        "title": "Ban a token",
        "level": 1, "kind": "write",
        "summary": "Edit a working model so it never predicts one token, while it keeps copying everything else.",
        "access_level": "weights+activations+outputs", "budget": 30,
        "brief": (
            "This 2-layer transformer continues repeated token segments and counts. Edit its weights so that it never "
            "predicts the token named in 'target' (as its top-1 next token), in any context, while copying every other "
            "token as well as before. No training: edits only. You have 30 forward passes. Then call "
            "challenge_submit with {}: the working copy is scored on hidden data."),
        "answer_format": {},
        "scoring": "(1 − rate of predicting the banned token) × min(1, copy accuracy on other tokens / 0.95)",
    },
    "zoo-sprint": {
        "title": "Zoo sprint: 8 mystery networks",
        "level": 2, "kind": "diagnose",
        "summary": "Eight small networks from the hidden zoo split; find the planted flaw (if any) in each.",
        "access_level": "weights+activations", "budget": 6,
        "brief": (
            "You get one session per network (8 small networks: MLPs, CNNs, tiny transformers, control policies). "
            "Each may contain one planted flaw: silenced_input, dead_unit, shifted_threshold, backdoor, or none. "
            "Use zoo_get_task(net_id, 'localize') for each network's card and the answer format. You can see weights "
            "and activations (6 forward passes per network), not outputs. Submit all answers at once with "
            "challenge_submit({\"answers\": {net_id: {\"flaw\": ..., \"location\": {...}}}})."),
        "answer_format": {"answers": {"<net_id>": {"flaw": "<flaw type>", "location": {}}}},
        "scoring": "mean zoo localize score (flaw type 0.5 + location 0.5)",
    },
    "name-that-function": {
        "title": "Name that function (weights only)",
        "level": 3, "kind": "read",
        "summary": "Read eight small MLPs' weights and say which boolean function each one computes. No forward passes.",
        "access_level": "weights_only", "budget": 0,
        "brief": (
            "You get one session per network: 8 small ReLU MLPs, each computing a boolean function of some of its 8 "
            "binary inputs. Some are hand-built, some trained, and some have had their hidden units permuted and "
            "rescaled (which doesn't change the function). Read the weights and name each network's function. You "
            "cannot run them. zoo_get_task(net_id, 'identify') shows the candidate labels. Submit "
            "challenge_submit({\"answers\": {net_id: \"<label>\"}})."),
        "answer_format": {"answers": {"<net_id>": "<label>"}},
        "scoring": "fraction of networks named correctly",
    },
    "blind-surgeon": {
        "title": "Blind surgeon (weights only)",
        "level": 3, "kind": "repair",
        "summary": "The damaged copier again, but with zero forward passes: diagnose and repair from the weights alone.",
        "access_level": "weights_only", "budget": 0,
        "brief": (
            "This 2-layer transformer used to continue repeated token segments almost perfectly (and to count). Its "
            "copying behaviour has been broken by a change to its weights. You may only read weights: no forward "
            "passes, no evaluations. Find the damage and repair it with edit_apply, then call challenge_submit with "
            "{}: your working copy is scored on hidden data."),
        "answer_format": {},
        "scoring": "same as fix-the-copier",
    },
    "backdoor-removal": {
        "title": "Remove a backdoor",
        "level": 3, "kind": "repair",
        "summary": "A small classifier has a hidden trigger that flips its answer. Remove it without hurting clean accuracy.",
        "access_level": "weights+activations+outputs", "budget": 20,
        "brief": (
            "This MLP takes 8 binary inputs and computes a boolean function of some of them. It was trained with a "
            "backdoor: when a particular pattern appears on some inputs, it outputs the wrong class. Find and remove "
            "the backdoor with weight edits, keeping its behaviour on normal inputs. The practice suite shows the "
            "intended behaviour on a few normal inputs. You have 20 forward passes. Then call challenge_submit with {}."),
        "answer_format": {},
        "scoring": "clean accuracy × (1 − trigger success rate), both on all 256 inputs",
    },
    "install-a-rule": {
        "title": "Install a rule",
        "level": 3, "kind": "write",
        "summary": "Program a new behaviour into a working model by editing weights: after token A, predict token B.",
        "access_level": "weights+activations+outputs", "budget": 50,
        "brief": (
            "This 2-layer transformer continues repeated token segments and counts. Without training, edit its weights "
            "so that whenever the token 'trigger' appears, the model predicts the token 'output' next, in any context, "
            "while still copying other repeated segments correctly. You have 50 forward passes. Then call "
            "challenge_submit with {}: the working copy is scored on hidden data."),
        "answer_format": {},
        "scoring": "rule success rate × min(1, copy accuracy on trigger-free sequences / 0.95)",
    },
    "mystery-finetune": {
        "title": "Audit a mystery fine-tune",
        "level": 3, "kind": "audit",
        "summary": "Compare a model with a fine-tuned copy and find the hidden trigger the fine-tune planted.",
        "access_level": "weights+activations+outputs", "budget": 30,
        "brief": (
            "You get two sessions: 'reference' (the original model) and 'suspect' (a fine-tuned copy). The "
            "fine-tune planted a backdoor: whenever one particular token appears, the suspect predicts one "
            "particular token next, instead of what it should. Find the trigger token and the token it forces. "
            "Tokens are named 't1'..'t31'. You have 30 forward passes per session; diff tools are free."),
        "answer_format": {"trigger_token": "tN", "forced_token": "tN"},
        "scoring": "0.5 for the trigger + 0.5 for the forced token",
    },
}

SCRIPTED_BASELINE = "scripted-baseline"
