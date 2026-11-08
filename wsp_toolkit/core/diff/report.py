"""Audit report rendering: Markdown (canonical) and PDF (via reportlab, optional)."""

from __future__ import annotations

import io


def to_markdown(rep: dict) -> str:
    L = [f"# Fine-tune audit: {rep['title']}", ""]
    L += [f"- Base model: `{rep['base']['source']}` (hash `{rep['base']['hash'][:16]}`)",
          f"- Fine-tuned model: `{rep['tuned']['source']}` (hash `{rep['tuned']['hash'][:16]}`)",
          f"- Generated: {rep['generated_at']} by WSP Toolkit {rep['toolkit_version']}", ""]
    L += ["## Summary of findings", ""]
    for i, f in enumerate(rep["findings"], 1):
        L.append(f"{i}. **{f['type'].replace('_', ' ')}** — {f['text']}")
    wd = rep["weights"]
    L += ["", "## What changed (weights)", "",
          f"{wd['params_changed']} of {wd['params_total']} parameters changed ({wd['pct_changed']}%) across "
          f"{wd['tensors_changed']} tensors.", "",
          "| Tensor | Changed | Rel. change | Rank for 90% energy | Top-1 energy | Most-changed rows | cols |",
          "|---|---|---|---|---|---|---|"]
    for t in wd["most_changed_tensors"]:
        L.append(f"| `{t['tensor']}` | {t['changed']}/{t['numel']} | {t['rel_change']} | {t.get('rank_90', '')}"
                 f"/{t.get('max_rank', '')} | {t.get('top1_energy', '')} | "
                 f"{[r[0] for r in t.get('top_rows', [])]} | {[c[0] for c in t.get('top_cols', [])]} |")
    if wd["most_changed_heads"]:
        L += ["", "Most-changed attention heads: " + ", ".join(f"{h['head']} ({h['tensor'].split('.')[-2]}, "
                                                                 f"{h['delta_norm']})" for h in wd["most_changed_heads"][:6])]
    if rep["readouts"]:
        L += ["", "## Vocabulary readout of the top weight-change directions", "",
              "Singular directions have no inherent sign, so the two columns are the two poles of each direction.", "",
              "| Tensor | Component | Pole A | Pole B |", "|---|---|---|---|"]
        for r in rep["readouts"]:
            if r.get("readout"):
                L.append(f"| `{r['tensor']}` | {r['component']} | {', '.join(r['readout']['promotes'])} | "
                         f"{', '.join(r['readout']['suppresses'])} |")
    if rep["features"]:
        L += ["", "## Feature mapping (M6)", "",
              "Delta directions matched against the feature index; fidelity = how well the label predicts firing; "
              "unexplained = fraction of that layer's activity the dictionary does not explain.", ""]
        for m in rep["features"][:8]:
            fs = "; ".join(f"{f['label']} (cos {f['abs_cosine']}, fid {f['fidelity']})" for f in m["features"])
            L.append(f"- `{m['tensor']}` comp {m['component']} ({m['side']}) @ {m['hook']} "
                     f"[unexplained {m['dict_unexplained_fraction']}]: {fs}")
    L += ["", "## Behaviour (M3 suites)", "", "| Suite | Split | Base | Fine-tuned | Δ | ±SE |", "|---|---|---|---|---|---|"]
    for s in rep["suites"]:
        L.append(f"| {s['suite']} | {s['split']} | {s['base']} | {s['tuned']} | {s['delta']} | {s['stderr']} |")
    b = rep["behaviour"]
    L += ["", "## Divergence scan", ""]
    if b.get("kind") == "language_model":
        L += [f"{b['n_probes']} probe sequences, {b['positions']} positions; mean KL {b['mean_kl']}, "
              f"argmax disagreement {b['argmax_disagreement_rate']}.", ""]
        for key, lab in [("trigger_candidates_current_token", "current token"),
                         ("trigger_candidates_previous_token", "previous token"),
                         ("trigger_candidates_in_context", "anywhere in context")]:
            if b[key]:
                L.append(f"- Over-represented ({lab}): " + ", ".join(f"{x['token']} (lift {x['lift']}, n={x['count']})"
                                                                    for x in b[key][:5]))
        ms = b.get("mean_logprob_shift")
        if ms:
            L.append("- Average log-prob shift (vs median token) — most suppressed: " +
                     ", ".join(f"{x['token']} ({x['shift']})" for x in ms["most_suppressed"][:5]) +
                     "; most promoted: " + ", ".join(f"{x['token']} ({x['shift']})" for x in ms["most_promoted"][:5]))
        if b["tuned_model_now_predicts"]:
            L.append("- Where they disagree, fine-tuned predicts: " +
                     ", ".join(f"{x['token']} ({x['count']})" for x in b["tuned_model_now_predicts"][:5]))
    elif b:
        L += [f"{b['n_probes']} probe inputs; disagreement rate {b['disagreement_rate']}.",
              f"- Inputs associated with disagreement: {b['inputs_associated_with_disagreement']}",
              f"- Prediction shifts: {b['prediction_shift']}"]
    if rep["sensitive_flags"]:
        L += ["", "## Sensitive-component flags", ""]
        for f in rep["sensitive_flags"]:
            L.append(f"- {f['kind']} `{f['target']}`: {f.get('reason', '')} {f.get('alignment', '')}")
    L += ["", "## Method and caveats", "",
          "- Weight deltas are exact; low-rank structure from SVD of each delta.",
          "- Feature labels come from the configured labeler and are only as good as their fidelity scores.",
          "- The divergence scan only finds behaviours the probe set exercises; absence of evidence is not evidence "
          "of absence.", f"- Settings: {rep['settings']}"]
    return "\n".join(L) + "\n"


def markdown_to_pdf(md: str) -> bytes:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph, Preformatted, SimpleDocTemplate, Spacer
    except ImportError as e:
        raise RuntimeError("PDF export needs reportlab: pip install 'wsp-toolkit[pdf]'") from e
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, title="WSP fine-tune audit")
    st = getSampleStyleSheet()
    flow = []
    for line in md.splitlines():
        esc = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        esc = esc.replace("**", "")
        if line.startswith("# "):
            flow.append(Paragraph(esc[2:], st["Title"]))
        elif line.startswith("## "):
            flow.append(Paragraph(esc[3:], st["Heading2"]))
        elif line.startswith("|"):
            flow.append(Preformatted(line, st["Code"]))
        elif line.strip():
            flow.append(Paragraph(esc, st["BodyText"]))
        else:
            flow.append(Spacer(1, 4))
    doc.build(flow)
    return buf.getvalue()
