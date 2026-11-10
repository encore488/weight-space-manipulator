# `audit_get`

Markdown text of a stored audit report.

**Who can call it:** agents (MCP) and humans.

## Inputs

| Name | Type | Default |
|---|---|---|
| `audit_id` | `str` | required |
| `session_id` | `str | None` | `null` |

## Size limits

Response capped at 16000 bytes (paged via `result_page` if larger).

## Example

Call (MCP tool arguments, or `wsp call audit_get '<json>'`):

```json
{
 "audit_id": "audit_104776f003f3"
}
```

Result (real output from the induction fixture):

```json
{
 "audit_id": "audit_104776f003f3",
 "markdown": "# Fine-tune audit: fixture:induction_tf vs fixture:induction_tf_broken\n\n- Base model: `fixture:induction_tf` (hash `ed5c84a209e1deb3`)\n- Fine-tuned model: `fixture:induction_tf_broken` (hash `2efebd42c08ad9b4`)\n- Generated: 2026-11-10 02:48:09 by WSP Toolkit 0.2.0\n\n## Summary of findings\n\n1. **behavior change** \u2014 drop of 0.939 on induction_heldout\n2. **behavior change** \u2014 drop of 0.946 on induction_practice\n3. **behavior change** \u2014 drop of 0.956 on induction_regression\n4. **trigger candidate** \u2014 divergence concentrated where the token in context is t13, t14, t2 (lift 2.88)\n5. **prediction shift** \u2014 where the models disagree, the tuned model now predicts t26, t15, t3\n6. **vocab direction** \u2014 top change direction in blocks.1.attn.o_proj.weight separates ['t3', 't30', 't5'] from ['t17', 't19', 't21'] (the sign cannot be read from weights alone)\n7. **vocab direction** \u2014 top change direction in blocks.1.attn.o_proj.weight separates ['t29', 't10', 't4'] from ['t11', 't12', 't28'] (the sign cannot be read from weights alone)\n8. **weights** \u2014 largest relative change in blocks.1.attn.o_proj.weight (0.8952); 90% of its delta energy in rank 26\n\n## What changed (weights)\n\n4096 of 38400 parameters changed (10.67%) across 1 tensors.\n\n| Tensor | Changed | Rel. change | Rank for 90% energy | Top-1 energy | Most-changed rows | cols |\n|---|---|---|---|---|---|---|\n| `blocks.1.attn.o_proj.weight` | 4096/4096 | 0.8952 | 26/64 | 0.06005 | [6, 48, 9, 51, 20] | [53, 3, 39, 55, 63] |\n\nMost-changed attention heads: 1.3 (o_proj, 4.108), 1.2 (o_proj, 3.855), 1.0 (o_proj, 3.835), 1.1 (o_proj, 2.476)\n\n## Vocabulary readout of the top weight-change directions\n\nSingular directions h
  ... (truncated for the docs)
```
