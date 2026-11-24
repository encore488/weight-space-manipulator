# `feature_get`

Full feature record: label, fidelity, top examples, vocabulary readout, neurons, zoom links, up/downstream features.

**Who can call it:** agents (MCP) and humans.

## Inputs

| Name | Type | Default |
|---|---|---|
| `feature_id` | `str` | required |
| `session_id` | `str | None` | `null` |

## Size limits

Response capped at 16000 bytes (paged via `result_page` if larger).

## Example

Call (MCP tool arguments, or `wsp call feature_get '<json>'`):

```json
{
 "feature_id": "fd_e2bae00b39:13"
}
```

Result (fixture output with stable example IDs and times):

```json
{
 "feature_id": "fd_e2bae00b39:13",
 "dict_id": "fd_e2bae00b39",
 "sensitive": false,
 "hook": "L0.resid_post",
 "idx": 13,
 "label": "previous token is '<bos>'",
 "fidelity": 0.7059,
 "label_train_f1": 0.7551,
 "alternatives": [
  {
   "label": "position 1",
   "f1": 0.7551
  },
  {
   "label": "position 3",
   "f1": 0.2067
  }
 ],
 "firing_rate": 0.07639,
 "top_examples": [
  {
   "tokens": [
    "<bos>",
    "t13"
   ],
   "position": 1,
   "activation": 0.6772,
   "sequence_tag": "induction"
  },
  {
   "tokens": [
    "<bos>",
    "t13"
   ],
   "position": 1,
   "activation": 0.6772,
   "sequence_tag": "counting"
  },
  {
   "tokens": [
    "<bos>",
    "t13"
   ],
   "position": 1,
   "activation": 0.6772,
   "sequence_tag": "random"
  },
  {
   "tokens": [
    "<bos>",
    "t7"
   ],
   "position": 1,
   "activation": 0.6604,
   "sequence_tag": "induction"
  },
  {
   "tokens": [
    "<bos>",
    "t7"
   ],
   "position": 1,
   "activation": 0.6604,
   "sequence_tag": "counting"
  }
 ],
 "neurons": [
  [
   6,
   0.2548
  ],
  [
   29,
   -0.2439
  ],
  [
   48,
   -0.2411
  ],
  [
   3,
   -0.2258
  ],
  [
   58,
   0.2107
  ],
  [
   63,
   -0.2095
  ],
  [
   7,
   0.2086
  ],
  [
   15,
   -0.2077
  ]
 ],
 "direction_id": "dir_de5d5ae3e1493b3d",
 "readout": {
  "promotes": [
   "t14",
   "t29",
   "t24",
   "t20",
   "t8"
  ],
  "suppresses": [
   "t6",
   "t26",
   "<bos>",
   "t2",
   "t21"
  ]
 },
 "downstream": [
  [
   "fd_95fe6301b7:22",
   0.5221
  ],
  [
   "fd_95fe6301b7:29",
   0.1357
  ],
  [
   "fd_95fe6301b7:0",
   0.09634
  ]
 ],
 "dictionary_unexplained_fraction": 0.4985
}
```
