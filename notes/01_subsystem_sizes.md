# Block 1.3 — Subsystem sizes

**Command pattern:** `find <dir> -name "*.py" -not -path "*/__pycache__/*" | wc -l`

Fresh counts (this machine):

| Dir | `.py` count | Mark | Note |
|---|---:|---|---|
| `ml/data` | 36 | **large** | Worth more time (gold, video, registry). |
| `ml/models` | 35 | **large** | Worth more time (forward/Stage A–B). |
| `ml/training` | 30 | **large** | Worth more time (loop, export, losses). |
| `ml/retrieval` | 26 | **large** | Worth more time (advisory RAG path). |
| `tools/simulation` | 26 | **large** | Worth more time (runtime integration surface). |
| `app/` | 19 | **medium** | Personal mode + First Wave app layer. |
| `ml/therapy` | 14 | **medium** | Closed-loop engine; dense but smaller surface. |
| `ml/infra` | 10 | **small / skim** | Cloud/ops helpers; sample key files only. |
| `ml/pipeline` | 5 | **tiny / skim** | Offline entrypoints; quick pass. |

## Time allocation hint

Spend deep-read budget on **large** rows first (`data`, `models`, `training`, `retrieval`, simulator). Skim `pipeline` and spot-check `infra` unless Block 2/5 governance checks pull you in.
