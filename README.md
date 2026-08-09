# OrbitDesk Support Agent

A local-first, graph-orchestrated support agent for the fictional OrbitDesk product. The system uses **LangGraph**, local Hugging Face models, retrieval over the provided knowledge base, and deterministic verification to produce grounded support responses.

No remote LLM API is required.

## Architecture

```text
START
  |
triage
  |
  +-- requires_clarification / out_of_scope
  |              |
  |              v
  |       format_response -> END
  |
  +-- answerable / requires_escalation
                 |
                 v
             retrieval
                 |
                 v
             generation
                 |
                 v
            verification
             /       \
          pass       fail
           |          |
           v          v
   format_response  increment_retry
           |          |
           v          v
          END      generation
                       |
                       v
                  verification
                   /        \
                pass        fail
                 |            |
                 v            v
          format_response  safe_failure
                              |
                              v
                             END
```

### Node Responsibilities

| Node              | Responsibility                                                                |
| ----------------- | ----------------------------------------------------------------------------- |
| `triage`          | Classifies requests as answerable, clarification, escalation, or out-of-scope |
| `retrieval`       | Retrieves relevant passages from KB documents and resolved cases              |
| `generation`      | Generates an answer using retrieved evidence                                  |
| `verification`    | Checks grounding, source attribution, schema, and unsafe content              |
| `increment_retry` | Increments the retry counter after verification failure                       |
| `format_response` | Produces the final structured response                                        |
| `safe_failure`    | Terminates safely when an answer cannot be verified                           |

The retry loop is capped at **one retry** to prevent infinite graph execution.

## Project Structure

```text
orbitdesk-support-agent/
├── data/
│   ├── *.md
│   ├── resolved_cases.json
│   └── sample_questions.json
├── docs/
│   └── DESIGN.md
├── scripts/
│   ├── run_cli.py
│   └── run_sample_questions.py
├── src/
│   ├── graph.py
│   ├── nodes.py
│   ├── models.py
│   ├── retrieval.py
│   └── state.py
├── tests/
│   ├── test_graph_routing.py
│   └── test_schema.py
├── requirements.txt
└── README.md
```

## Local Models

| Purpose    | Model                                    | Revision                                   |
| ---------- | ---------------------------------------- | ------------------------------------------ |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |
| Generation | `Qwen/Qwen2.5-0.5B-Instruct`             | `7ae557604adf67be50417f59c2c2f167def9a775` |

Both models run locally through `sentence-transformers` and `transformers`.

The demonstrated runs use **CPU only**. The terminal confirms:

```text
No device provided, using cpu
```

The retrieval index currently contains **53 chunks**.

## Setup

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The first run downloads the local models from Hugging Face. Once cached, the application can run without a remote LLM API.

## Running

```powershell
python scripts/run_cli.py "Can a read-only Viewer create an API credential?"
```

For the complete sample workflow:

```powershell
python scripts/run_sample_questions.py
```

## Sample Output

Example: directly answerable API credential question.

```text
NODE TRACE:
triage -> retrieval -> generation -> verification -> format_response
```

```json
{
  "classification": "answerable",
  "answer": "No, a read-only Viewer cannot create an API credential. (CASE-1058)",
  "sources": [
    {
      "source_id": "CASE-1058",
      "passage": "Title: Viewer could not create an API credential..."
    },
    {
      "source_id": "KB-002",
      "passage": "A Viewer cannot ... create API credentials."
    },
    {
      "source_id": "KB-005",
      "passage": "Only Owners and Admins can create or revoke credentials."
    }
  ],
  "confidence": 0.61,
  "requires_human": false,
  "reason": "Request references specific OrbitDesk workflow(s) that the knowledge base covers.",
  "clarification_question": null,
  "warnings": []
}
```

This demonstrates the complete **triage → retrieval → generation → verification** path and shows the evidence used to produce the answer.

## Testing

Run:

```powershell
pytest tests/ -v
```

`tests/test_graph_routing.py` verifies graph behavior without depending on the exact wording generated by the language model. It checks routing, conditional branches, retry behavior, and termination.

`tests/test_schema.py` validates the structured response schema.

## Hardware

| Component         | Configuration                    |
| ----------------- | -------------------------------- |
| OS                | Windows                          |
| Device            | CPU                              |
| GPU / Accelerator | None                             |
| RAM               | Add value from final machine run |
| Retrieval         | CPU                              |
| Generation        | CPU                              |

Recent real runs showed approximately **0.12–0.39 seconds** for embedding-model loading and **0.30–0.74 seconds** for generation-model loading. Generation itself is slower on CPU and varies depending on the query.

## Offline Fallback

A lightweight fallback is available for testing and development:

```powershell
$env:USE_LOCAL_HF_MODELS="0"
```

It uses TF-IDF retrieval and a deterministic template generator. The real submission path uses the local Hugging Face models.

## Limitations

* Verification currently uses a lexical-overlap heuristic rather than a semantic entailment model.
* Triage is rule-based and may not recognize every possible phrasing.
* The 0.5B generation model can occasionally produce incorrect answers, making verification and safe-failure important.
* CPU inference increases generation latency.

## Future Improvements

Potential improvements include:

* Semantic/NLI-based verification
* More robust intent classification
* Retrieval-index caching
* Faster local inference
* A persistent vector index for larger knowledge bases
* More adversarial and edge-case routing tests
