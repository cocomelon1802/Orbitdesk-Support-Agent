# Design notes

## Why rule-based triage instead of model-based triage

Triage (`src/nodes.py::triage_node`) uses deterministic regex patterns
rather than asking the local LLM to classify the request. Trade-offs:

**Pros:**
- Zero latency, zero model calls, 100% reproducible.
- Fully explainable: the `triage_reason` field always traces back to a
  specific pattern match, useful for debugging and for the "why did the
  system do this" part of the assignment's spirit.
- Makes the automated routing tests independent of any model's wording —
  directly satisfies "at least one automated test must verify graph
  routing without depending on the exact wording produced by the model."
- A 0.5B-parameter local model is not reliably good at multi-class
  classification with strict output formatting; rules avoid that failure
  mode entirely for a well-scoped, small domain.

**Cons / limitations:**
- Won't generalize to phrasings outside the observed patterns. A
  differently-worded ambiguous question (e.g. "things seem off with my
  reports") might slip through as `answerable` if it happens to mention a
  keyword like "schedule" in passing.
- A production system would likely use a hybrid: rules for the
  unambiguous/adversarial cases (refunds, prompt-injection attempts —
  where you want zero-tolerance determinism), and a small classifier or
  the LLM itself for the nuanced answerable/clarification/escalation
  distinction.

**What I'd improve with more time:** train or prompt-tune a lightweight
zero-shot classifier (e.g. `facebook/bart-large-mnli` or a distilled
sentence-transformer classification head) as a fallback when no rule
fires with high confidence, keeping the deterministic rules as a
non-negotiable safety net for out-of-scope/injection detection.

## Why a lexical-overlap grounding check instead of an NLI model

Verification's hallucination check (`_grounding_ratio` in
`src/nodes.py`) computes token overlap between the draft answer and the
retrieved evidence, rather than running a proper entailment/NLI model.

**Pros:** cheap, fast, no extra model to load, easy to reason about and
tune (`GROUNDING_OVERLAP_THRESHOLD`).

**Cons:** it's a weak proxy for "is this claim actually supported."
Common stopwords inflate the overlap score, so an answer could pass the
grounding check while still making an unsupported specific claim (e.g.
inventing a number that happens to be surrounded by real, evidence-backed
words). It also can't catch a *contradiction* of the evidence — only
totally unrelated text.

**What I'd improve with more time:** replace or supplement this with a
local NLI model (e.g. a small `cross-encoder/nli-...` model) run
per-sentence against the retrieved passages, which would catch
contradictions and unsupported specific claims rather than just
"topically unrelated" text.

### A real example of this limitation, found during testing

Running the real local model (`Qwen/Qwen2.5-0.5B-Instruct`) against Q-002
("Can a read-only Viewer create an API credential?") produced: *"Yes, a
read-only Viewer can create an API credential."* — which is wrong. Every
retrieved passage (KB-002, KB-005, CASE-1058) says the opposite: Viewers
**cannot** create API credentials. The lexical grounding check passed
this answer without complaint, because the answer shares plenty of real
vocabulary with the evidence ("Viewer," "API," "credential") — it just
flipped the polarity.

This is a genuinely useful finding about small-model behavior on
negation-heavy permission questions, and a good illustration of why
"grounded in evidence words" and "agrees with evidence claims" are
different properties. I added a second, narrower check
(`_contradiction_note` in `src/nodes.py`) specifically for this failure
mode: for yes/no-style questions, it compares the answer's polarity
(does it start with "Yes"/"No", or use "can"/"cannot" near the start)
against the polarity of the most topically-overlapping evidence sentence
(does that sentence contain a negation like "cannot"/"not allowed"). If
they disagree, verification fails and the retry fires with that specific
contradiction named in the revision feedback. I also strengthened the
system prompt with an explicit negation-handling instruction and a
worked example.

This is intentionally domain-shaped rather than a general solution — it
only fires for yes/no-phrased questions and only catches polarity
flips, not more subtle factual errors. See `tests/test_graph_routing.py
::test_contradiction_check_catches_wrong_polarity_answer` for the
regression test built directly from this real failure.

## Why the retry is capped at exactly 1

The assignment asks for "at least one retry, revision, or fallback path"
and "protection against an infinite graph loop." `MAX_RETRIES = 1` in
`src/nodes.py` plus the `retry_count` field in `GraphState` guarantee
`generation` runs at most twice per request: once normally, once after a
single revision prompt that includes the verification failure reason. If
the second attempt also fails, the graph deterministically routes to
`safe_failure` rather than retrying indefinitely. This is enforced at the
graph-edge level (`route_after_verification`), not just by convention, so
it holds even if a future node change tried to loop back further.

## Why resolved cases are chunked one-per-case rather than split further

The 8 resolved cases are short and each describes one coherent incident.
Splitting them further (e.g. symptoms vs. resolution as separate chunks)
would risk retrieving a resolution step without its corresponding
symptom context, or vice versa — worse for a system whose job is to match
a *situation* to a *procedure*. Keeping each case atomic trades a
slightly larger chunk size for coherence.

## Why superseded cases stay in the index

`CASE-0914` (legacy personal API tokens, removed in OrbitDesk 4.0) is
deliberately kept retrievable rather than filtered out at index-build
time, per the README's own instruction: *"A resolved case marked
'superseded' may be useful for testing retrieval and verification, but
its resolution must not be presented as current guidance."* The
generation prompt explicitly instructs the model not to present
superseded resolutions as current, and each retrieved chunk carries its
`doc_status` so this is checkable, not just requested. A stronger version
of this (not implemented due to time) would have verification actively
penalize any answer that cites a `superseded`-status source without an
explicit caveat.

## Known limitations / what I'd do with more time

1. **Grounding check is lexical, not semantic** — see above.
2. **Triage is rule-based** — see above.
3. **No reranking step** — retrieval returns raw embedding-similarity
   top-k. A local cross-encoder reranker (also explicitly allowed by the
   assignment: "embedding, reranking or classification model") would
   likely improve precision on the two-document questions (Q-001, which
   needs both KB-003 and KB-004 to fully answer).
4. **Single-turn only** — a real clarification flow would carry
   conversation history so the user's follow-up answer feeds back into a
   second triage/retrieval pass. Currently `requires_clarification`
   terminates the graph with a question for the user, and a fresh
   `graph.invoke()` call would be needed once they respond.
5. **Confidence score is model-reported, not calibrated** — the local LLM
   is asked to self-report a confidence between 0 and 1, which is a weak
   signal. A calibrated confidence would need either a held-out
   validation set or a proxy like retrieval-score-weighted grounding
   ratio.
