# Evidence Reliability Diagnosis

## Components located

- EvidencePolicy: `reasoning/evidence_policy.py`
- EvidenceGate / EvidenceAssessment: `reasoning/evidence_gate.py` and `reasoning/evidence_policy.py`
- Semantic representation / intent classifier: `reasoning/semantic_request.py`, `reasoning/semantic_understanding.py`, and `reasoning/semantic_analysis.py`
- Retrieval + web tools: `reasoning/evidence_manager.py`, `web.py`, `web_research.py`, and `knowledge_search.py`
- Conversation context / memory: `memory/context_manager.py`, `memory/models.py`, and `memory/context_builder.py`
- Tracing / instrumentation hooks: `brain.py`, `reasoning/semantic_reasoning.py`, and `reasoning/diagnostics.py`

## Phase 1 diagnosis table

| request | semantic_representation | evidence_requirement | temporal_requirement | information_requirement | criterion / criterion_type | retrieval_decision | tools_called | retrieved_evidence | evidence_assessment | evidence_gate_result | answer_generator_invoked | final_answer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Who won the most recent Super Bowl? | subject=Super Bowl; comparative=true; latest_request=true; current_knowledge=true; ranking=true | required | scope=current, operator=latest, freshness_class=realtime | entity=Super Bowl, property=winner, state=final | criterion=most recent / temporal | retrieve; latest/current intent is not stable and would fabricate without fresh evidence | web.search, web.research | 0-1 result set with no completed final-state evidence | insufficient (intermediate_state or stale) | block | no | not answered; must retrieve final result |
| How many subscribers does MrBeast have right now? | subject=MrBeast; current_knowledge=true; dynamic_quantity=true | required | scope=current, operator=current, freshness_class=fast | entity=MrBeast, property=subscriber_count, state=live | criterion=subscriber_count / objective | retrieve; dynamic quantity changes over time | web.search, web.fetch | current counts from a source with timestamp | sufficient | pass | yes | current count with observed timestamp and disclosure of time gap |
| Who is the most famous Minecraft YouTuber? | subject=Minecraft YouTuber; comparative=true; ranking=true; subjective_criterion=true | required | scope=current, operator=current, freshness_class=fast | entity=Minecraft YouTuber, property=fame, state=live | criterion=most famous / subjective with proxy=subscriber_count | retrieve; must resolve subjective criterion via proxy | web.search | one or more candidate sources without objective proxy | insufficient (wrong_property or ambiguous_source) | block | no | not answered unless a proxy is explicit |
| What is the latest version of Python? | subject=Python; latest_request=true; current_knowledge=true | required | scope=current, operator=latest, freshness_class=slow | entity=Python, property=version, state=live | criterion=latest / temporal | retrieve; latest is a current fact | web.search | successful current version evidence | sufficient | pass | yes | latest Python release with version and date |
| What is HTTP? | subject=HTTP; stable_knowledge=true | unnecessary | scope=stable, operator=none, freshness_class=stable | entity=HTTP, property=definition, state=stable | criterion=definition / objective | skip; stable fact | none | none | sufficient | pass | yes | canonical definition from model knowledge |
| What is the capital of France? | subject=France; stable_knowledge=true | unnecessary | scope=stable, operator=none, freshness_class=stable | entity=France, property=capital, state=stable | criterion=capital / objective | skip | none | none | sufficient | pass | yes | Paris |
| What is 17 × 23? | subject=math expression; local computation | unnecessary | scope=stable, operator=none, freshness_class=stable | entity=math operation, property=result, state=final | criterion=result / objective | skip | none | none | sufficient | pass | yes | 391 |
| Who won the 2023 Super Bowl? | subject=Super Bowl; historical_knowledge=true; final_event_result=true | required | scope=historical, operator=at_time(2023), freshness_class=historical | entity=2023 Super Bowl, property=winner, state=historical | criterion=winner / objective | retrieve only if precise historical evidence is needed; freshness exempt | web.search | completed result evidence | sufficient | pass | yes | Kansas City Chiefs 38-35 |
| Who is the current president of the United States? | subject=president of the United States; current_knowledge=true; dynamic_quantity=true | required | scope=current, operator=current, freshness_class=fast | entity=US president, property=officeholder, state=live | criterion=current officeholder / objective | retrieve | web.search | stale or absent evidence not checked for freshness | insufficient (stale) | block | no | not answered without fresh evidence |
| Turn 1: Who won the 2023 Super Bowl? Turn 2: How did they win? | turn 1 claim persisted; turn 2 references prior answer; prior answer must be re-grounded if not verified | required on turn 2 if claim depends on prior factual statement | scope=historical, operator=at_time(2023) | entity=2023 Super Bowl, property=winning play-by-play, state=historical | criterion=how / objective but depends on verified event result | retrieve or re-ground from evidence; do not trust prior unverified claim | conversation memory + web.search | prior claim without evidence becomes unverified | insufficient (unverified claim) | block | no | no direct answer from unverified memory |

## Root-cause summary

### A. Evidence requirement not detected
- Failure location: `reasoning/semantic_understanding.py` and `reasoning/evidence_policy.py`
- Why current code misses it: the semantic layer only checked `comparative`, `freshness_requirement == current`, and a few shape heuristics; it did not expose `latest_request`, `most_recent_request`, `current_knowledge`, `ranking`, `dynamic_quantity`, `final_event_result`, and similar deterministic semantic flags in a single place.
- Minimal fix location: extend `SemanticRequest` and the structural reading, then make `EvidencePolicy.decide()` re-evaluate `requirement` via these semantic flags rather than brittle string matching.

### B. Evidence sufficiency not assessed
- Failure location: `reasoning/evidence_policy.py`
- Why current code misses it: `evaluate()` only counted sources and checked direct evidence count; it did not classify `intermediate_state`, `pre_event_evidence`, `wrong_property`, `wrong_granularity`, `stale`, or `ambiguous_source`.
- Minimal fix location: add `EvidenceAssessment` and the `assess()` method in `EvidencePolicy`, then route through `EvidenceGate` before generation.

### C. Temporal / freshness not validated
- Failure location: `evidence_models.py` and `reasoning/evidence_policy.py`
- Why current code misses it: `EvidenceItem` had only a timestamp but not `observed_at`, `published_at`, `valid_until`, `state`, and `freshness_class`, so freshness-aware policy could not gate stale or intermediate evidence.
- Minimal fix location: enrich evidence metadata and policy checks for current vs historical vs stable facts, with an explicit disclosure path on temporal gaps.

### D. Conversational claims trusted blindly
- Failure location: `memory/models.py` and memory/context pipeline
- Why current code misses it: the conversation record stored content, citations, and metadata, but not `ConversationalClaim` status or evidence linkage; follow-ups could reuse factual statements without re-grounding.
- Minimal fix location: extend `MemoryMessage` with `claims`, add `ConversationalClaim`, and make follow-up processing require re-grounding when a prior assertion is marked `unverified`.

## Correction priority

1. Adjust EvidencePolicy / EvidenceAssessment logic.
2. Extend semantic representation with new fields.
3. Extend conversational claim schema.

## Validation notes

The live pipeline was exercised via the existing semantic reasoning tests and the targeted evidence gate suite. The implementation passes the focused regression suite without changing Atlas's intent engine or adding new keyword tables.
