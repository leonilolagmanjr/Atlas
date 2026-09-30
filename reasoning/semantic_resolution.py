"""Shared semantic resolution: entities, destinations, and research queries.

This module is the *one* place where an uncertain natural-language span is
resolved to something concrete. It exists because the same job was previously
implemented three times (a topic-correction list inside topic extraction, an
application-name list inside the interpreter, another inside the intent engine),
and because a typo such as ``sykrim`` could not be corrected at all once the
correct spelling was not already in a hardcoded list.

Design rules:

* **Layered, cheapest-first.** Deterministic exact/known matching runs before
  fuzzy matching; fuzzy matching runs before any evidence lookup; a local model
  is consulted only when everything deterministic failed. Nothing here performs
  I/O of its own: a web lookup or a model call is *injected* by the caller, so
  this module stays usable offline and testable without a network.
* **Never blindly normalize.** A resolution is returned only when it is
  supported (confidence, uniqueness, and a real-word guard). Competing candidates
  of similar strength produce an explicit *ambiguous* result rather than a guess.
* **Raw and resolved are both kept.** Every :class:`Resolution` carries the raw
  text, the resolved text, the method, the confidence, the evidence, and the
  alternatives, so provenance survives into the Task IR.
* **One abstraction, small policies.** :class:`EntityResolver`,
  :class:`DestinationResolver`, and :func:`normalize_research_query` share the
  same primitives (:func:`similarity`, :func:`bounded_edit_distance`,
  :class:`Resolution`). This is a small shared service, not a framework.

Nothing in this module executes tools, launches applications, or mutates state.
"""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

# ---------------------------------------------------------------------------
# Resolution methods (provenance vocabulary)
# ---------------------------------------------------------------------------

#: The candidate was returned unchanged because nothing better was found.
UNCHANGED = "unchanged"
#: The candidate matched a known name exactly.
EXACT_KNOWN = "exact_known_entity"
#: The candidate was close to a local vocabulary entry (corpus similarity).
LOCAL_VOCABULARY = "local_vocabulary"
#: The candidate was a bounded lexical near-miss of a known name.
LEXICAL_FUZZY = "lexical_fuzzy"
#: The candidate was resolved from external evidence (search results).
SEARCH_EVIDENCE = "web_entity_resolution"
#: The candidate was resolved with the local model's structured suggestion.
MODEL_ASSISTED = "model_assisted"
#: The candidate could not be resolved; the raw value is preserved.
UNRESOLVED = "unresolved"
#: Several equally plausible candidates exist; a guess is refused.
AMBIGUOUS = "ambiguous"

#: Real English words that must never be "corrected" to a known name. A real
#: word ("mode" -> "code") is not a typo; rewriting it changes the meaning of the
#: request, which the confidence policy forbids. This is a small
#: damage-prevention *cache*, never the intelligence layer: the actual
#: resolution comes from vocabulary, evidence, or a validated model suggestion.
REAL_WORD_GUARD: frozenset[str] = frozenset(
    {
        "code", "mode", "model", "node", "note", "notes", "word", "words",
        "load", "road", "toad", "made", "make", "rate", "late", "game",
        "games", "name", "names", "time", "times", "line", "lines", "page",
        "pages", "file", "files", "city", "site", "sites", "data", "date",
        "dates", "play", "plan", "plans", "port", "part", "parts", "form",
        "list", "lists", "test", "text", "cost", "case", "care",
        "area", "type", "types", "item", "items", "user", "users", "home",
        "tree", "free", "life", "love", "meme", "term", "terms", "news",
    }
)

#: Sentence scaffolding that never carries an entity on its own.
FUNCTION_WORDS: frozenset[str] = frozenset(
    {
        "the", "a", "an", "and", "or", "of", "for", "to", "in", "into", "on",
        "at", "by", "with", "from", "about", "regarding", "concerning", "my",
        "our", "your", "their", "its", "this", "that", "these", "those", "it",
        "them", "how", "what", "why", "when", "where", "which", "who", "is",
        "are", "was", "were", "be", "do", "does", "did", "can", "could",
        "should", "would", "will", "please", "me", "some", "any", "all",
    }
)


def _clean(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _key(value: object) -> str:
    return _clean(value).casefold()


def _token_text(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", value or "")


def similarity(a: str, b: str) -> float:
    """Deterministic string similarity in ``[0, 1]`` (no model, no network)."""

    left, right = _key(a), _key(b)
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    return difflib.SequenceMatcher(None, left, right).ratio()


def bounded_edit_distance(a: str, b: str, *, limit: int = 1) -> int | None:
    """Levenshtein distance, returning ``None`` once it exceeds ``limit``.

    Treats an adjacent transposition ("skryim" -> "skyrim" when lengths match)
    as a single edit, which is the common real-world typo this must catch.
    """

    if a == b:
        return 0
    if abs(len(a) - len(b)) > limit:
        return None
    # Fast transposition check for equal-length tokens.
    if len(a) == len(b) and limit >= 1:
        diffs = [i for i in range(len(a)) if a[i] != b[i]]
        if len(diffs) == 2 and diffs[1] == diffs[0] + 1:
            i, j = diffs
            if a[i] == b[j] and a[j] == b[i]:
                return 1
    # Bounded edit distance (row-by-row, early exit when a row is all over limit).
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        best = i
        for j, char_b in enumerate(b, start=1):
            cost = 0 if char_a == char_b else 1
            value = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
            current.append(value)
            best = min(best, value)
        if best > limit:
            return None
        previous = current
    distance = previous[-1]
    return distance if distance <= limit else None


#: Methods that count as a genuine change of the user's wording.
CHANGING_METHODS: frozenset[str] = frozenset(
    {EXACT_KNOWN, LOCAL_VOCABULARY, LEXICAL_FUZZY, SEARCH_EVIDENCE, MODEL_ASSISTED}
)

#: Confidence floor for a resolution to be allowed to replace user wording.
DEFAULT_MIN_CONFIDENCE: float = 0.8
#: Two candidates closer than this are treated as a genuine ambiguity.
DEFAULT_CLOSE_CALL: float = 0.12



@dataclass(frozen=True)
class Resolution:
    """The outcome of attempting to resolve one semantic slot."""

    kind: str
    raw: str
    resolved: str = ""
    method: str = UNRESOLVED
    confidence: float = 0.0
    evidence: tuple[str, ...] = ()
    alternatives: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    context: str = ""

    @property
    def changed(self) -> bool:
        """True when the resolved value differs from what the user wrote."""

        return bool(self.resolved) and _key(self.resolved) != _key(self.raw)

    @property
    def resolved_ok(self) -> bool:
        return self.method in CHANGING_METHODS or self.method in {UNCHANGED, EXACT_KNOWN}

    @property
    def ambiguous(self) -> bool:
        return self.method == AMBIGUOUS

    @property
    def unresolved(self) -> bool:
        return self.method == UNRESOLVED

    def value(self, fallback: str = "") -> str:
        """The resolved value when there is one, otherwise the raw value."""

        return self.resolved or self.raw or fallback

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "raw": self.raw,
            "resolved": self.resolved,
            "method": self.method,
            "confidence": round(self.confidence, 3),
            "changed": self.changed,
            "evidence": list(self.evidence),
            "alternatives": list(self.alternatives),
            "notes": list(self.notes),
            "context": self.context,
        }


#: Title/path noise that is never an entity name on its own.
_EVIDENCE_NOISE: frozenset[str] = frozenset(
    {
        "home", "wiki", "wikipedia", "official", "site", "website", "news",
        "the", "and", "for", "with", "from", "about", "search", "results",
        "video", "videos", "youtube", "watch", "page", "pages", "index",
        "com", "org", "net", "www", "http", "https", "html", "htm", "reddit",
        "blog", "article", "articles", "review", "reviews", "download",
    }
)


def candidate_spans(text: str, *, max_words: int = 4) -> list[str]:
    """Extract entity-looking spans from free text (a title, snippet, or URL).

    Used to turn *search evidence* into candidates: a result titled
    "Skyrim - The Elder Scrolls V Wiki" contributes "Skyrim" and "The Elder
    Scrolls V Wiki" as possible spellings of what the user mistyped. This is
    purely lexical; the resolver decides which candidate is credible.
    """

    spans: list[str] = []
    cleaned = re.sub(r"https?://\S+", " ", str(text or ""))
    for chunk in re.split(r"[|•—–\[\](){}<>\"'/\\,:;!?]+|\s+-\s+", cleaned):
        words = [
            word
            for word in re.findall(r"[A-Za-z][A-Za-z0-9'’.-]*", chunk)
            if _key(word) not in _EVIDENCE_NOISE
        ]
        for size in range(1, max_words + 1):
            for start in range(0, max(0, len(words) - size + 1)):
                span = " ".join(words[start:start + size]).strip(" .-")
                if span and len(span) >= 3:
                    spans.append(span)
    return list(dict.fromkeys(spans))


def evidence_candidates(evidence: Iterable[str], *, max_words: int = 4) -> list[str]:
    """Expand raw evidence strings (titles/snippets/URLs) into candidate spans."""

    candidates: list[str] = []
    for item in evidence:
        text = str(item or "").strip()
        if not text:
            continue
        candidates.append(text)
        candidates.extend(candidate_spans(text, max_words=max_words))
    return list(dict.fromkeys(candidates))

def _canonical_map(names: Iterable[str]) -> dict[str, str]:
    """Map a casefolded name to its canonical spelling (first spelling wins)."""

    mapping: dict[str, str] = {}
    for name in names:
        text = _clean(name)
        if text:
            mapping.setdefault(text.casefold(), text)
    return mapping


def _looks_like_proper_name(token: str) -> bool:
    return bool(token[:1].isupper()) and not token.isupper()


class EntityResolver:
    """Resolve an uncertain entity span to a supported spelling or reading.

    Layer order (cheapest first, and the order matters for trust):

    ``exact known`` -> ``local vocabulary`` -> ``lexical fuzzy`` ->
    ``web evidence`` -> ``model assisted`` -> ``unresolved``.

    ``suggest`` (optional) is an injected evidence lookup: it receives the raw
    term and returns raw evidence strings (search titles/snippets/URLs). It may
    perform network I/O; nothing else here does.
    """

    def __init__(
        self,
        *,
        vocabulary: Iterable[str] = (),
        suggest: Callable[[str], Sequence[str]] | None = None,
        ask: Callable[..., str] | None = None,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        close_call: float = DEFAULT_CLOSE_CALL,
        allow_model: bool = True,
    ) -> None:
        self._vocabulary = _canonical_map(vocabulary)
        self._suggest = suggest
        self._ask = ask
        self._min_confidence = float(min_confidence)
        self._close_call = float(close_call)
        self._allow_model = bool(allow_model)

    # -- introspection -----------------------------------------------------------

    @property
    def vocabulary(self) -> tuple[str, ...]:
        return tuple(self._vocabulary.values())

    def has(self, name: str) -> bool:
        return _key(name) in self._vocabulary

    def exact(self, name: str) -> str | None:
        return self._vocabulary.get(_key(name))

    def with_vocabulary(self, extra: Iterable[str]) -> "EntityResolver":
        """Return a resolver extended with more known names (bounded change)."""

        return EntityResolver(
            vocabulary=list(self._vocabulary.values()) + [str(item) for item in extra if item],
            suggest=self._suggest,
            ask=self._ask,
            min_confidence=self._min_confidence,
            close_call=self._close_call,
            allow_model=self._allow_model,
        )

    # -- resolution --------------------------------------------------------------

    def resolve(
        self,
        candidate: str,
        *,
        context: str = "",
        evidence: Iterable[str] = (),
        kind: str = "entity",
    ) -> Resolution:
        """Resolve one span, trying the cheapest supported method first."""

        raw = _clean(candidate)
        if not raw:
            return Resolution(kind=kind, raw="", method=UNRESOLVED, context=context)

        exact = self._vocabulary.get(_key(raw))
        if exact:
            return Resolution(
                kind=kind, raw=raw, resolved=exact, method=EXACT_KNOWN,
                confidence=0.98, context=context, notes=("known name",),
            )

        fuzzy = self._lexical(raw, kind=kind, context=context)
        if fuzzy is not None:
            return fuzzy

        evidence_candidates_list = list(evidence_candidates(evidence)) if evidence else []
        from_evidence = self._from_evidence(raw, evidence_candidates_list, kind=kind, context=context)
        if from_evidence is not None:
            return from_evidence

        if self._suggest is not None:
            suggested = self._suggest(raw) or ()
            suggested_candidates = list(evidence_candidates([str(item) for item in suggested]))
            from_search = self._from_evidence(raw, suggested_candidates, kind=kind, context=context)
            if from_search is not None:
                return from_search

        if self._allow_model and self._ask is not None:
            from_model = self._from_model(raw, kind=kind, context=context)
            if from_model is not None:
                return from_model

        return Resolution(
            kind=kind, raw=raw, resolved=raw, method=UNRESOLVED, confidence=0.0,
            context=context, alternatives=tuple(tuple(self._alternatives(raw))[:3]),
            notes=("no supported candidate found",),
        )

    # -- subject-level resolution ------------------------------------------------

    def resolve_subject(
        self,
        subject: str,
        *,
        context: str = "",
        evidence: Iterable[str] = (),
        kind: str = "topic",
    ) -> Resolution:
        """Resolve a whole subject span, token by token, without a phrase table.

        Compound subjects stay whole ("history of Skyrim", "how Skyrim's
        leveling system works"): only the qualifying token is corrected, and the
        user's structure is preserved. A genuine multi-sense ambiguity that the
        evidence could not settle is reported instead of guessed.
        """

        raw = _clean(subject)
        tokens = raw.split()
        if not tokens:
            return Resolution(kind=kind, raw="", method=UNRESOLVED, context=context)

        readings = [
            self.resolve(token, context=context, evidence=evidence, kind=kind)
            for token in tokens
        ]
        changed = [reading for reading in readings if reading.changed]
        ambiguous = [reading for reading in readings if reading.ambiguous]

        if ambiguous and not changed:
            best = max(ambiguous, key=lambda item: item.confidence)
            return Resolution(
                kind=kind, raw=raw, resolved=raw, method=AMBIGUOUS,
                confidence=best.confidence, context=context,
                alternatives=best.alternatives, notes=best.notes,
            )
        if not changed:
            return Resolution(
                kind=kind, raw=raw, resolved=raw, method=UNCHANGED,
                confidence=1.0, context=context, notes=("subject preserved",),
            )

        resolved = " ".join(reading.value(token) for reading, token in zip(readings, tokens))
        weakest = min(reading.confidence for reading in changed)
        strongest = max(changed, key=lambda item: item.confidence)
        return Resolution(
            kind=kind, raw=raw, resolved=resolved, method=strongest.method,
            confidence=round(weakest, 3), context=context,
            evidence=strongest.evidence,
            notes=tuple(
                note for reading in changed for note in (reading.notes or (f"normalized '{reading.raw}'",))
            ),
        )

    # -- layers ------------------------------------------------------------------

    def _alternatives(self, raw: str) -> list[str]:
        word = _token_text(raw).casefold()
        if not word:
            return []
        found: list[str] = []
        for candidate in self._vocabulary.values():
            candidate_word = _token_text(candidate).casefold()
            if candidate_word and candidate_word != word and abs(len(candidate_word) - len(word)) <= 1:
                if bounded_edit_distance(word, candidate_word, limit=1) == 1:
                    found.append(candidate)
        return found

    def _corpus(self, raw: str, *, kind: str, context: str) -> Resolution | None:
        """Local vocabulary / corpus similarity (cheap, no external evidence)."""

        word = _token_text(raw).casefold()
        if len(word) < 4 or word in REAL_WORD_GUARD or word in FUNCTION_WORDS:
            return None
        scored: list[tuple[float, str]] = []
        for candidate in self._vocabulary.values():
            candidate_word = _token_text(candidate).casefold()
            if not candidate_word or candidate_word == word:
                continue
            if abs(len(candidate_word) - len(word)) > 1:
                continue
            ratio = similarity(word, candidate_word)
            if ratio >= 0.88:
                scored.append((ratio, candidate))
        if not scored:
            return None
        scored.sort(key=lambda item: (-item[0], item[1]))
        top_ratio, top = scored[0]
        if len(scored) > 1 and scored[1][0] >= top_ratio - 0.02:
            return Resolution(
                kind=kind, raw=raw, resolved=raw, method=AMBIGUOUS, confidence=0.5,
                context=context, alternatives=tuple(item[1] for item in scored[:3]),
                notes=("corpus similarity is split between several names",),
            )
        return Resolution(
            kind=kind, raw=raw, resolved=top, method=LOCAL_VOCABULARY,
            confidence=0.82, context=context,
            notes=(f"local corpus match for '{raw}'",),
        )

    def _lexical(self, raw: str, *, kind: str, context: str) -> Resolution | None:
        """Bounded lexical near-miss against known names (no dictionary needed)."""

        word = _token_text(raw)
        if len(word) < 4 or len(word) > 20:
            return None
        lowered = word.casefold()
        if lowered in REAL_WORD_GUARD or lowered in FUNCTION_WORDS:
            # A real English word is not a typo. Evidence or the model may still
            # resolve it later; fuzzy matching alone may not.
            return None
        corpus = self._corpus(raw, kind=kind, context=context)
        if corpus is not None:
            return corpus
        if not _looks_like_proper_name(raw) and len(word) < 5:
            return None
        matches: list[str] = []
        for candidate in self._vocabulary.values():
            candidate_word = _token_text(candidate).casefold()
            if not candidate_word or candidate_word == lowered:
                continue
            if candidate_word in REAL_WORD_GUARD:
                continue
            if abs(len(candidate_word) - len(lowered)) > 1:
                continue
            if bounded_edit_distance(lowered, candidate_word, limit=1) == 1:
                matches.append(candidate)
        if not matches:
            return None
        if len(matches) > 1:
            return Resolution(
                kind=kind, raw=raw, resolved=raw, method=AMBIGUOUS, confidence=0.5,
                context=context, alternatives=tuple(matches[:3]),
                notes=("several known names are one edit away",),
            )
        return Resolution(
            kind=kind, raw=raw, resolved=matches[0], method=LEXICAL_FUZZY,
            confidence=0.87, context=context,
            notes=(f"corrected '{raw}' -> '{matches[0]}'",),
        )

    def _from_evidence(
        self,
        raw: str,
        candidates: Sequence[str],
        *,
        kind: str,
        context: str,
    ) -> Resolution | None:
        """Resolve from external evidence (search titles, snippets, URLs).

        Two distinct signals are read out of the same evidence:

        1. **A spelling the user mistyped.** "sykrim" appears nowhere, but
           "Skyrim" appears in result titles one or two edits away, so the
           evidence itself supplies the correction. No dictionary is required —
           this is what makes ``sykrim -> Skyrim`` work for an entity Atlas has
           never seen before.
        2. **Genuinely competing senses.** When the subject is already spelled
           correctly but the evidence attaches *several* qualified names to it
           ("Jaguar Cars", "Jaguar animal"), that is ambiguity, not a typo, so
           the resolver refuses to choose.
        """

        raw_token = _token_text(raw)
        raw_key = raw_token.casefold()
        if not raw_key:
            return None
        limit = 2 if len(raw_key) >= 6 else 1
        guard = raw_key in REAL_WORD_GUARD or raw_key in FUNCTION_WORDS

        scored: list[tuple[float, str, int]] = []
        frequencies: dict[str, int] = {}
        senses: list[str] = []
        for candidate in candidates:
            candidate_text = _clean(candidate)
            candidate_token = _token_text(candidate_text)
            if not candidate_token:
                continue
            candidate_key = candidate_token.casefold()
            frequencies[candidate_key] = frequencies.get(candidate_key, 0) + 1
            # A qualified reading of the *same* word ("Jaguar Cars" for "jaguar")
            # is a sense, never a spelling fix.
            if " " in candidate_text and re.match(
                rf"^{re.escape(raw_token)}\b", candidate_text, re.IGNORECASE
            ):
                candidate_text = candidate_text.strip(" .,-—")
                if candidate_text and candidate_text.casefold() not in {s.casefold() for s in senses}:
                    senses.append(candidate_text)
                continue
            if guard or " " in candidate_token or candidate_key == raw_key:
                continue
            if abs(len(candidate_key) - len(raw_key)) > limit:
                continue
            distance = bounded_edit_distance(raw_key, candidate_key, limit=limit)
            if not distance:
                continue
            ratio = similarity(raw_key, candidate_key)
            if ratio < 0.55:
                continue
            score = 0.72 + 0.24 * ratio - 0.06 * (distance - 1)
            scored.append((score, candidate_token, distance))

        if scored:
            best_score = max(item[0] for item in scored)
            top = [item for item in scored if item[0] >= best_score - 0.001]
            rivals = [
                item for item in scored
                if item not in top and item[0] >= best_score - self._close_call
            ]
            if rivals:
                return Resolution(
                    kind=kind, raw=raw, resolved=raw, method=AMBIGUOUS, confidence=0.5,
                    context=context,
                    alternatives=tuple(dict.fromkeys([item[1] for item in (top + rivals)][:3])),
                    notes=("evidence supports several readings equally",),
                )
            resolved = top[0][1]
            confidence = min(0.96, best_score + 0.02 * (frequencies.get(resolved.casefold(), 1) - 1))
            if confidence < self._min_confidence:
                return None
            return Resolution(
                kind=kind, raw=raw, resolved=resolved, method=SEARCH_EVIDENCE,
                confidence=round(confidence, 3), context=context,
                evidence=tuple(senses[:3]),
                notes=(f"evidence suggests '{resolved}' for '{raw}'",),
            )

        if len(senses) >= 2:
            # The subject is spelled the way the user wrote it, but the web knows
            # several different things by that name. Surface it; do not guess.
            return Resolution(
                kind=kind, raw=raw, resolved=raw, method=AMBIGUOUS, confidence=0.5,
                context=context, alternatives=tuple(senses[:3]),
                notes=("the request may refer to more than one thing",),
            )
        return None

    def _from_model(self, raw: str, *, kind: str, context: str) -> Resolution | None:
        """Use the local model only as a validated *suggestion* (never blindly)."""

        data = _safe_model_call(
            system_prompt_kind=kind,
            candidate=raw,
            context=context,
            ask=self._ask,
        )
        if not data:
            return None
        candidate = _clean(data.get("candidate"))
        if not candidate or _key(candidate) == _key(raw):
            return None
        try:
            confidence = float(data.get("confidence"))
        except (TypeError, ValueError):
            return None
        if confidence < self._min_confidence:
            return None
        alternatives: list[str] = []
        raw_alternatives = data.get("alternatives")
        if isinstance(raw_alternatives, (list, tuple)):
            alternatives = [_clean(item) for item in raw_alternatives if _clean(item)]
        reason = _clean(data.get("reason"))
        return Resolution(
            kind=kind, raw=raw, resolved=candidate, method=MODEL_ASSISTED,
            confidence=min(0.97, max(self._min_confidence, confidence)), context=context,
            alternatives=tuple(alternatives[:3]),
            notes=(reason or "model suggestion",),
        )


def _safe_model_call(
    *,
    system_prompt_kind: str,
    candidate: str,
    context: str,
    ask: Callable[..., str] | None,
) -> dict[str, Any]:
    """Call the model layer defensively; any failure is simply a non-resolution.

    The model is an *interpretation* component: it may only propose a name. It
    cannot execute tools, and a malformed or over-confident answer is discarded.
    """

    if ask is None:
        return {}
    try:
        from reasoning.json_llm import safe_reasoning_call
        from reasoning.prompts import ENTITY_RESOLUTION_SYSTEM, entity_resolution_user_prompt

        data = safe_reasoning_call(
            system_prompt=ENTITY_RESOLUTION_SYSTEM,
            user_prompt=entity_resolution_user_prompt(
                candidate=candidate, context=context, kind=system_prompt_kind
            ),
            ask=ask,
        )
    except Exception:  # pragma: no cover - defensive: a model failure is not fatal
        return {}
    return data if isinstance(data, dict) else {}


class DestinationResolver(EntityResolver):
    """Resolve where a result should go (an application, or unresolved).

    A destination is a *semantic role*, not a keyword: any phrase that names a
    place to put a result is a candidate, and it is validated against the
    application registry (:func:`atlas.application_names`) rather than against a
    growing list of sentence patterns. An unknown destination is reported as
    unresolved instead of being invented.
    """

    def resolve_destination(
        self,
        candidate: str,
        *,
        context: str = "",
        evidence: Iterable[str] = (),
    ) -> Resolution:
        resolution = self.resolve(candidate, context=context, evidence=evidence, kind="destination")
        if resolution.method in {UNCHANGED, EXACT_KNOWN} and resolution.resolved:
            # The user's own spelling of a registered name is authoritative.
            return resolution
        if resolution.changed and resolution.confidence >= self._min_confidence:
            return resolution
        if resolution.ambiguous:
            return resolution
        return Resolution(
            kind="destination", raw=resolution.raw, resolved="", method=UNRESOLVED,
            confidence=0.0, context=context,
            alternatives=resolution.alternatives,
            notes=("destination is not a registered application",),
        )



# ---------------------------------------------------------------------------
# Shared registries (single source of truth)
# ---------------------------------------------------------------------------

_APPLICATION_NAMES: tuple[str, ...] = ()
_WEB_PLATFORMS: tuple[str, ...] = (
    "youtube", "google", "bing", "the web", "the internet", "online", "wikipedia",
)


def application_names() -> tuple[str, ...]:
    """Friendly names of applications Atlas can actually drive.

    The registry lives beside the launch capability (``computer.launch``), so a
    destination is validated against what Atlas can really resolve instead of
    against a second, drifting list inside the interpreter.
    """

    global _APPLICATION_NAMES
    if not _APPLICATION_NAMES:
        names: list[str] = []
        try:
            from computer.launch import known_application_names

            names.extend(known_application_names())
        except Exception:  # pragma: no cover - a missing registry is not fatal
            names = []
        _APPLICATION_NAMES = tuple(dict.fromkeys(names))
    return _APPLICATION_NAMES


def web_platforms() -> tuple[str, ...]:
    """Web platforms a request can name as a source rather than a destination."""

    return _WEB_PLATFORMS


def destination_resolver(*, extra_vocabulary: Iterable[str] = ()) -> DestinationResolver:
    """A resolver whose vocabulary is the live application registry."""

    return DestinationResolver(vocabulary=list(application_names()) + [str(n) for n in extra_vocabulary if n])


# ---------------------------------------------------------------------------
# Research-query normalization (one path for every research caller)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NormalizedQuery:
    """The outcome of normalizing a research query."""

    raw: str
    query: str
    resolution: Resolution
    notes: tuple[str, ...] = ()
    candidates: tuple[str, ...] = ()

    @property
    def changed(self) -> bool:
        return _key(self.query) != _key(self.raw)

    @property
    def ambiguous(self) -> bool:
        return self.resolution.ambiguous

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw": self.raw,
            "query": self.query,
            "changed": self.changed,
            "resolution": self.resolution.to_dict(),
            "notes": list(self.notes),
            "candidates": list(self.candidates),
        }


def normalize_research_query(
    query: str,
    *,
    resolver: EntityResolver | None = None,
    vocabulary: Iterable[str] = (),
    evidence: Iterable[str] = (),
    context: str = "",
) -> NormalizedQuery:
    """Normalize a research query through the shared semantic pipeline.

    Both research paths (the reasoning engine's iterative retrieval and the
    ``web.research`` tool) call this, so a query is cleaned and its entities
    resolved the same way no matter which path started the retrieval:

        raw query -> instruction-language separation -> entity resolution -> query

    The raw and resolved values are both returned, because the retrieval layer
    needs to know what the user actually wrote.
    """

    raw = _clean(query)
    if not raw:
        return NormalizedQuery(
            raw="", query="", resolution=Resolution(kind="query", raw="", method=UNRESOLVED)
        )

    # Instruction language ("about", "information about") is separated from the
    # subject by the canonical topic extractor. Imported lazily: the topic
    # extractor imports this module's primitives.
    from reasoning.topic_extraction import extract_topic

    notes: list[str] = []
    cleaned = raw
    try:
        reading = extract_topic(
            raw,
            known_entities=list(resolver.vocabulary) if resolver else list(vocabulary),
        )
    except Exception:  # pragma: no cover - the normalizer must never fail a search
        reading = None
    if reading is not None and reading.subject:
        cleaned = reading.subject
        if _key(cleaned) != _key(raw):
            notes.append(f"stripped instruction language from '{raw}'")

    active = resolver or EntityResolver(vocabulary=list(vocabulary))
    resolution = active.resolve_subject(cleaned, context=context or raw, evidence=evidence, kind="query")
    resolved = resolution.value(cleaned) or cleaned
    if resolution.changed:
        notes.append(f"resolved '{cleaned}' -> '{resolved}'")
    if resolution.ambiguous:
        notes.append("the topic may refer to more than one thing")
    return NormalizedQuery(
        raw=raw,
        query=resolved,
        resolution=resolution,
        notes=tuple(notes),
        candidates=resolution.alternatives,
    )
