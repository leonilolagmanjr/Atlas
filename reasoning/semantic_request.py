"""Open-ended semantic request representation.

This is the *understanding* layer's data model. It replaces "which one of N
predefined intents is this?" with a set of **descriptive dimensions** that any
natural-language goal can be described by, without inventing a new intent for
every phrasing:

    SemanticRequest(
        goal=...,                    # what the user is trying to accomplish
        subject=...,                 # what the request is about
        entities=[...],              # resolved named entities
        operation=...,               # explain, rank, transform, act, ...
        constraints=[...],           # limitations or requirements
        requested_output=...,        # desired form of the result
        context_dependencies=[...],  # prior turns / referents
        freshness_requirement=...,   # current / stable / any
        evidence_requirement=...,    # required / preferred / unnecessary
        ambiguity=...,               # low / medium / high
        confidence=...,              # interpreter confidence
    )

Design rules (they matter):

* **Descriptive, not taxonomic.** ``operation`` and ``goal`` are free text that
  describes the request; they are never validated against a closed enum of user
  intents. Existing enums survive only as *internal execution categories*
  downstream of capability planning.
* **No keyword routing.** Nothing in this module decides behaviour from a magic
  word list. The deterministic fallback reasons over structure (a comparative
  question, a superlative criterion, an anaphoric subject) rather than over an
  enumerated phrase table. Structured hint fields exist so a *model's* structured
  reading can be normalized; they are not a rule engine.
* **Serializable and inspectable.** The whole reading round-trips through
  ``to_dict`` / ``from_mapping`` so it can be recorded on the Task IR, traced in
  diagnostics, and asserted on in tests.
* **Additive.** Nothing here executes tools or mutates state.

The module deliberately has no imports from the planner, executor, or tool
registry: understanding must be able to run before any capability routing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: Freshness of information the request depends on.
#: ``current``  - the answer can change over time (must be verified live).
#: ``stable``   - the answer is established knowledge (safe from the model).
#: ``any``      - freshness does not matter for this request.
FRESHNESS_CURRENT = "current"
FRESHNESS_STABLE = "stable"
FRESHNESS_ANY = "any"
FRESHNESS_LEVELS: frozenset[str] = frozenset(
    {FRESHNESS_CURRENT, FRESHNESS_STABLE, FRESHNESS_ANY}
)

#: Whether external evidence is needed to answer reliably.
#: ``required``    - a plausible-sounding answer would be fabrication without it.
#: ``preferred``   - evidence materially improves reliability.
#: ``unnecessary`` - the model can answer on its own.
EVIDENCE_REQUIRED = "required"
EVIDENCE_PREFERRED = "preferred"
EVIDENCE_UNNECESSARY = "unnecessary"
EVIDENCE_LEVELS: frozenset[str] = frozenset(
    {EVIDENCE_REQUIRED, EVIDENCE_PREFERRED, EVIDENCE_UNNECESSARY}
)

AMBIGUITY_LOW = "low"
AMBIGUITY_MEDIUM = "medium"
AMBIGUITY_HIGH = "high"
AMBIGUITY_LEVELS: frozenset[str] = frozenset(
    {AMBIGUITY_LOW, AMBIGUITY_MEDIUM, AMBIGUITY_HIGH}
)

#: Operations that describe *what the user wants done to the subject*. These are
#: descriptive vocabulary shared by the interpreter and the capability planner so
#: they agree on one word for the same job; an unseen operation is still valid
#: and simply falls back to structural capability inference.
OP_EXPLAIN = "explain"
OP_RETRIEVE = "retrieve"
OP_RANK = "rank"
OP_COMPARE = "compare"
OP_IDENTIFY = "identify"
OP_TRANSFORM = "transform"
OP_CREATE = "create"
OP_ACT = "act"
OP_CONVERSE = "converse"
OP_UNKNOWN = "unknown"


@dataclass
class SemanticRequest:
    """A descriptive, open-ended reading of what the user is trying to accomplish.

    Every field is descriptive: a human reading the structure should be able to
    say whether Atlas understood the request, without knowing Atlas's code.
    """

    #: What the user is trying to accomplish, in plain language.
    goal: str = ""
    #: What the request is about (the subject/target of the goal).
    subject: str = ""
    #: Resolved named entities (free-form; values may be strings or mappings).
    entities: list[Any] = field(default_factory=list)
    #: The operation requested on the subject (explain, rank, transform, act...).
    operation: str = OP_UNKNOWN
    #: Limitations or requirements the user stated (length, tone, format, count...).
    constraints: list[str] = field(default_factory=list)
    #: The desired form of the result (answer, ranking, list, file, app text...).
    requested_output: str = ""
    #: Prior turns / referents this request depends on.
    context_dependencies: list[str] = field(default_factory=list)
    #: current | stable | any
    freshness_requirement: str = FRESHNESS_ANY
    #: required | preferred | unnecessary
    evidence_requirement: str = EVIDENCE_UNNECESSARY
    #: low | medium | high
    ambiguity: str = AMBIGUITY_LOW
    #: Interpreter confidence in this reading (0..1).
    confidence: float = 0.0

    # -- reasoning-support fields (descriptive, not routing keys) ---------------
    #: The criterion a ranking/comparison is judged by ("fame", "subscribers",
    #: "size"). Free text: it is the *user's* word, not a system enum.
    criterion: str = ""
    #: An objective proxy chosen when the criterion is subjective
    #: ("subscriber count" for "most famous"). Empty when none is defensible.
    criterion_proxy: str = ""
    #: True when the request is a comparison/ranking over candidates.
    comparative: bool = False
    #: Candidate set the comparison ranges over ("Minecraft YouTube creators").
    candidate_set: str = ""
    #: True when the request targets the machine/desktop Atlas runs on.
    local: bool = False
    #: True when the request is a follow-up grounded in the conversation.
    contextual: bool = False
    #: True when the subject depends on an earlier turn's referent.
    subject_is_referent: bool = False
    #: Short, inspectable reasons for the reading (never chain-of-thought).
    notes: list[str] = field(default_factory=list)
    #: True when the local model produced (or refined) this reading.
    model_assisted: bool = False

    # -- temporal / evidence-shape extensions -----------------------------------
    #: Stable facts (definitions, geography, historical results) do not usually
    #: require fresh verification while live facts do.
    stable_knowledge: bool = False
    current_knowledge: bool = False
    historical_knowledge: bool = False
    latest_request: bool = False
    most_recent_request: bool = False
    dynamic_quantity: bool = False
    ranking: bool = False
    subjective_criterion: bool = False
    objective_criterion: bool = False
    final_event_result: bool = False

    # -- derived helpers -------------------------------------------------------

    @property
    def needs_external_evidence(self) -> bool:
        """True when Atlas must retrieve before it may answer."""

        return self.evidence_requirement in {EVIDENCE_REQUIRED, EVIDENCE_PREFERRED}

    @property
    def is_current(self) -> bool:
        return self.freshness_requirement == FRESHNESS_CURRENT

    @property
    def is_ambiguous(self) -> bool:
        return self.ambiguity in {AMBIGUITY_MEDIUM, AMBIGUITY_HIGH}

    @property
    def is_action(self) -> bool:
        """True when the request asks Atlas to *do* something, not just answer."""

        return self.operation in {OP_ACT, OP_CREATE, OP_TRANSFORM}

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "subject": self.subject,
            "entities": list(self.entities),
            "operation": self.operation,
            "constraints": list(self.constraints),
            "requested_output": self.requested_output,
            "context_dependencies": list(self.context_dependencies),
            "freshness_requirement": self.freshness_requirement,
            "evidence_requirement": self.evidence_requirement,
            "ambiguity": self.ambiguity,
            "confidence": round(float(self.confidence), 4),
            "criterion": self.criterion,
            "criterion_proxy": self.criterion_proxy,
            "comparative": self.comparative,
            "candidate_set": self.candidate_set,
            "local": self.local,
            "contextual": self.contextual,
            "subject_is_referent": self.subject_is_referent,
            "notes": list(self.notes),
            "model_assisted": self.model_assisted,
            "stable_knowledge": self.stable_knowledge,
            "current_knowledge": self.current_knowledge,
            "historical_knowledge": self.historical_knowledge,
            "latest_request": self.latest_request,
            "most_recent_request": self.most_recent_request,
            "dynamic_quantity": self.dynamic_quantity,
            "ranking": self.ranking,
            "subjective_criterion": self.subjective_criterion,
            "objective_criterion": self.objective_criterion,
            "final_event_result": self.final_event_result,
        }

    @classmethod
    def from_mapping(cls, data: Any) -> "SemanticRequest":
        """Build a reading from untrusted structured output without raising.

        Unknown keys are ignored, wrong types are coerced or dropped, and
        out-of-vocabulary levels fall back to safe defaults. A malformed model
        answer produces an empty reading rather than an exception.
        """

        if not isinstance(data, dict):
            return cls()

        def text(key: str, default: str = "") -> str:
            value = data.get(key)
            if value is None:
                return default
            if isinstance(value, (str, int, float)):
                found = str(value).strip()
                if found and found.casefold() not in {"null", "none", "n/a", "unknown"}:
                    return found
            return default

        def items(key: str) -> list[Any]:
            value = data.get(key)
            if isinstance(value, str):
                found = value.strip()
                return [found] if found else []
            if isinstance(value, (list, tuple)):
                return [item for item in value if item not in (None, "", [])]
            return []

        def flag(key: str, default: bool = False) -> bool:
            value = data.get(key)
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                lowered = value.strip().casefold()
                if lowered in {"true", "yes", "1"}:
                    return True
                if lowered in {"false", "no", "0"}:
                    return False
            return default

        def confidence() -> float:
            try:
                number = float(data.get("confidence"))
            except (TypeError, ValueError):
                return 0.0
            return max(0.0, min(1.0, number))

        def level(key: str, allowed: frozenset[str], default: str) -> str:
            found = text(key).casefold()
            return found if found in allowed else default

        # The model may phrase freshness/evidence as free text; map the common
        # intents onto the vocabulary without a keyword rule engine (a single
        # synonym normalization, never a routing decision).
        return cls(
            goal=text("goal"),
            subject=text("subject"),
            entities=items("entities"),
            operation=text("operation", OP_UNKNOWN).casefold().replace(" ", "_"),
            constraints=[str(item).strip() for item in items("constraints") if str(item).strip()],
            requested_output=text("requested_output"),
            context_dependencies=[
                str(item).strip() for item in items("context_dependencies") if str(item).strip()
            ],
            freshness_requirement=_normalize_freshness(text("freshness_requirement")),
            evidence_requirement=_normalize_evidence(text("evidence_requirement")),
            ambiguity=level("ambiguity", AMBIGUITY_LEVELS, AMBIGUITY_LOW),
            confidence=confidence(),
            criterion=text("criterion"),
            criterion_proxy=text("criterion_proxy"),
            comparative=flag("comparative"),
            candidate_set=text("candidate_set"),
            local=flag("local"),
            contextual=flag("contextual"),
            subject_is_referent=flag("subject_is_referent"),
            notes=[str(item).strip() for item in items("notes") if str(item).strip()],
            model_assisted=flag("model_assisted", True),
            stable_knowledge=flag("stable_knowledge"),
            current_knowledge=flag("current_knowledge"),
            historical_knowledge=flag("historical_knowledge"),
            latest_request=flag("latest_request"),
            most_recent_request=flag("most_recent_request"),
            dynamic_quantity=flag("dynamic_quantity"),
            ranking=flag("ranking"),
            subjective_criterion=flag("subjective_criterion"),
            objective_criterion=flag("objective_criterion"),
            final_event_result=flag("final_event_result"),
        )


#: Small synonym bridges from free phrasing onto the freshness vocabulary. This is
#: *normalization* of one field, not routing: it never selects a capability.
_FRESHNESS_SYNONYMS: dict[str, str] = {
    "live": FRESHNESS_CURRENT, "real-time": FRESHNESS_CURRENT, "now": FRESHNESS_CURRENT,
    "up_to_date": FRESHNESS_CURRENT, "up-to-date": FRESHNESS_CURRENT,
    "recent": FRESHNESS_CURRENT, "time_sensitive": FRESHNESS_CURRENT,
    "timeless": FRESHNESS_STABLE, "static": FRESHNESS_STABLE,
    "evergreen": FRESHNESS_STABLE, "knowledge": FRESHNESS_STABLE,
    "none": FRESHNESS_ANY, "irrelevant": FRESHNESS_ANY, "n/a": FRESHNESS_ANY,
}

_EVIDENCE_SYNONYMS: dict[str, str] = {
    "needed": EVIDENCE_REQUIRED, "necessary": EVIDENCE_REQUIRED,
    "mandatory": EVIDENCE_REQUIRED, "must": EVIDENCE_REQUIRED,
    "strongly_preferred": EVIDENCE_PREFERRED, "helpful": EVIDENCE_PREFERRED,
    "optional": EVIDENCE_UNNECESSARY, "none": EVIDENCE_UNNECESSARY,
    "not_needed": EVIDENCE_UNNECESSARY, "no": EVIDENCE_UNNECESSARY,
}


def _normalize_freshness(value: str) -> str:
    found = (value or "").strip().casefold().replace(" ", "_")
    if found in FRESHNESS_LEVELS:
        return found
    return _FRESHNESS_SYNONYMS.get(found, FRESHNESS_ANY)


def _normalize_evidence(value: str) -> str:
    found = (value or "").strip().casefold().replace(" ", "_")
    if found in EVIDENCE_LEVELS:
        return found
    return _EVIDENCE_SYNONYMS.get(found, EVIDENCE_UNNECESSARY)
