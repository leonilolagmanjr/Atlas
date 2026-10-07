"""Source selection: choose the ordered information sources for a request.

The selector turns :class:`~reasoning.query_router.RoutingSignals` plus the
Task IR into an explicit, ordered :class:`~reasoning.reasoning_models.SourcePlan`.

Rules that matter:

* Semantic intent dominates lexical signals. A file request that a naive
  search-verb reading would send to the web stays local when the file domain is
  what the user actually referred to.
* A source is only selected when its capability is actually registered, so a
  plan can never reference a tool this runtime does not have.
* Freshness drives the web: requests about current information consult the web
  before the model, and always fall back to the model if retrieval fails.
"""

from __future__ import annotations

from reasoning.query_router import RoutingSignals
from reasoning.reasoning_models import (
    ConfidenceLevel,
    RequestType,
    ResponseMode,
    SourcePlan,
    SourceType,
)
from tools.capabilities import CapabilityRegistry

#: Capabilities that mutate local state.
_MUTATING_PREFIXES: tuple[str, ...] = (
    "filesystem.write",
    "filesystem.create_folder",
    "filesystem.move",
    "filesystem.copy",
    "applications.launch",
    "applications.write_text",
    "powershell.execute",
    "processes.",
)


class SourceSelector:
    """Select and order sources deterministically from routing signals."""

    def __init__(self, capabilities: CapabilityRegistry) -> None:
        self._capabilities = capabilities

    # -- public API ---------------------------------------------------------------

    def select(self, signals: RoutingSignals, *, task_type: str = "unknown") -> SourcePlan:
        """Return the ordered source plan for a request."""

        del task_type  # the Task IR already encodes task_type; kept for callers
        if signals.needs_clarification:
            return self._plan((), "the interpreted request needs clarification")

        if signals.has_actions and signals.wants_mutation:
            return self._action_plan(signals)

        # An unresolved reference ("find that file", "make it better") must not be
        # silently resolved into a confident source. When the interpreter's only
        # proposal is a weak web default from a bare search verb, ask instead.
        if signals.ambiguous_reference and not signals.explicit_web_request and not self._resolved_reference(signals):
            return self._plan((), "no resolvable target; clarification required before any source")

        # The semantic reasoning layer's evidence requirement is the primary
        # routing input. A request that clearly needs evidence (a ranking, a
        # current-information question, or an information request about an
        # external entity) consults the web *whether or not* it named a search
        # verb, so semantically equivalent requests reach the same strategy.
        semantic_plan = self._semantic_evidence_plan(signals)
        if semantic_plan is not None:
            return semantic_plan

        if signals.sources:
            allowed = {source.value: source for source in SourceType}
            if all(source in allowed for source in signals.sources):
                sources = tuple(dict.fromkeys(allowed[source] for source in signals.sources))
                if signals.time_sensitive and not any(
                    source in sources for source in (SourceType.FILES, SourceType.SYSTEM, SourceType.CONVERSATION, SourceType.MEMORY)
                ):
                    sources = (SourceType.WEB,) + tuple(source for source in sources if source is not SourceType.WEB)
                return self._plan(
                    sources,
                    "sources proposed by the semantic Task interpreter",
                    current_information_required=signals.time_sensitive,
                    requires_action=SourceType.COMPUTER in sources,
                )

        if signals.is_self_query:
            return self._plan(
                (SourceType.SELF,),
                "the request is about Atlas itself; answered from the capability registry",
            )

        if signals.ambiguous_reference:
            # No referent anywhere: the only correct next step is clarification.
            return self._plan((), "no resolvable target; clarification required before any source")

        if signals.is_memory_query:
            return self._plan(
                (SourceType.CONVERSATION, SourceType.MEMORY),
                "the request refers to earlier turns; answered from conversation memory",
            )

        # A contextual follow-up that names no new target is answered *grounded in
        # the conversation*: the retrieved turns are the evidence, so the answer is
        # a continuation of the discussion rather than a fresh knowledge lookup.
        # An executable contextual request never reaches here (it carries actions
        # and is handled by the action plan above).
        if signals.contextual:
            return self._plan(
                (SourceType.CONVERSATION, SourceType.MODEL),
                "the message continues the conversation; it is grounded in the earlier turns",
            )

        if signals.file_intent is not None and self._file_intent_dominates(signals):
            return self._file_plan(signals)

        if signals.has_actions or (signals.wants_mutation and not signals.is_question):
            return self._action_plan(signals)

        if signals.system_reference:
            return self._system_plan(signals)

        if signals.time_sensitive or signals.explicit_web_request:
            return self._web_plan(signals)

        # An ordinary question: ground it in local documents when possible, then
        # answer from the model's own knowledge instead of reporting failure.
        return self._plan(
            (SourceType.KNOWLEDGE, SourceType.MODEL),
            "ordinary question: local knowledge first, then model knowledge",
        )

    # -- plans --------------------------------------------------------------------

    def _semantic_evidence_plan(self, signals: RoutingSignals) -> SourcePlan | None:
        """Route from the semantic evidence requirement, or return None to defer.

        A *required* evidence request (a ranking/comparison, or current
        information) must consult the web before the model: a plausible-looking
        answer that was not retrieved is a fabrication. A *preferred* evidence
        request consults the web too, because for an external entity a grounded
        answer is materially more reliable - but the plan keeps the model as the
        fallback so a retrieval failure still produces an honest answer.

        This defers (returns None) for requests whose source is genuinely local
        (files, the machine, the conversation, Atlas itself) so a semantic
        evidence flag can never drag a local request onto the web. A request that
        merely *contains* a second-person word ("what do you know about MrBeast")
        still names an external subject, so a strict self-query read does not
        suppress its evidence requirement.
        """

        requirement = signals.evidence_requirement
        if requirement not in {"required", "preferred"}:
            return None
        if not self._available("web.search"):
            return None
        # Local domains keep their own routing, but a self-query deferral only
        # holds when the request has *no* external subject of its own.
        if signals.system_reference or signals.is_memory_query:
            return None
        if signals.is_self_query and not self._names_external_subject(signals):
            return None
        if signals.file_intent is not None and self._file_intent_dominates(signals):
            return None
        # A contextual follow-up with no subject of its own is grounded in the
        # conversation; but a follow-up that *does* name an external subject must
        # still consult evidence, so only defer when the request carries no
        # subject to look up.
        if signals.contextual and not signals.comparative and not signals.sources \
                and not self._names_external_subject(signals):
            return None
        # An explicit local file target ("find my resume") is never a web search.
        if signals.has_actions and not signals.explicit_web_request:
            planned = {source for source in signals.sources}
            if planned and planned <= {"files", "computer", "system"}:
                return None

        reasons: list[str] = []
        if signals.comparative:
            reasons.append(
                "the request is a ranking/comparison and must be answered from "
                "retrieved candidates, not generated"
            )
        if requirement == "required":
            reasons.append("the semantic reading judged external evidence required")
            return self._plan(
                (SourceType.WEB, SourceType.MODEL),
                "; ".join(reasons),
                capabilities=tuple(
                    name for name in ("web.search", "web.fetch") if self._available(name)
                ),
                current_information_required=True,
            )
        # "preferred" evidence: the web is consulted for a grounded answer, but
        # the request does NOT mandate live current information. The plan keeps
        # the model as the fallback so a retrieval failure still produces an honest
        # answer rather than a fabricated one.
        reasons.append(
            "the request concerns an external entity; grounded evidence is more reliable"
        )
        return self._plan(
            (SourceType.WEB, SourceType.MODEL),
            "; ".join(reasons),
            capabilities=tuple(
                name for name in ("web.search", "web.fetch") if self._available(name)
            ),
            current_information_required=bool(signals.time_sensitive),
        )

    @staticmethod
    def _names_external_subject(signals: RoutingSignals) -> bool:
        """True when the semantic reading carries a concrete, non-self subject.

        A self-query is only *really* about Atlas when it has no external subject
        ("what can you do?"). A message that merely contains "you" but names an
        entity ("what do you know about MrBeast") is an external-information
        request, and its evidence requirement must be honoured.
        """

        for note in signals.semantic_notes:
            if note.startswith("local:"):
                continue
        # The subject carried by the semantic reading, if any, is exposed through
        # the evidence features appended by the router.
        return bool(signals.semantic_subject)

    def _file_plan(self, signals: RoutingSignals) -> SourcePlan:
        intent = signals.file_intent
        assert intent is not None
        capabilities: list[str] = []
        if intent.wants_content_search and self._available("filesystem.search_content"):
            capabilities.append("filesystem.search_content")
        elif intent.wants_read:
            if intent.subject or intent.folder:
                capabilities.append("filesystem.search")
            if self._available("filesystem.read"):
                capabilities.append("filesystem.read")
        elif intent.wants_list and self._available("filesystem.list"):
            capabilities.append("filesystem.list")
        elif self._available("filesystem.search"):
            capabilities.append("filesystem.search")
        sources: tuple[SourceType, ...] = (SourceType.FILES,)
        if intent.wants_read and not intent.folder and not intent.extension:
            sources = (SourceType.KNOWLEDGE, SourceType.FILES)
        return self._plan(
            sources,
            "the request is about local files, so file evidence is consulted first",
            capabilities=tuple(capabilities),
        )

    def _system_plan(self, signals: RoutingSignals) -> SourcePlan:
        capabilities = tuple(
            name for name in ("system.info", "processes.list") if self._available(name)
        )
        sources: tuple[SourceType, ...] = (SourceType.SYSTEM, SourceType.MODEL)
        return self._plan(
            sources,
            "the request is about this machine's state, so local observations come first",
            capabilities=capabilities,
        )

    def _web_plan(self, signals: RoutingSignals) -> SourcePlan:
        if not self._available("web.search"):
            return self._plan(
                (SourceType.MODEL,),
                "current information was requested but no web capability is registered",
                current_information_required=True,
            )
        reasons: list[str] = []
        if signals.time_sensitive:
            reasons.append("the answer depends on current information")
        if signals.explicit_web_request:
            reasons.append("the user asked for web information")
        return self._plan(
            (SourceType.WEB, SourceType.MODEL),
            "; ".join(reasons) or "web information was requested",
            capabilities=tuple(
                name for name in ("web.search", "web.fetch") if self._available(name)
            ),
            current_information_required=True,
        )

    def _action_plan(self, signals: RoutingSignals) -> SourcePlan:
        """Route a request that already carries validated actions.

        ``requires_action`` means the request *mutates* local state. A read-only
        request that happens to carry an action (an explicit web search) is
        observed, not confirmed.
        """

        mutating = signals.wants_mutation
        sources: list[SourceType] = []
        capabilities: list[str] = []

        # A hybrid request observes evidence first, then acts on it.
        if signals.explicit_web_request and self._available("web.search"):
            sources.append(SourceType.WEB)
            capabilities.append("web.search")
        if mutating:
            sources.append(SourceType.COMPUTER)
        elif not sources:
            # Read-only action without a web component (e.g. a filesystem read).
            sources.append(SourceType.COMPUTER)

        return self._plan(
            tuple(sources),
            "the request carries validated actions; evidence is gathered before acting",
            capabilities=tuple(dict.fromkeys(capabilities)),
            requires_action=mutating,
        )

    # -- helpers ------------------------------------------------------------------

    @staticmethod
    def _resolved_reference(signals: RoutingSignals) -> bool:
        # The reference is resolvable when the interpreter named a concrete,
        # non-web target. A lone "files" source with no subject, folder, or
        # extension ("find that file") is still an unresolved reference.
        allowed = {source.value for source in SourceType}
        others = [source for source in signals.sources if source in allowed and source != "web"]
        if not others:
            return False
        if set(others) <= {"files"}:
            intent = signals.file_intent
            if intent is None:
                return False
            return bool(intent.subject or intent.folder or intent.extension)
        return True
    def _file_intent_dominates(self, signals: RoutingSignals) -> bool:
        """Return True when file intent outranks the interpreter's web default.

        The interpreter defaults a bare "find"/"search" verb to the web. When the
        user actually referred to files, folders, or documents, the file domain is
        the semantic intent and must win. An explicitly named web platform still
        wins over a weak file hint.
        """

        intent = signals.file_intent
        if intent is None:
            return False
        if not (intent.subject or intent.folder or intent.extension):
            return False
        if any(
            host in signals.text.casefold()
            for host in ("youtube", "google", "the web", "the internet", "online")
        ):
            return False
        return True

    def _available(self, capability: str) -> bool:
        return self._capabilities.exists(capability)

    def _plan(
        self,
        sources: tuple[SourceType, ...],
        reason: str,
        *,
        capabilities: tuple[str, ...] = (),
        current_information_required: bool = False,
        requires_action: bool = False,
    ) -> SourcePlan:
        return SourcePlan(
            sources=sources,
            reason=reason,
            capabilities=capabilities,
            current_information_required=current_information_required,
            requires_action=requires_action
            or any(capability.startswith(_MUTATING_PREFIXES) for capability in capabilities),
        )


def response_mode_for(plan: SourcePlan, request_type: RequestType) -> ResponseMode:
    """Map a selected source plan onto the response mode Atlas will use."""

    if request_type is RequestType.CLARIFICATION:
        return ResponseMode.CLARIFICATION
    if request_type is RequestType.SELF_QUERY:
        return ResponseMode.SELF_DESCRIPTION
    if request_type is RequestType.MEMORY_QUERY:
        return ResponseMode.MEMORY_RECALL
    head = plan.sources[0] if plan.sources else None
    if head is SourceType.FILES:
        return ResponseMode.FILE_LOOKUP
    if head is SourceType.WEB:
        return ResponseMode.WEB_RESEARCH
    if head is SourceType.SYSTEM:
        return ResponseMode.SYSTEM_DIAGNOSIS
    if head is SourceType.KNOWLEDGE:
        return ResponseMode.GROUNDED_ANSWER
    if request_type in {RequestType.ACTION, RequestType.HYBRID}:
        return ResponseMode.ACTION_REPORT
    return ResponseMode.DIRECT_ANSWER


#: Deterministic answerability weight per source.
_SOURCE_WEIGHT: dict[SourceType, float] = {
    SourceType.SELF: 0.95,
    SourceType.FILES: 0.85,
    SourceType.SYSTEM: 0.85,
    SourceType.KNOWLEDGE: 0.8,
    SourceType.WEB: 0.75,
    SourceType.CONVERSATION: 0.75,
    SourceType.MEMORY: 0.7,
    SourceType.MODEL: 0.6,
    SourceType.COMPUTER: 0.6,
}


def confidence_for_plan(plan: SourcePlan, *, requires_confirmation: bool) -> float:
    """Return a deterministic 0..1 answerability score for a source plan."""

    if plan.requires_action and requires_confirmation:
        return 0.5
    if not plan.sources:
        return 0.2
    return max(_SOURCE_WEIGHT.get(source, 0.5) for source in plan.sources)


def level_for_plan(plan: SourcePlan, *, requires_confirmation: bool) -> ConfidenceLevel:
    """Return the coarse confidence bucket for a plan before execution."""

    if requires_confirmation and plan.requires_action:
        return ConfidenceLevel.CRITICAL
    if not plan.sources:
        return ConfidenceLevel.LOW
    if plan.sources[0] in {
        SourceType.SELF,
        SourceType.FILES,
        SourceType.SYSTEM,
        SourceType.KNOWLEDGE,
        SourceType.WEB,
    }:
        return ConfidenceLevel.HIGH
    return ConfidenceLevel.MEDIUM