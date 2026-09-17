"""Task-aware web retrieval: understand *what* the user wants, not just match words.

The chunk-ranking approach in :mod:`web_research` answers "which text is closest
to the query", which is not the same as "which source delivers the thing the user
asked for". A Wikipedia page ABOUT a film can out-score the film's actual script
because it says the film's name more often. This module adds the missing task
layer that the retrieval pipeline runs before it ranks chunks:

* :class:`RetrievalTask` - the structured goal: what is wanted, of what type, and
  whether it must be the artifact itself or information about it.
* :func:`infer_retrieval_task` - deterministic classification of a request into a
  goal and a :class:`ContentType`, with room for an LLM to confirm ambiguity.
* :func:`detect_source_type` / :func:`score_source` - rank candidate *sources*
  (title, URL, structure, completeness, presence of the target) before any chunk
  is scored, so a describe-the-thing page cannot beat the thing itself.
* :func:`validate_content` - decide whether extracted content actually satisfies
  the task, and explain a rejection so the caller can reformulate the query.

Scoring here is deterministic and inspectable. The language model is consulted
only for genuinely ambiguous judgment calls, never for routing or execution.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping

# ---------------------------------------------------------------------------
# Content types
# ---------------------------------------------------------------------------

#: Requested/observed content types. ``generic`` is plain informational text.
CONTENT_TYPES: tuple[str, ...] = (
    'movie_script',
    'transcript',
    'lyrics',
    'article',
    'news',
    'review',
    'documentation',
    'forum',
    'product',
    'video',
    'reference',
    'social',
    'code',
    'list',
    'image',
    'document_host',
    'listing',
    'generic',
)

#: Types that describe an artifact the user wants *as text in full*.
DOCUMENT_TYPES: frozenset[str] = frozenset(
    {'movie_script', 'transcript', 'lyrics', 'documentation', 'code', 'list'}
)

# ---------------------------------------------------------------------------
# Retrieval goals
# ---------------------------------------------------------------------------

#: The user's goal, which changes what counts as a good source.
GOALS: tuple[str, ...] = (
    'retrieve_document',  # the artifact itself (script, transcript, lyrics, code)
    'find_information',   # information about a subject
    'find_page',          # a specific page/site
    'find_review',        # opinions about a subject
    'find_media',         # video/audio/image
    'find_reference',     # an encyclopedic/reference entry
    'create_content',     # generate new content (no retrieval)
    'navigate',           # go to a place
    'perform_action',     # run an action (may combine with retrieval)
)

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

#: Phrase -> content type. Longer, more specific phrases first.
_CONTENT_TYPE_TERMS: tuple[tuple[str, str], ...] = (
    ('movie script', 'movie_script'),
    ('film script', 'movie_script'),
    ('screenplay', 'movie_script'),
    ('shooting script', 'movie_script'),
    ('script', 'movie_script'),
    ('transcript', 'transcript'),
    ('subtitles', 'transcript'),
    ('transcription', 'transcript'),
    ('lyrics', 'lyrics'),
    ('song lyrics', 'lyrics'),
    ('source code', 'code'),
    ('code', 'code'),
    ('implementation', 'code'),
    ('documentation', 'documentation'),
    ('docs', 'documentation'),
    ('manual', 'documentation'),
    ('readme', 'documentation'),
    ('api reference', 'documentation'),
    ('tutorial', 'documentation'),
    ('review', 'review'),
    ('critique', 'review'),
    ('rating', 'review'),
    ('trailer', 'video'),
    ('teaser', 'video'),
    ('clip', 'video'),
    ('video', 'video'),
    ('watch', 'video'),
    ('article', 'article'),
    ('blog post', 'article'),
    ('essay', 'article'),
    ('news', 'news'),
    ('headline', 'news'),
    ('wiki', 'reference'),
    ('wikipedia', 'reference'),
    ('encyclopedia', 'reference'),
    ('reference', 'reference'),
    ('forum', 'forum'),
    ('reddit', 'forum'),
    ('thread', 'forum'),
    ('discussion', 'forum'),
    ('product', 'product'),
    ('buy', 'product'),
    ('price', 'product'),
    ('spec sheet', 'product'),
    ('list of', 'list'),
    ('list', 'list'),
)

#: Phrases that mean "the artifact itself", not information about it.
_ARTIFACT_PHRASES: tuple[str, ...] = (
    'full script', 'complete script', 'entire script', 'the script',
    'full transcript', 'complete transcript', 'entire transcript', 'the transcript',
    'full lyrics', 'the lyrics', 'full text', 'complete text', 'whole text',
    'the source code', 'full code', 'entire code', 'the raw text',
)

#: Phrases that mean "information about the subject".
_INFO_PHRASES: tuple[str, ...] = (
    'information about', 'info about', 'information on', 'learn about',
    'tell me about', 'explain', 'what is', 'who is', 'overview of',
    'background on', 'facts about', 'details about',
)

#: Interrogative openers that ask *about* a subject ("who wrote the bee movie
#: script", "how long is the bee movie script") rather than requesting the
#: artifact itself. Polite request frames (please/can you/could you) are
#: stripped first because they introduce a request, not a question.
_QUESTION_OPENERS = re.compile(
    r'^(?:please\s+|can\s+you\s+|could\s+you\s+|would\s+you\s+|will\s+you\s+)*'
    r'(?:what|who|when|where|why|how|which|whose|is|are|was|were|does|do|did)\b'
)

_WEB_HOSTS = ('the web', 'the internet', 'online', 'internet')

#: Sources whose primary role is reference/encyclopedic.
_REFERENCE_HOSTS = ('wikipedia.org', 'wikiwand.com', 'britannica.com', 'wikimedia.org')

#: Sources that host community/Q&A/forum content.
_FORUM_HOSTS = ('reddit.com', 'stackexchange.com', 'stackoverflow.com', 'quora.com', 'discourse')

#: Sources commonly hosting scripts/transcripts.
_SCRIPT_HOSTS = (
    'imsdb.com', 'scriptslug.com', 'dailyscript.com', 'scripts.com',
    'subslikescript.com', 'springfieldspringfield.co.uk', 'youdrivewhat.com',
    'transcripts.fandom.com', 'fandom.com', 'genius.com', 'azlyrics.com',
    'gutenberg.org', 'archive.org',
)

#: Sources that tend to be commercial/product or social.
_PRODUCT_HOSTS = ('amazon.', 'ebay.', 'etsy.', 'walmart.', 'shop.')
_SOCIAL_HOSTS = ('twitter.com', 'x.com', 'facebook.com', 'instagram.com', 'tiktok.com', 'pinterest.')
#: Hosts that present a *landing/listing page about* an uploaded document rather
#: than the document itself (Scribd-style). Their text is card boilerplate, not
#: the requested artifact, so an artifact request must not be satisfied by them.
_DOCUMENT_HOSTS = (
    'scribd.com', 'studylib.net', 'slideshare.net', 'coursehero.com', 'pdfcoffee.com',
    'docslib.org', 'pdfslide.net', 'vdocuments.', 'dokumen.pub', 'dokumen.tips',
    'academia.edu', 'researchgate.net', 'everand.com', 'baixardoc.com', 'dokument.pub',
)
#: Phrases repeated in document-store card chrome (“0 ratings”, “views”...).
_CARD_BOILERPLATE = (
    'found this document useful', 'ratings', 'views', 'uploaded by',
    'download to read', 'download now', 'save for later', 'embed', 'share',
    'report this document', 'document description', 'original title',
)


@dataclass(frozen=True)
class RetrievalTask:
    """What the user is actually trying to obtain.

    ``goal`` says what to do; ``content_type`` says what kind of thing is wanted;
    ``must_be_artifact`` distinguishes "the script itself" from "a page about the
    script". ``destination`` records where the result should go, if anywhere.
    """

    goal: str = 'find_information'
    target: str = ''
    content_type: str = 'generic'
    desired_output: str = 'summary'  # full_text | summary | links | media
    must_be_artifact: bool = False
    destination: str | None = None
    search_required: bool = True
    site: str | None = None
    queries: tuple[str, ...] = ()
    reason: str = ''
    source: str = 'deterministic'  # deterministic | llm

    def to_dict(self) -> dict[str, Any]:
        return {
            'goal': self.goal,
            'target': self.target,
            'content_type': self.content_type,
            'desired_output': self.desired_output,
            'must_be_artifact': self.must_be_artifact,
            'destination': self.destination,
            'search_required': self.search_required,
            'site': self.site,
            'queries': list(self.queries),
            'reason': self.reason,
            'source': self.source,
        }


@dataclass
class SourceScore:
    """A candidate page scored against the retrieval task."""

    url: str
    title: str
    score: float
    detected_type: str
    signals: dict[str, float] = field(default_factory=dict)
    accepted: bool = False
    reason: str = ''

    def to_dict(self) -> dict[str, Any]:
        return {
            'url': self.url,
            'title': self.title,
            'score': round(self.score, 4),
            'detected_type': self.detected_type,
            'signals': {key: round(value, 4) for key, value in self.signals.items()},
            'accepted': self.accepted,
            'reason': self.reason,
        }


# ---------------------------------------------------------------------------
# Term helpers
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.casefold())


def _phrase_in(phrase: str, lowered: str) -> bool:
    return re.search(r'(?<![a-z0-9])' + re.escape(phrase) + r'(?![a-z0-9])', lowered) is not None


# ---------------------------------------------------------------------------
# Task inference
# ---------------------------------------------------------------------------

def infer_retrieval_task(
    request: str,
    *,
    entities: Mapping[str, Any] | None = None,
    site: str | None = None,
    destination: str | None = None,
    must_be_artifact: bool | None = None,
    goal: str | None = None,
) -> RetrievalTask:
    """Classify a request into a :class:`RetrievalTask` (deterministic).

    The interpreter's extracted entities (``topic``, ``content_type``,
    ``application``, ``site``) refine the result when present, so this function
    can be called either with a raw request or with the Task IR's view of it.
    """

    entities = entities or {}
    lowered = request.casefold()
    # Prefer the interpreter's topic; otherwise reduce the request to its object.
    target = str(entities.get('topic') or '').strip() or _strip_command(lowered)
    content_type = _detect_requested_type(lowered, entities)
    if must_be_artifact is None:
        must_be_artifact = _wants_artifact(lowered, content_type)
    elif must_be_artifact and content_type == 'generic':
        # An explicit artifact request still needs a document-ish type so the
        # query variants and validation target the right thing.
        content_type = 'article'
    if goal is None:
        goal = _infer_goal(lowered, content_type, must_be_artifact, site)

    if goal == 'find_review':
        content_type = content_type if content_type == 'review' else 'review'
    if goal == 'find_media':
        content_type = content_type if content_type in {'video', 'image'} else 'video'
    if goal == 'find_reference':
        content_type = content_type if content_type == 'reference' else 'reference'

    desired_output = 'full_text' if must_be_artifact else (
        'media' if goal == 'find_media' else 'links' if goal in {'find_page', 'navigate'} else 'summary'
    )

    effective_site = site or (str(entities.get('site')) if entities.get('site') else None)
    dest = destination or (
        str(entities.get('application') or entities.get('filename') or '') or None
    )

    return RetrievalTask(
        goal=goal,
        target=target,
        content_type=content_type,
        desired_output=desired_output,
        must_be_artifact=must_be_artifact,
        destination=dest,
        search_required=goal != 'create_content',
        site=effective_site,
        queries=_initial_queries(lowered, target, content_type, must_be_artifact, effective_site),
        reason=_explain(goal, content_type, must_be_artifact),
    )


def _detect_requested_type(lowered: str, entities: Mapping[str, Any]) -> str:
    # An explicit content_type entity from the interpreter is authoritative.
    entity_type = str(entities.get('content_type') or '').strip().casefold()
    if entity_type in CONTENT_TYPES:
        return entity_type
    # The interpreter's fast content nouns map onto types directly.
    if entity_type in {'script', 'screenplay'}:
        return 'movie_script'
    if entity_type in {'lyrics', 'song'}:
        return 'lyrics'
    if entity_type in {'video', 'clip', 'trailer'}:
        return 'video'
    if entity_type in {'review', 'rating'}:
        return 'review'
    # Otherwise match the most specific phrase in the request text.
    for phrase, content_type in _CONTENT_TYPE_TERMS:
        if _phrase_in(phrase, lowered):
            return content_type
    return 'generic'


def _wants_artifact(lowered: str, content_type: str) -> bool:
    if any(phrase in lowered for phrase in _ARTIFACT_PHRASES):
        return True
    if any(phrase in lowered for phrase in _INFO_PHRASES):
        # "tell me about the script" is information, not the artifact itself.
        return False
    if _QUESTION_OPENERS.match(lowered):
        # "who wrote the bee movie script" / "how long is the bee movie script"
        # ask a question *about* the subject, so a page that answers it is the
        # right answer; the artifact itself is not being requested.
        return False
    # "find/get/the X script|transcript|lyrics|code" without an info phrase is a
    # request for the artifact.
    if content_type in DOCUMENT_TYPES:
        ct_keyword = content_type.split('_')[-1]
        # The content type was detected from the request text, so the keyword
        # (e.g. "script") is already present. A bare mention like "bee movie
        # script" — with no article, no retrieval verb, and no info phrase —
        # is still a direct request for the artifact itself.
        if re.search(r'(?<![a-z0-9])' + re.escape(ct_keyword) + r'(?![a-z0-9])', lowered):
            return True
        article = 'the|a|an|my|this|that|full|complete|entire|whole'
        if re.search(r'(?<![a-z0-9])(?:' + article + r')\s+[a-z ]*' + re.escape(ct_keyword), lowered):
            return True
        # Bare "bee movie script" as the object of a retrieval verb.
        if re.search(r'(?:find|get|fetch|grab|copy|retrieve|search for|look up|pull)\b', lowered):
            return True
    return False


def _infer_goal(lowered: str, content_type: str, must_be_artifact: bool, site: str | None) -> str:
    if _phrase_in('wikipedia', lowered) or _phrase_in('wiki page', lowered) or _phrase_in('wiki', lowered):
        return 'find_reference'
    if content_type == 'review' or re.search(r'(?:review|reviews|rating|critique)s?\b', lowered):
        return 'find_review'
    if content_type in {'video', 'image'} or re.search(r'\b(?:trailer|teaser|watch|video|clip)\b', lowered):
        return 'find_media'
    if must_be_artifact or content_type in DOCUMENT_TYPES:
        return 'retrieve_document'
    if _youtube_is_media(lowered, site, content_type):
        return 'find_media'
    if any(phrase in lowered for phrase in _INFO_PHRASES):
        return 'find_information'
    if re.search(r'\b(?:page|website|site|url)\b', lowered):
        return 'find_page'
    return 'find_information'


def _initial_queries(
    lowered: str, target: str, content_type: str, must_be_artifact: bool, site: str | None
) -> tuple[str, ...]:
    """Seed queries; for an artifact request, bias toward the artifact wording."""

    base = target or _strip_command(lowered)
    queries: list[str] = []
    if must_be_artifact:
        # "Bee Movie" -> "Bee Movie full script transcript"
        suffix = _artifact_suffix(content_type)
        queries.append(f'{base} {suffix}'.strip())
        queries.append(f'{base} script transcript'.strip())
    queries.append(base.strip())
    return tuple(dict.fromkeys(query for query in queries if query))


def _artifact_suffix(content_type: str) -> str:
    return {
        'movie_script': 'full script transcript',
        'transcript': 'full transcript',
        'lyrics': 'full lyrics',
        'code': 'source code',
        'documentation': 'complete documentation',
        'list': 'full list',
    }.get(content_type, 'full text')


def _strip_command(lowered: str) -> str:
    # Remove a trailing placement clause first, then the leading command/target
    # scaffolding, so "search the web for bee movie script and copy it in
    # notepad" reduces to "bee movie script".
    text = re.sub(r'\b(?:and|then)\s+(?:copy|paste|type|put|write|save|place|add|insert)\b.*$', ' ', lowered)
    text = re.sub(r'\b(?:copy|paste|type|put|write|save|place|add|insert)\b.*$', ' ', text)
    text = re.sub(
        r'\b(?:search|find|get|fetch|grab|retrieve|look up|pull|read|google)\b', ' ', text
    )
    text = re.sub(r'\b(?:the web|the internet|on the web|online|web|internet)\b', ' ', text)
    text = re.sub(r'\b(?:youtube|google|bing)\b', ' ', text)
    text = re.sub(r'\bfor\b', ' ', text)
    # "for the bee movie script" / "about the bee movie" -> drop the lead-in.
    text = re.sub(r'^\s*(?:for|about|on|regarding|of)?\s*(?:the|a|an|my|this|that)?\s+', '', text)
    text = re.sub(r'\b(?:in|into|to|on)\s+(?:notepad|it|my files?|a file|the file)\b.*$', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def _youtube_is_media(lowered: str, site: str | None, content_type: str) -> bool:
    # "search youtube for popular car videos" wants videos; "search youtube for
    # information about cars" wants research about cars.
    if str(site or '').casefold() != 'youtube' and 'youtube' not in lowered:
        return False
    if content_type in {'video', 'image'}:
        return True
    if any(phrase in lowered for phrase in _INFO_PHRASES):
        return False
    return bool(re.search(r'\b(?:videos?|clips?|trailers?|popular|trending|watch)\b', lowered))


def _explain(goal: str, content_type: str, must_be_artifact: bool) -> str:
    if must_be_artifact:
        return f'the request asks for the {content_type} itself, not a page about it'
    return f'{goal} for a {content_type}'


# ---------------------------------------------------------------------------
# Source detection and ranking
# ---------------------------------------------------------------------------

def _host(url: str) -> str:
    match = re.match(r'https?://([^/]+)', url or '')
    return match.group(1).casefold() if match else ''


def detect_source_type(url: str, title: str, text: str) -> str:
    """Best-effort content type of a source page (deterministic)."""

    host = _host(url)
    lowered_title = (title or '').casefold()
    raw_sample = (text or '')[:4000]
    sample = raw_sample.casefold()

    if any(token in host for token in _REFERENCE_HOSTS) or 'wikipedia' in lowered_title:
        return 'reference'
    if any(token in host for token in _FORUM_HOSTS):
        return 'forum'
    if any(token in host for token in _SOCIAL_HOSTS):
        return 'social'
    if any(token in host for token in _PRODUCT_HOSTS):
        return 'product'
    if _host_is_video(url, host, lowered_title):
        return 'video'
    # A document-store landing page presents card chrome about an uploaded
    # document, not the document itself, even when its title names the artifact.
    if any(token in host for token in _DOCUMENT_HOSTS) or _boilerplate_ratio(raw_sample) >= 0.5:
        return 'document_host'
    # Documentation heuristics.
    if any(_phrase_in(phrase, sample) for phrase in ('api reference', 'documentation', 'getting started', 'installation')):
        return 'documentation'
    # Scripts/transcripts read as dialogue-heavy text (speaker names are
    # uppercase, so this must run on the raw, non-lowercased sample).
    if _looks_like_dialogue(raw_sample):
        return 'transcript'
    if any(_phrase_in(phrase, lowered_title) for phrase in ('script', 'transcript', 'screenplay')):
        return 'movie_script'
    if _phrase_in('lyrics', lowered_title) or any(token in host for token in ('azlyrics', 'genius.com')):
        return 'lyrics'
    if _phrase_in('review', lowered_title) or _phrase_in('rating', lowered_title):
        return 'review'
    if _looks_like_code(sample):
        return 'code'
    if _phrase_in('news', lowered_title) or any(_phrase_in(phrase, lowered_title) for phrase in ('breaking', 'report')):
        return 'news'
    if any(token in host for token in _SCRIPT_HOSTS):
        return 'article'
    return 'generic'


def _host_is_video(url: str, host: str, title: str) -> bool:
    if any(token in host for token in ('youtube.com', 'youtu.be', 'vimeo.com', 'dailymotion')):
        return True
    return any(_phrase_in(phrase, title) for phrase in ('trailer', 'teaser', 'watch '))


def _looks_like_dialogue(sample: str) -> bool:
    # Dialogue lines such as "BARRY: ..." / "NAME: ..." repeated many times.
    speaker_lines = len(re.findall(r'(?m)(?:^|\s)[A-Z][A-Z0-9 .]{1,24}:', sample))
    return speaker_lines >= 4


def _looks_like_code(sample: str) -> bool:
    signals = (
        sample.count('def '), sample.count('function '), sample.count('class '),
        sample.count('import '), sample.count('{\n'),
    )
    return sum(signals) >= 4

def _boilerplate_ratio(text: str) -> float:
    # Document-store/listing pages are dominated by card chrome ("0 ratings",
    # "found this document useful", "Uploaded by", "views", "pages"). Real
    # content has almost none of these terms. This counts distinct chrome phrases
    # rather than raw repetition, so a legitimately repeated chorus or line in
    # real content is not mistaken for a listing page.
    lowered = (text or '').casefold()
    if not lowered.strip():
        return 0.0
    hits = sum(1 for phrase in _CARD_BOILERPLATE if phrase in lowered)
    return min(1.0, hits / 3.0)


def score_source(
    task: RetrievalTask,
    *,
    url: str,
    title: str,
    snippet: str = '',
    text: str = '',
) -> SourceScore:
    """Rank one candidate source against the task before chunk scoring.

    Rewards title/URL/entity match, requested-type match, presence of the target
    artifact, source quality, and completeness; penalises describe-the-subject
    pages when the artifact itself was requested.
    """

    lowered_title = (title or '').casefold()
    lowered_url = (url or '').casefold()
    lowered_snippet = (snippet or '').casefold()
    lowered_text = (text or '').casefold()
    detected = detect_source_type(url, title, text)

    terms = [term for term in _tokens(task.target or '') if len(term) >= 3]
    signals: dict[str, float] = {}

    signals['title_match'] = _coverage(terms, lowered_title)
    signals['url_match'] = _coverage(terms, lowered_url)
    signals['snippet_match'] = _coverage(terms, lowered_snippet)
    signals['body_match'] = _coverage(terms, lowered_text[:6000])
    signals['type_match'] = 1.0 if _type_matches(task.content_type, detected) else 0.0
    signals['artifact_presence'] = _artifact_presence(task, detected, lowered_title, lowered_text)
    signals['quality'] = _source_quality(url, detected)
    signals['completeness'] = _completeness(lowered_text)

    penalty = 0.0
    if task.must_be_artifact:
        if _is_descriptive_page(detected, lowered_title, lowered_text):
            penalty += 0.6
        if detected == 'reference' and task.content_type != 'reference':
            penalty += 0.3
    # A page that never mentions the target is irrelevant regardless of type.
    if terms and signals['title_match'] == 0 and signals['body_match'] == 0 and signals['url_match'] == 0:
        penalty += 0.5

    score = (
        0.22 * signals['title_match']
        + 0.12 * signals['url_match']
        + 0.10 * signals['snippet_match']
        + 0.16 * signals['body_match']
        + 0.20 * signals['type_match']
        + 0.12 * signals['artifact_presence']
        + 0.05 * signals['quality']
        + 0.03 * signals['completeness']
        - penalty
    )
    accepted = score >= 0.35
    return SourceScore(
        url=url,
        title=title,
        score=max(0.0, score),
        detected_type=detected,
        signals=signals,
        accepted=accepted,
        reason=_source_reason(task, detected, signals, penalty, accepted),
    )


def _coverage(terms: list[str], haystack: str) -> float:
    if not terms:
        return 0.0
    matched = sum(1 for term in terms if term in haystack)
    return matched / len(terms)


def _type_matches(requested: str, detected: str) -> bool:
    # A document-store landing page or a generic listing is never the requested
    # content type, whatever was asked for.
    if detected in {'document_host', 'listing'}:
        return False
    if requested == 'generic':
        return detected not in {'product', 'social'}
    if requested == detected:
        return True
    # Transcripts and movie scripts are the same family for ranking purposes.
    families = [
        {'movie_script', 'transcript'},
        {'article', 'news', 'review'},
        {'documentation', 'code'},
    ]
    return any(requested in family and detected in family for family in families)


def _artifact_presence(task: RetrievalTask, detected: str, title: str, text: str) -> float:
    if task.content_type in DOCUMENT_TYPES or task.must_be_artifact:
        structural = 0.0
        if _looks_like_dialogue(text[:4000]) and task.content_type in {'movie_script', 'transcript'}:
            structural += 0.7
        if any(_phrase_in(phrase, title) for phrase in ('script', 'transcript', 'screenplay', 'code', 'documentation')):
            structural += 0.3
        # A document-store page / listing names the artifact in its title but its
        # body is card chrome, so it must not earn artifact presence for that.
        if detected in {'document_host', 'listing'} or _boilerplate_ratio(text) >= 0.5:
            structural *= 0.15
        return min(1.0, structural)
    return 0.5 if _type_matches(task.content_type, detected) else 0.2


def _source_quality(url: str, detected: str) -> float:
    host = _host(url)
    if any(token in host for token in _REFERENCE_HOSTS):
        return 0.8
    if any(token in host for token in _SCRIPT_HOSTS):
        return 0.9
    if any(token in host for token in _SOCIAL_HOSTS + _PRODUCT_HOSTS):
        return 0.3
    if detected in {'forum', 'social', 'product', 'document_host', 'listing'}:
        return 0.3
    return 0.6


def _completeness(lowered_text: str) -> float:
    length = len(lowered_text)
    if length >= 8000:
        return 1.0
    if length >= 3000:
        return 0.7
    if length >= 800:
        return 0.4
    return 0.15 if length else 0.0


def _is_descriptive_page(detected: str, title: str, text: str) -> bool:
    # Reference/review/news pages describe the subject rather than being it, and
    # a document-store/listing page merely advertises the document.
    if detected in {'reference', 'review', 'news', 'product', 'forum', 'social', 'document_host', 'listing'}:
        return True
    descriptive_markers = (' was a ', ' is a ', ' released in ', ' directed by ', ' stars ',
                           ' box office ', ' received ', ' plot ', ' cast ')
    sample = text[:4000]
    return sum(1 for marker in descriptive_markers if marker in sample) >= 3 and not _looks_like_dialogue(sample)


def _source_reason(
    task: RetrievalTask, detected: str, signals: dict[str, float], penalty: float, accepted: bool
) -> str:
    parts = [f'detected={detected}', f'type_match={signals["type_match"]:.2f}',
             f'title={signals["title_match"]:.2f}', f'body={signals["body_match"]:.2f}']
    if penalty:
        parts.append(f'penalty={penalty:.2f}')
    parts.append('accepted' if accepted else 'rejected')
    return '; '.join(parts)


# ---------------------------------------------------------------------------
# Content validation
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Whether extracted content satisfies the task, plus an explanation."""

    ok: bool
    content_type_match: bool
    contains_target: bool
    completeness: float
    reason: str
    action: str  # extract | search_again
    source: str = 'deterministic'  # deterministic | llm

    def to_dict(self) -> dict[str, Any]:
        return {
            'match': self.ok,
            'content_type_match': self.content_type_match,
            'contains_target': self.contains_target,
            'completeness': round(self.completeness, 3),
            'reason': self.reason,
            'action': self.action,
            'source': self.source,
        }


def validate_content(
    task: RetrievalTask,
    *,
    content: str,
    detected_type: str,
    title: str = '',
    url: str = '',
    min_completeness: float = 0.2,
) -> ValidationResult:
    """Check that extracted content actually satisfies the task.

    A describe-the-subject page fails validation when the artifact itself was
    requested, so the caller can retrieve again instead of acting on it.
    """

    text = content or ''
    lowered = text.casefold()
    terms = [term for term in _tokens(task.target or '') if len(term) >= 3]
    contains_target = (not terms) or any(term in lowered for term in terms) or _coverage(
        terms, (title or '').casefold()
    ) > 0
    type_match = _type_matches(task.content_type, detected_type)
    completeness = _content_completeness(task, text, detected_type)

    if not text.strip():
        return ValidationResult(False, False, False, 0.0,
                                'no content was extracted', 'search_again')
    if not contains_target:
        return ValidationResult(False, type_match, False, completeness,
                                'the content does not mention the requested subject', 'search_again')
    # Content dominated by repeated card chrome (a document-store/listing page)
    # advertises the document but is not it; reject it for any goal.
    if detected_type in {'document_host', 'listing'} or _boilerplate_ratio(text) >= 0.5:
        return ValidationResult(
            False, False, contains_target, completeness,
            'the source is a listing/advert page about the content, not the content itself',
            'search_again',
        )

    if task.must_be_artifact or task.content_type in DOCUMENT_TYPES:
        if not type_match:
            return ValidationResult(
                False, False, contains_target, completeness,
                f'expected a {task.content_type} but the source is {detected_type}', 'search_again',
            )
        if _is_descriptive_page(detected_type, title, text):
            return ValidationResult(
                False, type_match, contains_target, completeness,
                'the source describes the subject but is not the requested document', 'search_again',
            )
        if completeness < min_completeness:
            return ValidationResult(
                False, type_match, contains_target, completeness,
                'the extracted document is too short to be the requested content', 'search_again',
            )
        structural = _looks_like_dialogue(text[:6000]) if task.content_type in {'movie_script', 'transcript'} else True
        if not structural:
            return ValidationResult(
                False, type_match, contains_target, completeness,
                'the content does not read like the requested document type', 'search_again',
            )
        return ValidationResult(True, True, True, completeness,
                                'the source contains the requested document', 'extract')

    # Informational goals are satisfied by relevant content of a sane type.
    if task.content_type not in {'generic'} and not type_match:
        return ValidationResult(
            False, False, contains_target, completeness,
            f'expected {task.content_type} content but found {detected_type}', 'search_again',
        )
    return ValidationResult(True, type_match, True, completeness,
                            'the content is relevant to the request', 'extract')


def _content_completeness(task: RetrievalTask, content: str, detected_type: str) -> float:
    length = len(content.strip())
    if task.must_be_artifact or task.content_type in DOCUMENT_TYPES:
        # A real script/transcript is long; a synopsis is not.
        if length >= 12000:
            return 1.0
        if length >= 4000:
            return 0.6
        if length >= 1200:
            return 0.35
        return 0.1
    if length >= 800:
        return 0.9
    if length >= 250:
        return 0.6
    return 0.3 if length else 0.0


# ---------------------------------------------------------------------------
# Query reformulation
# ---------------------------------------------------------------------------

def reformulate_query(
    task: RetrievalTask,
    *,
    reason: str,
    attempt: int,
) -> str:
    """Produce a better query from the *missing* content type, not random words.

    Deterministic: each attempt biases toward a stronger artifact/intent wording.
    """

    base = task.target or 'the requested content'
    content_type = task.content_type
    if content_type == 'movie_script':
        variants = [
            f'{base} full script transcript',
            f'{base} script screenplay pdf',
            f'{base} "full script" transcript dialogue',
        ]
    elif content_type == 'transcript':
        variants = [f'{base} full transcript', f'{base} complete transcript subtitles']
    elif content_type == 'lyrics':
        variants = [f'{base} full lyrics', f'{base} lyrics complete song']
    elif content_type == 'code':
        variants = [f'{base} source code', f'{base} full source code github']
    elif content_type == 'review':
        variants = [f'{base} review', f'{base} reviews rating critique']
    elif content_type == 'reference':
        variants = [f'{base} wikipedia', f'{base} encyclopedia entry']
    elif content_type in {'video', 'image'}:
        variants = [f'{base} trailer video', f'{base} official video']
    elif content_type == 'documentation':
        variants = [f'{base} documentation', f'{base} official docs reference']
    else:
        variants = [f'{base} {_artifact_suffix(content_type)}', base]
    index = min(max(attempt, 0), len(variants) - 1)
    return variants[index]
def llm_validate_content(
    task: RetrievalTask,
    *,
    content: str,
    detected_type: str,
    title: str = '',
    url: str = '',
    request: str = '',
    ask: Any,
) -> ValidationResult | None:
    """Ask the local model to judge a borderline retrieval decision.

    This is the *only* place the LLM touches retrieval, and it never executes
    anything: it returns a structured judgment that the deterministic layer then
    honours. Returns ``None`` when the model is unavailable or unhelpful, in
    which case the caller keeps the deterministic decision.
    """

    from reasoning.json_llm import safe_reasoning_call
    from reasoning.prompts import (
        RETRIEVAL_VALIDATOR_SYSTEM,
        retrieval_validator_user_prompt,
    )

    data = safe_reasoning_call(
        system_prompt=RETRIEVAL_VALIDATOR_SYSTEM,
        user_prompt=retrieval_validator_user_prompt(
            request=request or task.target or task.content_type,
            task=task.to_dict(),
            source_title=title,
            source_url=url,
            detected_type=detected_type,
            content_sample=content,
        ),
        ask=ask,
    )
    if not isinstance(data, dict):
        return None

    def flag(name: str, default: bool) -> bool:
        value = data.get(name)
        return value if isinstance(value, bool) else default

    def ratio(name: str, default: float) -> float:
        try:
            return max(0.0, min(1.0, float(data.get(name, default))))
        except (TypeError, ValueError):
            return default

    action = str(data.get('action') or '').strip().lower()
    match = flag('match', False)
    if action in {'extract', 'search_again'}:
        match = action == 'extract'
    return ValidationResult(
        ok=match,
        content_type_match=flag('content_type_match', match),
        contains_target=flag('contains_target', match),
        completeness=ratio('completeness', 0.0),
        reason=str(data.get('reason') or 'model judgment'),
        action='extract' if match else 'search_again',
        source='llm',
    )
