"""Lexical file location for tasks that name no path.

FastPath's value depends on the model seeing the code it must change. When the
task text offers a filename, `resolve_prefetch_paths` finds it. Real issue
reports do not: measured on the five derivable benchmark statements, the path
regex matched 0/5 and the L1 classifier guessed right 1/4. Those runs reach the
model with an empty prompt context and take the documented exit, at 23k-44k
tokens each.

This module ranks the repository's own files against the statement's vocabulary,
so a statement like "saferepr() crashes on a broken __repr__" reaches
`saferepr.py` without anyone naming it.

Measured recall on those five statements, base commit trees, cap of 3:

    IDF alone                    40%
    + filename stem affinity     40%   (@1 20%, @5 60%)
    + test-file downweight       60%
    both                         60%   (@1 40%, @5 80%)

Both mechanisms are justified without reference to the sample, which matters
because n=5 cannot support a tuned parameter:

  - Test downweight: `SYSTEM_PROMPT` forbids the model from adding or editing
    tests, so a prefetch slot holding a test file is a slot holding a file it may
    not touch.
  - Stem affinity's 5-character floor is the smallest value that accepts
    "evaluation" -> `evaluate.py` (shared "evaluat") while still rejecting
    "skipif" -> `skipping.py` (shared "skip"), which is a plausible file but the
    wrong answer.

Two mechanisms were measured and rejected rather than omitted for simplicity.
Definition-site weighting (extra score when a file defines a name from the
statement) is flat at @3 and worse at @5: test files define many functions named
after statement words, so it amplifies exactly what the downweight suppresses.
Identifier-exact matching, in place of substring, drops @3 to 20%.

The result is advisory. It flows to `prefetch_hint`, never to `files_hint`,
because `files_hint` is also condition 3's enforcement contract -- a guess
promoted there would reject a correct patch instead of merely wasting tokens.
"""
from __future__ import annotations

import math
import os
import re
from collections import defaultdict
from pathlib import Path

# Extensions worth ranking. Narrower than the path regex's set: a task is
# located by the code that implements it, and ranking a lockfile or a changelog
# first spends a prefetch slot on a file nobody needs to edit.
_RANKED_SUFFIXES = frozenset({
    ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".rb", ".java",
    ".c", ".h", ".cpp", ".hpp", ".cs", ".php", ".swift", ".kt", ".sh", ".sql",
})

# Ceiling on files considered. Beyond this the locator returns nothing rather
# than ranking a partial tree, because a truncated walk would rank whichever
# subtree `os.walk` happened to reach first and present it as evidence.
#
# Measured: reading and scoring 220 files / 2.3MB costs ~50ms (~70MB/s), so 2000
# files is roughly 450ms worst case. That is affordable against the 23k-44k
# tokens a blind FastPath call spends. It is also not hypothetical protection --
# this repository's own tree holds ~60k Python files under gitignored benchmark
# result directories, which no name-based exclusion list would catch.
MAX_LOCATOR_FILES = 2_000

# Per-file read ceiling. A generated or vendored file megabytes long contributes
# vocabulary out of all proportion to its relevance, and reading it is most of
# the cost. Truncation is fine: the ranking needs which words appear, not all of
# them.
MAX_LOCATOR_FILE_BYTES = 200_000

# Total bytes read across the tree. Bounds the worst case independently of file
# count, since one repository's 2000 files can be 20x another's.
MAX_LOCATOR_TOTAL_BYTES = 20_000_000

# Directories never worth ranking. Same rationale as fastpath's prefetch scan:
# build output and vendored dependencies hold no task-relevant code, and they
# are the main source of vocabulary that swamps IDF.
EXCLUDED_DIRS = frozenset({
    ".git", ".hg", ".svn", ".tox", ".nox", ".venv", "venv", "env",
    "node_modules", "__pycache__", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", "site-packages", "dist", "build", ".eggs",
})

# Shortest token worth searching for. Below this, tokens are mostly English
# fragments that match everywhere and carry no location signal.
MIN_TOKEN_LEN = 4

# A token in more than this fraction of the tree cannot locate anything,
# whatever its raw count: `pytest` inside pytest is not an address.
UBIQUITY_CUTOFF = 0.5

# Weight multiplier when the filename itself carries the token. A statement
# about saferepr() and a file called `saferepr.py` is nearly an address, and no
# amount of body-text frequency elsewhere should outrank that.
FILENAME_BONUS = 2.0

# Shortest shared prefix accepted as a morphological filename match. See the
# module docstring for why 5 rather than 4.
MIN_STEM_PREFIX = 5

# Multiplier applied to test files. Not zero: a test file can still be the best
# available context for understanding an API, and the model is told not to edit
# it. Small enough that any non-test file with real signal outranks it.
TEST_PENALTY = 0.25

_TEST_DIRS = frozenset({"test", "tests", "testing"})

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# Closed-class function words and version-control verbs. Deliberately contains
# no domain vocabulary: an earlier hand-written list that included defect words
# ("regression", "missing", "twice") scored 60% at @3, but leave-one-out showed
# the gain vanished once the measured task's own words could not contribute --
# the list had been fitted to the answers. Every word here is one that would
# appear in a stop list written for any repository, in any language, before
# seeing a single task.
STOP_WORDS = frozenset({
    "the", "this", "that", "then", "than", "with", "without", "from", "into",
    "have", "has", "had", "was", "were", "will", "would", "when", "where",
    "which", "while", "what", "does", "done", "just", "only", "also",
    "because", "since", "already", "still", "even", "same", "each", "every",
    "some", "any", "not", "but", "and", "for", "its", "so", "now", "here",
    "there", "they", "them", "their", "such", "other", "another", "both",
    "either", "neither", "more", "less", "most", "least", "very", "must",
    "should", "could", "can", "cannot", "never", "always", "like", "over",
    "under", "before", "after", "again", "instead", "rather",
    "add", "adds", "added", "fix", "fixes", "fixed", "remove", "removes",
    "removed", "update", "updates", "updated", "change", "changes", "changed",
    "use", "uses", "used", "using",
})


def statement_tokens(task: str) -> tuple[str, ...]:
    """Identifier-shaped words from the task worth searching the repo for.

    Whole tokens and their parts are both kept, so `files_hint` in a task
    matches a literal `files_hint` in code while also reaching `hint`. Dunder
    names lose their underscores, letting `__repr__` reach `repr`. CamelCase is
    split so `SafeRepr` reaches both halves.
    """
    found: set[str] = set()

    def keep(word: str) -> None:
        low = word.lower()
        if len(low) >= MIN_TOKEN_LEN and low not in STOP_WORDS:
            found.add(low)

    for raw in _IDENT.findall(task):
        stripped = raw.strip("_")
        keep(stripped)
        for part in re.findall(r"[A-Z]?[a-z]{3,}|[A-Z]{3,}", raw):
            keep(part)
        for part in stripped.split("_"):
            keep(part)

    return tuple(sorted(found))


def _candidate_paths(root: Path) -> tuple[str, ...] | None:
    """Rankable files under `root`, or None if the tree is too large.

    None and `()` mean different things and the caller treats them alike, but
    the distinction is kept because it is the difference between "nothing here
    matches" and "this tree is too big to have an opinion about". Ranking a
    truncated walk would silently present whichever subtree `os.walk` reached
    first as if it were the whole repository.

    `os.walk` rather than `rglob` so excluded directories can be pruned in place:
    a `.venv` is skipped without descending into it, which is where the cost is.
    """
    found: list[str] = []

    for dirpath, dirnames, filenames in os.walk(root, onerror=None):
        # Mutating `dirnames` in place is what prunes the walk. Slice assignment
        # is required -- rebinding the name would not affect os.walk.
        dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIRS]

        for name in filenames:
            if os.path.splitext(name)[1] not in _RANKED_SUFFIXES:
                continue
            found.append(os.path.relpath(os.path.join(dirpath, name), root))
            if len(found) > MAX_LOCATOR_FILES:
                return None

    return tuple(sorted(found))


def _read_bounded(root: Path, paths: tuple[str, ...]) -> dict[str, str]:
    """Lowered text of each file, within the read budget.

    Files are read in sorted order so the budget cuts off deterministically:
    the same repository must yield the same ranking every time, or A/B results
    become unreproducible. Unreadable files are skipped -- a tree at an
    arbitrary commit can hold a broken symlink or a file this process cannot
    open, and that says nothing about the task.
    """
    texts: dict[str, str] = {}
    total = 0

    for rel in paths:
        if total >= MAX_LOCATOR_TOTAL_BYTES:
            break
        try:
            with open(root / rel, encoding="utf-8", errors="replace") as fh:
                text = fh.read(MAX_LOCATOR_FILE_BYTES)
        except OSError:
            continue
        texts[rel] = text.lower()
        total += len(text)

    return texts


def _shared_prefix_len(a: str, b: str) -> int:
    limit = min(len(a), len(b))
    i = 0
    while i < limit and a[i] == b[i]:
        i += 1
    return i


def _filename_carries(token: str, rel: str) -> bool:
    """Whether the filename itself points at this token.

    Substring covers the direct case (`saferepr` in `saferepr.py`). Prefix
    matching covers the morphological one, which substring misses entirely: a
    statement says "evaluation" and the file is `evaluate.py`.
    """
    base = rel.replace(os.sep, "/").rsplit("/", 1)[-1]
    if token in base:
        return True
    stem = base.rsplit(".", 1)[0]
    return _shared_prefix_len(token, stem) >= MIN_STEM_PREFIX


def _is_test_path(rel: str) -> bool:
    parts = rel.replace(os.sep, "/").split("/")
    if not _TEST_DIRS.isdisjoint(p.lower() for p in parts[:-1]):
        return True
    base = parts[-1].lower()
    return base.startswith("test_") or base.endswith("_test.py")


def _rank(texts: dict[str, str], tokens: tuple[str, ...]) -> list[tuple[str, float]]:
    """Files ordered by summed IDF of the tokens they contain, best first.

    IDF carries the ranking: a token in 200 files says nothing about location,
    while one in two files is nearly an address. Without it the ranking
    degenerates to "whichever file is longest".

    Ties break on path so the output is deterministic. Two files with identical
    scores are equally good evidence, and picking between them by dict order
    would make the same repository rank differently across runs.
    """
    n_total = max(len(texts), 1)
    score: dict[str, float] = defaultdict(float)

    for token in tokens:
        body = [rel for rel, text in texts.items() if token in text]
        # A file whose *name* carries the token holds it as surely as one whose
        # body mentions it, so both count toward the document frequency. Scoring
        # only body matches would drop a token that appears nowhere but in a
        # filename: `not body` would skip it, and the filename bonus below --
        # the entire point of stem affinity -- would never run. Measured neutral
        # on the five benchmark statements (identical ranks), because in a
        # 200-file tree a token in a filename is also somewhere in body text.
        # It matters in small trees, and it is what the mechanism claims to do.
        named = [rel for rel in texts if _filename_carries(token, rel)]
        holders = sorted(set(body) | set(named))
        if not holders or len(holders) > n_total * UBIQUITY_CUTOFF:
            continue
        idf = math.log(n_total / len(holders))

        for rel in holders:
            score[rel] += idf
        for rel in named:
            score[rel] += idf * FILENAME_BONUS

    for rel in list(score):
        if _is_test_path(rel):
            score[rel] *= TEST_PENALTY

    return sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))


def locate_files(task: str, root: Path, limit: int) -> tuple[str, ...]:
    """Up to `limit` repo-relative paths whose contents best match the task.

    Advisory only. The caller must route this to `prefetch_hint` and never to
    `files_hint`: a wrong guess here should cost tokens, not reject a correct
    patch by narrowing condition 3's contract.

    Returns `()` when the task yields no usable tokens, when nothing scores, or
    when the tree exceeds `MAX_LOCATOR_FILES`. An empty result is the pre-locator
    behaviour, so failing this way costs nothing beyond the walk.
    """
    if limit <= 0:
        return ()

    tokens = statement_tokens(task)
    if not tokens:
        return ()

    paths = _candidate_paths(root)
    if not paths:
        return ()

    ranked = _rank(_read_bounded(root, paths), tokens)
    return tuple(rel for rel, _ in ranked[:limit])
