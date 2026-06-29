# Paper Review Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `tools/review_paper.py` — a single-script BMAD-style workflow that dispatches 4 specialist Claude agents in parallel to review `paper_draft.docx`, then synthesises their findings into a ranked action list.

**Architecture:** Four specialist agents (ScientificReviewer, RhetoricReviewer, Proofreader, AIDetectionReviewer) run concurrently via `asyncio.gather`. Each writes a structured `.md` report. An Orchestrator agent then reads all four reports and writes `synthesis.md`. All output lands in `docs/review/YYYY-MM-DD/`.

**Tech Stack:** Python 3.11+, `anthropic` (async client), `python-docx`, `pypdf`, `pytest`, `pytest-asyncio`

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `tools/__init__.py` | Create (empty) | Makes `tools` a package |
| `tools/review_paper.py` | Create | All logic: extraction, agent prompts, async runner, CLI |
| `tests/__init__.py` | Create (empty) | Test package root |
| `tests/tools/__init__.py` | Create (empty) | Test sub-package |
| `tests/tools/test_review_paper.py` | Create | All unit + integration tests |

---

## Task 1: Scaffold + Install Dependencies

**Files:**
- Create: `tools/__init__.py`
- Create: `tools/review_paper.py` (skeleton only)
- Create: `tests/__init__.py`
- Create: `tests/tools/__init__.py`
- Create: `tests/tools/test_review_paper.py` (failing tests for extraction utilities)

- [ ] **Step 1: Install test dependencies**

```bash
pip install pytest pytest-asyncio
```

Expected output: `Successfully installed pytest-... pytest-asyncio-...`

- [ ] **Step 2: Create directory structure and empty init files**

```bash
mkdir -p tools tests/tools
touch tools/__init__.py tests/__init__.py tests/tools/__init__.py
```

- [ ] **Step 3: Write failing tests for text extraction utilities**

Create `tests/tools/test_review_paper.py`:

```python
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import docx as docx_lib
import pytest
import pytest_asyncio

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_load_api_key_reads_and_strips(tmp_path):
    key_file = tmp_path / "api_key.txt"
    key_file.write_text("  sk-ant-test123\n  ")

    from tools.review_paper import load_api_key
    with patch("tools.review_paper.API_KEY_PATH", key_file):
        key = load_api_key()

    assert key == "sk-ant-test123"


def test_extract_docx_text_includes_headings_and_body(tmp_path):
    doc = docx_lib.Document()
    doc.add_heading("Introduction", level=1)
    doc.add_paragraph("This is the intro text.")
    doc.add_heading("Methods", level=2)
    doc.add_paragraph("We used PatchTST.")
    doc_path = tmp_path / "test.docx"
    doc.save(str(doc_path))

    from tools.review_paper import extract_docx_text
    text = extract_docx_text(doc_path)

    assert "Introduction" in text
    assert "This is the intro text." in text
    assert "Methods" in text
    assert "We used PatchTST." in text


def test_extract_docx_text_nonexistent_raises(tmp_path):
    from tools.review_paper import extract_docx_text
    with pytest.raises(Exception):
        extract_docx_text(tmp_path / "missing.docx")


def test_extract_pdf_text_nonexistent_raises(tmp_path):
    from tools.review_paper import extract_pdf_text
    with pytest.raises(Exception):
        extract_pdf_text(tmp_path / "missing.pdf")
```

- [ ] **Step 4: Run tests to verify they fail (ImportError expected)**

```bash
pytest tests/tools/test_review_paper.py -v
```

Expected: `ImportError` or `ModuleNotFoundError` — `tools.review_paper` does not exist yet.

- [ ] **Step 5: Create the skeleton `tools/review_paper.py`**

```python
import asyncio
import argparse
from datetime import date
from pathlib import Path

import anthropic
import docx
import pypdf

ROOT = Path(__file__).parent.parent
API_KEY_PATH = ROOT / ".claude" / "api_key.txt"
SPRINGER_PDF_PATH = ROOT / "docs" / "Instructions+for+proceedings+authors+(pdf).pdf"
DEFAULT_PAPER = ROOT / "paper_draft.docx"
DEFAULT_OUT = ROOT / "docs" / "review"
MODEL = "claude-opus-4-7"
TODAY = str(date.today())


def load_api_key() -> str:
    return API_KEY_PATH.read_text().strip()


def extract_docx_text(path: Path) -> str:
    doc = docx.Document(str(path))
    lines = []
    for para in doc.paragraphs:
        if not para.text.strip():
            continue
        if para.style.name.startswith("Heading"):
            raw_level = para.style.name.split()[-1]
            level = int(raw_level) if raw_level.isdigit() else 1
            lines.append(f"\n{'#' * level} {para.text}\n")
        else:
            lines.append(para.text)
    return "\n".join(lines)


def extract_pdf_text(path: Path) -> str:
    reader = pypdf.PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)
```

- [ ] **Step 6: Run tests to verify extraction tests pass**

```bash
pytest tests/tools/test_review_paper.py -v
```

Expected:
```
PASSED tests/tools/test_review_paper.py::test_load_api_key_reads_and_strips
PASSED tests/tools/test_review_paper.py::test_extract_docx_text_includes_headings_and_body
PASSED tests/tools/test_review_paper.py::test_extract_docx_text_nonexistent_raises
PASSED tests/tools/test_review_paper.py::test_extract_pdf_text_nonexistent_raises
```

- [ ] **Step 7: Commit**

```bash
git add tools/__init__.py tools/review_paper.py tests/__init__.py tests/tools/__init__.py tests/tools/test_review_paper.py
git commit -m "feat: scaffold review_paper.py with text extraction utilities"
```

---

## Task 2: Agent System Prompts

**Files:**
- Modify: `tools/review_paper.py` — add `AGENTS` dict and `ORCHESTRATOR_SYSTEM`

No unit tests for string constants. Verified structurally in Task 3.

- [ ] **Step 1: Add `REPORT_FORMAT` template and `AGENTS` dict to `tools/review_paper.py`**

Append after the `extract_pdf_text` function:

```python
_REPORT_FORMAT = """
Output a structured markdown report with this exact format:

# {name} Review — paper_draft.docx
*Reviewed: {today}*

## Summary
(2–3 sentences overall assessment)

## Critical Issues
(Each as: - **[Section N]** Description. Quote problematic text if relevant.)

## Major Issues
(Same format)

## Minor Issues
(Same format)

## Positive Notes
(Briefly note genuine strengths)
"""


def _report_fmt(name: str) -> str:
    return _REPORT_FORMAT.replace("{name}", name).replace("{today}", TODAY)


AGENTS: dict[str, dict] = {
    "scientific": {
        "name": "ScientificReviewer",
        "filename": "scientific.md",
        "needs_springer": False,
        "system": (
            "You are a senior ML conference reviewer with expertise in time series forecasting "
            "and multimodal learning (NeurIPS/ICML style).\n\n"
            "Review the provided academic paper. Focus exclusively on scientific rigor:\n"
            "- Verify each contribution claim in §1 Introduction is supported by experimental "
            "results in §6–7\n"
            "- Check all numbers in tables for internal consistency (degradation ratios vs raw "
            "MSE values, means vs reported conclusions)\n"
            "- Flag unsupported or overstated claims (e.g. 'outperforms' without statistical "
            "backing, causal claims from correlational evidence)\n"
            "- Identify missing, misrepresented, or insufficient citations in §3 Related Work\n"
            "- Assess whether the failure mode taxonomy (α-collapse, gate suppression, FiLM "
            "fragility) is adequately evidenced by the described diagnostics\n"
            "- Check experimental protocol completeness: seeds reported, train/val/test splits "
            "defined, metrics specified, datasets described\n\n"
        ) + _report_fmt("ScientificReviewer"),
    },
    "rhetoric": {
        "name": "RhetoricReviewer",
        "filename": "rhetoric.md",
        "needs_springer": False,
        "system": (
            "You are an academic writing coach with experience in machine learning venues.\n\n"
            "Review the provided paper for argument structure and rhetoric only. Do not comment "
            "on grammar, spelling, or formatting. Focus exclusively on:\n"
            "- Whether the abstract accurately reflects the body (contributions, key findings, "
            "datasets used, main conclusions)\n"
            "- Intro↔conclusion alignment: does the conclusion answer what the introduction "
            "promised?\n"
            "- Clarity and parallelism of the four stated contributions\n"
            "- Section-by-section argument flow: does each section set up the next?\n"
            "- Missing or weak transitions between sections\n"
            "- Sections disproportionately long or short relative to their role\n"
            "- Whether §8 Practitioner Guide follows naturally from §6–7 results\n"
            "- Whether the revised framing ('which fusion mechanisms can exploit text') is "
            "consistently maintained throughout\n\n"
        ) + _report_fmt("RhetoricReviewer"),
    },
    "proofread": {
        "name": "Proofreader",
        "filename": "proofread.md",
        "needs_springer": True,
        "system": (
            "You are an expert copy editor with academic publishing experience, specialising "
            "in Springer LNCS/CCIS proceedings.\n\n"
            "Review the paper for language errors AND Springer compliance. You will receive "
            "the paper text and the official Springer author instructions.\n\n"
            "LANGUAGE (check every section):\n"
            "- Grammar, spelling, and punctuation errors (quote the error and give the correction)\n"
            "- British/American English consistency — identify which is used, flag deviations\n"
            "- Number formatting: '%' vs 'percent' (pick one), spacing around '=', decimal "
            "separators\n"
            "- Hyphenation consistency (e.g. 'text-augmented' must be hyphenated consistently)\n"
            "- Sentence-level clarity issues: ambiguous pronouns, dangling modifiers\n\n"
            "SPRINGER LNCS/CCIS COMPLIANCE:\n"
            "- Heading capitalisation: nouns, verbs, adjectives capitalised; articles, "
            "prepositions, conjunctions lowercase — check every heading\n"
            "- Only H1 and H2 numbered; H3 and below unnumbered\n"
            "- Citations in brackets [n], never superscript; multiple citations as [4-6, 9] "
            "in numerical order\n"
            "- Figure captions below figures; table captions above tables\n"
            "- Acknowledgments section present (third-level heading)\n"
            "- Disclosure of Interests section present\n"
            "- Corresponding author marked with a symbol in the header; email mandatory\n"
            "- ORCID identifiers in header\n"
            "- Full paper target: 12–15 pages\n\n"
            "For each issue quote the original text and give the corrected version.\n\n"
        ) + _report_fmt("Proofreader"),
    },
    "ai_detection": {
        "name": "AIDetectionReviewer",
        "filename": "ai_detection.md",
        "needs_springer": False,
        "system": (
            "You are an AI writing naturalness auditor. Identify text that reads as "
            "AI-generated and propose specific natural rewrites.\n\n"
            "Focus on these patterns in priority order:\n"
            "1. Vague intensifiers: 'robust', 'comprehensive', 'significant', 'novel', "
            "'noteworthy', 'state-of-the-art' used without specific referents\n"
            "2. Over-hedging: 'it is worth noting that', 'it can be observed that', "
            "'it is important to note', 'in this regard'\n"
            "3. Hollow section openers: 'In this section, we...', 'We now turn to...', "
            "'Having established X, we now...'\n"
            "4. Repetitive sentence structures: consecutive sentences beginning with 'We' or "
            "following the same Subject-Verb-Object pattern\n"
            "5. LLM-signature phrases: 'delve into', 'shed light on', 'testament to', "
            "'in the realm of', 'a myriad of', 'the intersection of'\n"
            "6. Empty transition summaries: 'As discussed above...', 'As shown in the "
            "previous section...'\n\n"
            "Check abstract, introduction, and conclusion with highest priority.\n\n"
            "For EACH flagged passage:\n"
            "1. Give the section reference (e.g. §1, Abstract)\n"
            "2. Quote the exact original text\n"
            "3. Explain in one sentence what makes it sound AI-generated\n"
            "4. Propose a specific, natural rewrite\n\n"
            "Do not flag standard technical or genuinely appropriate academic phrasing.\n\n"
        ) + _report_fmt("AIDetectionReviewer"),
    },
}

ORCHESTRATOR_SYSTEM = (
    "You are a review coordinator synthesising four specialist reviews of an academic paper.\n\n"
    "Your job:\n"
    "1. Read all four reports (ScientificReviewer, RhetoricReviewer, Proofreader, "
    "AIDetectionReviewer)\n"
    "2. Identify duplicate or overlapping findings — merge into a single item, noting all "
    "source agents\n"
    "3. Assign severity:\n"
    "   - Critical: blocks submission (factual errors, unsupported central claims, major "
    "Springer violations)\n"
    "   - Major: strongly recommended before submission\n"
    "   - Minor: polish (typos, minor phrasing, small inconsistencies)\n"
    "4. Produce a single ranked action list\n\n"
    f"Output this exact format:\n\n"
    f"# Paper Review Synthesis\n"
    f"*Generated: {TODAY}*\n\n"
    "## Overall Assessment\n"
    "(One paragraph: paper's readiness for submission, strongest areas, most urgent gaps)\n\n"
    "## Critical — fix before submission\n"
    "- **[SourceAgent(s)]** Description of issue and specific fix.\n\n"
    "## Major — strongly recommended\n"
    "- **[SourceAgent(s)]** ...\n\n"
    "## Minor — polish\n"
    "- **[SourceAgent(s)]** ...\n\n"
    "## Duplicate findings resolved\n"
    "(Brief note on findings that appeared in multiple reports and how they were merged)\n"
)
```

- [ ] **Step 2: Verify the module still imports cleanly**

```bash
python3 -c "from tools.review_paper import AGENTS, ORCHESTRATOR_SYSTEM; print(list(AGENTS.keys()))"
```

Expected:
```
['scientific', 'rhetoric', 'proofread', 'ai_detection']
```

- [ ] **Step 3: Commit**

```bash
git add tools/review_paper.py
git commit -m "feat: add agent system prompts to review_paper.py"
```

---

## Task 3: Async Agent Runner

**Files:**
- Modify: `tools/review_paper.py` — add `run_agent()`
- Modify: `tests/tools/test_review_paper.py` — add async agent runner tests

- [ ] **Step 1: Add failing tests for `run_agent`**

Append to `tests/tools/test_review_paper.py`:

```python
@pytest.mark.asyncio
async def test_run_agent_writes_report_to_file(tmp_path):
    from tools.review_paper import run_agent

    mock_client = MagicMock()
    mock_message = MagicMock()
    mock_message.content = [MagicMock(text="# ScientificReviewer Review\n\n## Summary\nSolid.")]
    mock_client.messages.create = AsyncMock(return_value=mock_message)

    result = await run_agent(mock_client, "scientific", "paper text here", "", tmp_path)

    assert "ScientificReviewer" in result
    report_path = tmp_path / "scientific.md"
    assert report_path.exists()
    assert report_path.read_text() == result


@pytest.mark.asyncio
async def test_run_agent_proofreader_sends_springer_text(tmp_path):
    from tools.review_paper import run_agent

    mock_client = MagicMock()
    mock_message = MagicMock()
    mock_message.content = [MagicMock(text="# Proofreader Review\n\n## Summary\nOK.")]
    mock_client.messages.create = AsyncMock(return_value=mock_message)

    await run_agent(mock_client, "proofread", "paper text", "springer instructions", tmp_path)

    call_kwargs = mock_client.messages.create.call_args.kwargs
    user_content = call_kwargs["messages"][0]["content"]
    assert "springer instructions" in user_content


@pytest.mark.asyncio
async def test_run_agent_scientific_does_not_send_springer_text(tmp_path):
    from tools.review_paper import run_agent

    mock_client = MagicMock()
    mock_message = MagicMock()
    mock_message.content = [MagicMock(text="# ScientificReviewer Review\n\n## Summary\nOK.")]
    mock_client.messages.create = AsyncMock(return_value=mock_message)

    await run_agent(mock_client, "scientific", "paper text", "springer instructions", tmp_path)

    call_kwargs = mock_client.messages.create.call_args.kwargs
    user_content = call_kwargs["messages"][0]["content"]
    assert "springer instructions" not in user_content
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/tools/test_review_paper.py::test_run_agent_writes_report_to_file -v
```

Expected: `ImportError` — `run_agent` not defined yet.

- [ ] **Step 3: Add `run_agent` to `tools/review_paper.py`**

Append after `ORCHESTRATOR_SYSTEM`:

```python
async def run_agent(
    client: anthropic.AsyncAnthropic,
    agent_key: str,
    paper_text: str,
    springer_text: str,
    out_dir: Path,
) -> str:
    agent = AGENTS[agent_key]
    user_content = f"<paper>\n{paper_text}\n</paper>"
    if agent["needs_springer"]:
        user_content += f"\n\n<springer_instructions>\n{springer_text}\n</springer_instructions>"
    user_content += f"\n\nToday's date: {TODAY}"

    message = await client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=agent["system"],
        messages=[{"role": "user", "content": user_content}],
    )
    report = message.content[0].text
    out_path = out_dir / agent["filename"]
    out_path.write_text(report, encoding="utf-8")
    print(f"[✓] {agent['name']} done → {out_path}")
    return report
```

- [ ] **Step 4: Run agent runner tests to verify they pass**

```bash
pytest tests/tools/test_review_paper.py -k "run_agent" -v
```

Expected:
```
PASSED tests/tools/test_review_paper.py::test_run_agent_writes_report_to_file
PASSED tests/tools/test_review_paper.py::test_run_agent_proofreader_sends_springer_text
PASSED tests/tools/test_review_paper.py::test_run_agent_scientific_does_not_send_springer_text
```

- [ ] **Step 5: Commit**

```bash
git add tools/review_paper.py tests/tools/test_review_paper.py
git commit -m "feat: add async agent runner run_agent()"
```

---

## Task 4: Orchestrator Runner

**Files:**
- Modify: `tools/review_paper.py` — add `run_orchestrator()`
- Modify: `tests/tools/test_review_paper.py` — add orchestrator tests

- [ ] **Step 1: Add failing tests for `run_orchestrator`**

Append to `tests/tools/test_review_paper.py`:

```python
@pytest.mark.asyncio
async def test_run_orchestrator_writes_synthesis(tmp_path):
    from tools.review_paper import run_orchestrator

    mock_client = MagicMock()
    mock_message = MagicMock()
    mock_message.content = [MagicMock(text="# Paper Review Synthesis\n\n## Overall Assessment\nReady to submit.")]
    mock_client.messages.create = AsyncMock(return_value=mock_message)

    reports = {
        "ScientificReviewer": "sci report text",
        "RhetoricReviewer": "rhetoric report text",
    }
    await run_orchestrator(mock_client, reports, tmp_path)

    synthesis_path = tmp_path / "synthesis.md"
    assert synthesis_path.exists()
    assert "Synthesis" in synthesis_path.read_text()


@pytest.mark.asyncio
async def test_run_orchestrator_includes_all_reports_in_prompt(tmp_path):
    from tools.review_paper import run_orchestrator

    mock_client = MagicMock()
    mock_message = MagicMock()
    mock_message.content = [MagicMock(text="# Paper Review Synthesis\n\nDone.")]
    mock_client.messages.create = AsyncMock(return_value=mock_message)

    reports = {
        "ScientificReviewer": "unique-sci-text",
        "Proofreader": "unique-proof-text",
    }
    await run_orchestrator(mock_client, reports, tmp_path)

    call_kwargs = mock_client.messages.create.call_args.kwargs
    user_content = call_kwargs["messages"][0]["content"]
    assert "unique-sci-text" in user_content
    assert "unique-proof-text" in user_content
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/tools/test_review_paper.py -k "run_orchestrator" -v
```

Expected: `ImportError` — `run_orchestrator` not defined yet.

- [ ] **Step 3: Add `run_orchestrator` to `tools/review_paper.py`**

Append after `run_agent`:

```python
async def run_orchestrator(
    client: anthropic.AsyncAnthropic,
    reports: dict[str, str],
    out_dir: Path,
) -> None:
    reports_text = "\n\n---\n\n".join(
        f"# {name} Report\n\n{text}" for name, text in reports.items()
    )
    message = await client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=ORCHESTRATOR_SYSTEM,
        messages=[{"role": "user", "content": f"<reports>\n{reports_text}\n</reports>"}],
    )
    synthesis = message.content[0].text
    out_path = out_dir / "synthesis.md"
    out_path.write_text(synthesis, encoding="utf-8")
    print(f"[✓] Orchestrator done → {out_path}")
```

- [ ] **Step 4: Run orchestrator tests to verify they pass**

```bash
pytest tests/tools/test_review_paper.py -k "run_orchestrator" -v
```

Expected:
```
PASSED tests/tools/test_review_paper.py::test_run_orchestrator_writes_synthesis
PASSED tests/tools/test_review_paper.py::test_run_orchestrator_includes_all_reports_in_prompt
```

- [ ] **Step 5: Commit**

```bash
git add tools/review_paper.py tests/tools/test_review_paper.py
git commit -m "feat: add orchestrator runner run_orchestrator()"
```

---

## Task 5: Main Entry Point + CLI

**Files:**
- Modify: `tools/review_paper.py` — add `run_review()` and `main()`
- Modify: `tests/tools/test_review_paper.py` — add end-to-end mock test

- [ ] **Step 1: Add failing end-to-end test**

Append to `tests/tools/test_review_paper.py`:

```python
@pytest.mark.asyncio
async def test_run_review_creates_all_output_files(tmp_path):
    from tools.review_paper import run_review

    # Create a minimal docx
    doc = docx_lib.Document()
    doc.add_heading("Introduction", level=1)
    doc.add_paragraph("Some paper content here.")
    paper_path = tmp_path / "paper.docx"
    doc.save(str(paper_path))

    # Mock api key
    key_path = tmp_path / "api_key.txt"
    key_path.write_text("sk-fake-key")

    def make_mock_message(text):
        msg = MagicMock()
        msg.content = [MagicMock(text=text)]
        return msg

    call_count = 0

    async def fake_create(**kwargs):
        nonlocal call_count
        call_count += 1
        # First 4 calls = specialists, 5th = orchestrator
        if call_count <= 4:
            system = kwargs.get("system", "")
            name = "ScientificReviewer" if "NeurIPS" in system else \
                   "RhetoricReviewer" if "rhetoric" in system.lower() else \
                   "Proofreader" if "copy editor" in system.lower() else \
                   "AIDetectionReviewer"
            return make_mock_message(f"# {name} Review\n\n## Summary\nTest.")
        return make_mock_message("# Paper Review Synthesis\n\n## Overall Assessment\nOK.")

    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(side_effect=fake_create)

    with patch("tools.review_paper.API_KEY_PATH", key_path), \
         patch("tools.review_paper.anthropic.AsyncAnthropic", return_value=mock_client), \
         patch("tools.review_paper.SPRINGER_PDF_PATH", tmp_path / "missing.pdf"):

        out_dir = await run_review(
            paper_path=paper_path,
            out_base=tmp_path / "review",
        )

    assert (out_dir / "synthesis.md").exists()
    existing = list(out_dir.glob("*.md"))
    assert len(existing) == 5  # 4 specialists + synthesis
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/tools/test_review_paper.py::test_run_review_creates_all_output_files -v
```

Expected: `ImportError` — `run_review` not defined yet.

- [ ] **Step 3: Add `run_review` and `main` to `tools/review_paper.py`**

Append after `run_orchestrator`:

```python
async def run_review(
    paper_path: Path = DEFAULT_PAPER,
    out_base: Path = DEFAULT_OUT,
    springer_path: Path = SPRINGER_PDF_PATH,
) -> Path:
    api_key = load_api_key()
    client = anthropic.AsyncAnthropic(api_key=api_key)

    print("Extracting paper text...")
    paper_text = extract_docx_text(paper_path)

    print("Extracting Springer instructions...")
    try:
        springer_text = extract_pdf_text(springer_path)
    except Exception as e:
        print(f"[!] Could not read Springer PDF: {e}. Proofreader runs without compliance context.")
        springer_text = ""

    out_dir = out_base / str(date.today())
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}\n")

    print("Running specialist reviewers in parallel...")
    tasks = [
        run_agent(client, key, paper_text, springer_text, out_dir)
        for key in AGENTS
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    reports: dict[str, str] = {}
    for key, result in zip(AGENTS.keys(), results):
        agent_name = AGENTS[key]["name"]
        if isinstance(result, Exception):
            print(f"[✗] {agent_name} failed: {result}")
            error_path = out_dir / AGENTS[key]["filename"]
            error_path.write_text(f"# ERROR\n\nAgent failed: {result}\n", encoding="utf-8")
        else:
            reports[agent_name] = result  # type: ignore[assignment]

    print("\nRunning Orchestrator...")
    await run_orchestrator(client, reports, out_dir)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BMAD-style paper review workflow")
    parser.add_argument("--paper", type=Path, default=DEFAULT_PAPER, help="Path to paper_draft.docx")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Base output directory")
    parser.add_argument("--springer-pdf", type=Path, default=SPRINGER_PDF_PATH,
                        help="Path to Springer author instructions PDF")
    args = parser.parse_args()
    asyncio.run(run_review(args.paper, args.out, args.springer_pdf))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run all tests to verify full suite passes**

```bash
pytest tests/tools/test_review_paper.py -v
```

Expected: all 9 tests pass.

```
PASSED tests/tools/test_review_paper.py::test_load_api_key_reads_and_strips
PASSED tests/tools/test_review_paper.py::test_extract_docx_text_includes_headings_and_body
PASSED tests/tools/test_review_paper.py::test_extract_docx_text_nonexistent_raises
PASSED tests/tools/test_review_paper.py::test_extract_pdf_text_nonexistent_raises
PASSED tests/tools/test_review_paper.py::test_run_agent_writes_report_to_file
PASSED tests/tools/test_review_paper.py::test_run_agent_proofreader_sends_springer_text
PASSED tests/tools/test_review_paper.py::test_run_agent_scientific_does_not_send_springer_text
PASSED tests/tools/test_review_paper.py::test_run_orchestrator_writes_synthesis
PASSED tests/tools/test_review_paper.py::test_run_orchestrator_includes_all_reports_in_prompt
PASSED tests/tools/test_review_paper.py::test_run_review_creates_all_output_files
```

- [ ] **Step 5: Verify the CLI entry point is importable**

```bash
python3 -c "from tools.review_paper import main; print('CLI entry point OK')"
```

Expected: `CLI entry point OK`

- [ ] **Step 6: Final commit**

```bash
git add tools/review_paper.py tests/tools/test_review_paper.py
git commit -m "feat: add run_review() and CLI entry point — paper review workflow complete"
```

---

## Usage

After implementation, run with:

```bash
# Default paths (uses paper_draft.docx and docs/review/)
python tools/review_paper.py

# Custom paths
python tools/review_paper.py --paper path/to/paper.docx --out path/to/output/
```

Reports appear in `docs/review/YYYY-MM-DD/`:
- `scientific.md` — ScientificReviewer
- `rhetoric.md` — RhetoricReviewer
- `proofread.md` — Proofreader (language + Springer compliance)
- `ai_detection.md` — AIDetectionReviewer
- `synthesis.md` — Orchestrator ranked action list
