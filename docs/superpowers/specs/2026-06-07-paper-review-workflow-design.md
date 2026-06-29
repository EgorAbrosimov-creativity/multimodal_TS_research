# Paper Review Workflow — Design Spec
*Date: 2026-06-07*

## Overview

A BMAD-style multi-agent workflow that reviews `paper_draft.docx` across four specialist dimensions, then synthesises findings into a single ranked action list. All output is Markdown; no edits are made to the source docx.

## Architecture

```
paper_draft.docx  (text extracted once, shared as input)
        │
  ┌─────┴──────┬──────────────┬───────────────┐
  ▼            ▼              ▼               ▼
Scientific   Rhetoric     Proofreader    AI-Detection
Reviewer     Reviewer                    Reviewer
  │            │              │               │
  ▼            ▼              ▼               ▼
scientific   rhetoric     proofread.md   ai_detection
  .md          .md                           .md
  └─────┬──────┴──────────────┴───────────────┘
        ▼
   Orchestrator
        │
        ▼
   synthesis.md  (ranked: Critical → Major → Minor)
```

Four specialist agents run **concurrently** via `asyncio`. The Orchestrator runs sequentially after all four complete.

## Implementation

**Entry point:** `tools/review_paper.py`

**Dependencies:** `anthropic`, `python-docx`, `pypdf` (all present in project env)

**API key:** read from `.claude/api_key.txt`

**Model:** `claude-opus-4-7` for all five agents

**Output directory:** `docs/review/YYYY-MM-DD/` (datestamped; re-runs never overwrite prior reviews)

### Execution steps

1. Extract full paper text from `paper_draft.docx` via `python-docx` (preserve section headings)
2. Extract Springer instructions text from `docs/Instructions+for+proceedings+authors+(pdf).pdf` via `pypdf`
3. Dispatch 4 specialist agents concurrently — each receives: system prompt (persona) + paper text + extra context where noted
4. Write each specialist report to `docs/review/YYYY-MM-DD/<agent>.md` as it completes
5. Pass all 4 report contents to Orchestrator (single sequential call)
6. Write `docs/review/YYYY-MM-DD/synthesis.md`

## Agent Definitions

### 1. ScientificReviewer

**Persona:** Senior ML conference reviewer (NeurIPS/ICML style)

**Context:** paper text only

**Mandate:**
- Verify each contribution claim in §1 is supported by results in §6–7
- Check all numbers in tables are internally consistent (e.g. ratios match raw MSE values)
- Flag unsupported or overstated claims
- Identify missing or misrepresented citations in related work
- Assess whether the failure mode taxonomy (α-collapse, gate suppression, FiLM fragility) is adequately evidenced
- Check experimental protocol completeness (seeds, splits, metrics)

### 2. RhetoricReviewer

**Persona:** Academic writing coach

**Context:** paper text only

**Mandate:**
- Evaluate argument flow section-by-section
- Check that abstract accurately reflects the body (contributions, findings)
- Check intro↔conclusion alignment
- Assess clarity and parallelism of the four stated contributions
- Flag weak or missing section transitions
- Identify any section that is disproportionately long or short relative to its role
- Check that the Practitioner Guide (§8) follows naturally from §6–7

### 3. Proofreader

**Persona:** Copy editor with academic publishing experience

**Context:** paper text + full Springer proceedings instructions (PDF)

**Mandate:**

*Language:*
- Grammar, spelling, punctuation errors
- British/American English consistency (pick one, flag deviations)
- Number formatting: "%" vs "percent", spacing around "=", decimal separators
- Hyphenation consistency (e.g. "text-augmented" vs "text augmented")

*Springer LNCS/CCIS compliance:*
- Heading capitalisation rules (nouns, verbs capitalised; articles, prepositions, conjunctions lowercase)
- Only H1 and H2 numbered; no "0" section numbers
- Citations in brackets `[n]`, not superscript; multiple citations as `[4-6, 9]`
- Figure captions below figures; table captions above tables
- No color in text, tables, or equations
- Acknowledgments section present (third-level heading, 9pt)
- Disclosure of Interests section present
- Corresponding author marked with symbol; email mandatory in header
- ORCID in header
- Page length: 12–15 pages for a full paper
- Appendix (if present) placed before references

### 4. AIDetectionReviewer

**Persona:** AI writing naturalness auditor

**Context:** paper text only

**Mandate:**
- Flag sentences that read as AI-generated: vague intensifiers ("robust", "comprehensive", "significant"), over-hedging ("it is worth noting that", "it can be observed"), hollow transitions ("in this section, we")
- Identify repetitive sentence structures within paragraphs
- Flag formulaic phrases that are common in LLM output
- For each flagged passage: quote the original and propose a specific, natural rewrite
- Focus on abstract, intro, and conclusion — highest visibility sections

### 5. Orchestrator

**Persona:** Review coordinator

**Context:** all four specialist reports (concatenated)

**Mandate:**
- Deduplicate overlapping findings across agents
- Assign severity to each unique finding: **Critical** (blocks submission) / **Major** (strongly recommended) / **Minor** (polish)
- Produce a single ranked action list grouped by severity
- Tag each item with its source agent(s)
- Add a one-paragraph overall assessment at the top

## Report Formats

### Specialist report (`<agent>.md`)

```markdown
# [Agent Name] Review — paper_draft.docx
*Reviewed: YYYY-MM-DD*

## Summary
2–3 sentences overall assessment.

## Critical Issues
- **[Section X]** Description. Quote if relevant.

## Major Issues
- ...

## Minor Issues
- ...

## Positive Notes
- ...
```

### Synthesis report (`synthesis.md`)

```markdown
# Paper Review Synthesis
*Generated: YYYY-MM-DD*

## Overall Assessment
One paragraph.

## Critical — fix before submission
- [source agents] Description.

## Major — strongly recommended
- ...

## Minor — polish
- ...

## Duplicate findings resolved
Brief note on any finding that appeared in multiple reports and how it was merged.
```

## Error Handling

- If a specialist agent call fails (API error / timeout): write an error placeholder to that agent's `.md` file and continue; the Orchestrator skips missing reports and notes them in synthesis
- If `paper_draft.docx` cannot be read: exit immediately with a clear message
- If the PDF cannot be read: Proofreader runs without Springer compliance context; a warning is printed
- Progress printed to stdout as each agent completes: `[✓] ScientificReviewer done → docs/review/.../scientific.md`

## File Layout

```
tools/
  review_paper.py          — entry point
docs/
  review/
    YYYY-MM-DD/
      scientific.md
      rhetoric.md
      proofread.md
      ai_detection.md
      synthesis.md
  Instructions+for+proceedings+authors+(pdf).pdf   — existing
.claude/
  api_key.txt              — existing
```

## Usage

```bash
python tools/review_paper.py
# or with explicit paths:
python tools/review_paper.py --paper paper_draft.docx --out docs/review/
```
