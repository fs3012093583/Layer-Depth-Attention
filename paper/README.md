# Paper Workspace

This directory is the public-facing paper workspace for the repository branch used for
paper writing and later open-sourcing.

## Intended contents

- `paper_draft.md`: the current manuscript source before LaTeX migration
- `main.tex`: the current LaTeX manuscript source
- `figures/`: publication figures referenced by the draft

## Notes

- Internal working memory files such as `dev_log.md` and `memory/` are intentionally
  excluded from this branch's public structure.
- Large transient analysis directories remain outside `paper/figures/`; only the
  curated figures needed by the manuscript are copied here with stable names.
- Core code for models and training remains in `src/` and `scripts/`.

## Build

The current `main.tex` is a generic pre-template manuscript intended to be migrated
into a venue-specific LaTeX template later. A typical local build command is:

```bash
cd paper
pdflatex main.tex
```
