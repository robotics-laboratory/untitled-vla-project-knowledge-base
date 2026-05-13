# Research Knowledge Base

This vault follows a specific structure built for research projects. Read this before adding files.

## Two types of documents

The structure separates two kinds of writing that must not mix: evergreen knowledge (what you understand about the domain, lives in `10-concepts/` and `20-papers/`) and temporal log (what you did and when, lives in `40-experiments/log/` and `50-daily/`). They evolve by different rules and connect through links. Mixing them produces either a dated diary where nothing is findable, or a sterile wiki with no chronology.

## Folder structure

```
.
├── 00-meta/          # project overview, roadmap, decision log, reading queue
├── 10-concepts/      # one note per concept, evergreen
├── 20-papers/        # one note per paper with your analysis
├── 30-models/        # deep dives into specific systems (openhelix/, pi05/, ...)
├── 40-experiments/   # actual work; one subfolder per experiment
├── 50-daily/         # daily log organized by week
├── 60-writing/       # paper drafts
├── 70-conversations/ # saved Claude/advisor conversations, unedited
├── 99-attic/         # abandoned ideas
└── _templates/       # templates for all note types
```

Each experiment folder has a fixed internal layout:

```
40-experiments/probe-0-cluster-structure/
├── 00-pre-registration.md   # locked before running
├── 01-design.md             # implementation details
├── 02-results.md            # numbers and figures
├── 03-discussion.md         # interpretation, evolves over time
├── 04-followups.md          # what to do next
├── log/                     # daily entries scoped to this experiment
│   ├── 2026-05-13.md
│   └── ...
└── artifacts/               # links to code and figures
```

Pre-registration is a separate locked file because it closes. Results grow. Discussion evolves. Keeping them in the same file makes you afraid to revise discussion without "corrupting" what was pre-registered.

## When to write what

**Pre-registration** goes before the experiment starts, written in one sitting, then locked. This is the contract with yourself that prevents post-hoc rationalization.

**Results** get filled in immediately after you have numbers. Not the next day.

**Daily log** takes 10 minutes every evening: what you did, what you learned, what is blocking you, what is next. Three lines is enough. Missing a day loses context you will not recover.

**Paper notes** are written right after reading. If you did not write a note, you did not read the paper.

**Concept notes** are written the first time you explain something to yourself and understand it. The trigger: after each "now I get X", open `10-concepts/x.md` and write it in your own words.

**Conversation dumps** go to `70-conversations/` immediately, unedited. Extract durable knowledge into concept notes separately.

## Templates

Copy the relevant template when creating a new note. All templates are in `_templates/`.

| Template | Use for |
|---|---|
| `paper-note.md` | Every paper you read (`20-papers/`) |
| `concept-note.md` | Every concept you internalize (`10-concepts/`) |
| `experiment-pre-registration.md` | Before running any experiment |
| `experiment-design.md` | Implementation details (`01-design.md`) |
| `experiment-results.md` | Raw numbers and figures (`02-results.md`) |
| `experiment-discussion.md` | Interpretation (`03-discussion.md`) |
| `experiment-followups.md` | Next steps (`04-followups.md`) |
| `daily-log.md` | Every working day (`50-daily/`) |
| `week-review.md` | Every Friday (`50-daily/YYYY-Www/`) |

## Linking

Use `[[note-name]]` links liberally. In a paper note, link every concept it uses. In an experiment, link the papers and concepts it builds on. In the daily log, link everything touched that day. Obsidian's backlinks panel gives you a live knowledge graph for free.

The naming convention for `20-papers/` is `{short-title}-{arxiv-id}.md`, e.g. `openhelix-2505.03912.md`. For `10-concepts/` use lowercase kebab-case, e.g. `flow-matching.md`.
