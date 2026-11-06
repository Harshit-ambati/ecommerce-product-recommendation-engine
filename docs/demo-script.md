# Demo recording plan (3–5 minutes)

1. 0:00–0:40: Explain top-ten discovery. Show Data overview, 20,982 synthetic events, quality checks and generation assumptions.
2. 0:40–1:30: Explain the three methods, chronological cutoffs, validation selection and held-out test.
3. 1:30–2:30: Select U0000 and collaborative filtering. Inspect a recommendation and contributing product in history. Switch to content and New shopper. Explain saved-artifact loading and fallback.
4. 2:30–3:30: Show metrics and coverage. Discuss collaborative cold-item failure, content's reach, and U0000's zero-hit error case.
5. 3:30–4:30: Show training/inference commands and passing tests. Explain limitations and AI assistance. Demonstrate understanding of event weights or ranking count.

## Submission checklist

- Record and upload the actual video; this script is not the required video.
- Repository: https://github.com/Harshit-ambati/ecommerce-product-recommendation-engine
- Branch: main. Use `git rev-parse HEAD` for the final commit.
- Grant reviewer access to the private repository and verify all links.
- Report and diagram: `docs/technical-report.md`.
- Dataset evidence: `docs/data-card.md` and `recommender/data.py`.
- Artifact: regenerate with `python -m recommender.pipeline` into `artifacts/recommender.pkl`.
- Declare local-only demo unless deployed later.
- Add candidate identification and exact invitation deadline to the form.
- Disclose AI assistance and prepare for the 15-minute walkthrough and 10-minute live change/debugging task.
