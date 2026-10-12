# Chelle AI: Gemini vs. OpenAI on a Multi-Step Career-Planning Workflow

This project compares how closely Gemini's and OpenAI's agents match a real college advisor's answers on a five-stage agentic workflow: building a career plan from high school through leadership.

## How it works

1. **Workflow** (`chelle-ai.py`). A [goose](https://pypi.org/project/goose-ai/) flow runs five dependent stages. Each stage's output is fed into the next stage's prompt.
   1. High school
   2. College
   3. Entry-level job
   4. Mid-career
   5. Management / leadership
2. **Two providers, same flow.** The identical workflow runs once with Gemini (`AIModel.GEMINI_FLASH`) and once with OpenAI (`AIModel.GPT_4O`).
3. **Scoring** (`token_matcher.py`). Each stage's output is compared with the advisor's answer for that stage using a weighted soft token matcher.
4. **Outputs.** Per-stage precision / recall / F1 for each provider are printed, and distribution graphs and raw model outputs are saved.

## The token matcher

A BERTScore-style matcher on a 2025 embedding model, `Alibaba-NLP/gte-modernbert-base` (built on ModernBERT, 8192-token context).

1. Each text is split into whole words.
2. A contextual embedding is computed for every word (the mean of its word-piece vectors).
3. Every advisor word is compared with every model word by cosine similarity.
4. For each word, the best match on the other side is kept. Matching is by meaning, so "internship" can match "co-op".
5. Each word is scaled by a weight from 0 to 1, where 1 is a complete match to the message's meaning and 0 is unrelated.
   - Weights come from manual labels when provided.
   - Otherwise filler words ("the", "of", ...) get 0 and all other words get 1.
6. Scores:
   - **Recall** = weighted share of the advisor's content that the model covered.
   - **Precision** = weighted share of the model's content that the advisor supports.
   - **F1** = harmonic mean of the two.

The distribution graphs are histograms of each advisor word's best-match similarity, weighted by its label, drawn per stage with both models overlaid.

### Interpreting scores

Embedding cosine similarities are compressed: even loosely related words often score 0.6 to 0.8. Raw scores will therefore look high. Use them to compare the two providers against each other rather than as absolute accuracy. Rescaling against an unrelated-text baseline would make absolute values more meaningful.

## Files

| File | Purpose |
|---|---|
| `chelle-ai.py` | Main script: prompts for career inputs, runs both providers, scores, saves outputs |
| `token_matcher.py` | Matcher, labeling helpers, and plotting |
| `advisor_answers.json` | Reference answers per stage (placeholders, replace with the real advisor's text) |
| `token_labels.json` | *Optional.* Hand-labelled word weights |
| `chelle-ai.py.bak` | Backup of the original single-provider script |

Generated on each run: `similarity_distributions.png` and `model_outputs.json`.

## Setup

```bash
pip install torch "transformers>=4.48" numpy matplotlib
# plus goose and its provider credentials (Gemini and OpenAI API keys)
```

Before running:

- In `chelle-ai.py`, confirm `AIModel.GPT_4O` is the correct member name in your goose `AIModel` enum, and that `GEMINI_FLASH` maps to the Gemini version you want to compare.
- Replace the contents of `advisor_answers.json` with the real advisor's answers.

## Usage

```bash
python chelle-ai.py
```

You will be asked for your current stage, target role, target company, timeline, and career goals. The script then runs both providers, prints scores per stage, and writes the graph and outputs.

## Data formats

**`advisor_answers.json`**: one entry per stage. Each answer is a list of lines (joined with newlines on load) or a plain string.

```json
{
  "high_school": ["First paragraph...", "", "Second paragraph..."],
  "college": "A plain string also works."
}
```

**`token_labels.json`** (optional): one `[word, weight]` pair per word, in the order produced by `words()` in `token_matcher.py`. The weight list for a text must be exactly as long as that text's word list, or scoring raises an error.

```json
{
  "high_school": {
    "advisor": [["Take", 0.4], ["courses", 1.0]],
    "gemini": [["..."]],
    "openai": [["..."]]
  }
}
```

Generate a correctly aligned starting point to edit by hand:

```python
from token_matcher import labeling_template
import json
print(json.dumps(labeling_template(text)))
```

## Limitations

- Scores depend on the quality and consistency of the manual weights.
- Raw similarity values are compressed (see above).
- Results come from a single run per provider, and LLM outputs vary between runs.
