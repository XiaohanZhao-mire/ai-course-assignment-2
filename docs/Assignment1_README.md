# Assignment 1: FastAPI Word Embeddings

This project follows the Module 3 `gentext-app` class activity and adds the
spaCy word-embedding function from Module 2 Practical 3. The original root and
bigram text-generation endpoints are retained.

## Run with Docker

Install and start Docker, then run these commands from this project folder:

```bash
docker build -t gentext-app .
docker run --rm -p 8000:8000 gentext-app
```

Open http://localhost:8000/docs to try the API interactively. The service and
container mapping both use port **8000** (the class handout mixes 80 and 8000).
The image installs the model during the build; startup does not download it.
The first build needs internet access and downloads a model of approximately
400 MB. Allow a few GB of free disk and memory for the image and model.
No API key, paid model service, or cloud subscription is required.

## Query the new endpoint

```bash
curl --get 'http://localhost:8000/embedding' --data-urlencode 'word=apple'
```

Response fields:

| Field | Meaning |
|---|---|
| `word` | Input word, with outer whitespace removed |
| `model` | `en_core_web_lg` |
| `dimensions` | 300 |
| `embedding` | The complete list of 300 floating-point values |

The implementation uses `nlp(word).vector`, as in the class notebook, and
converts the NumPy array to a JSON list. The model is loaded once at server
startup. Components unrelated to static vectors are excluded to reduce startup
work, while the tokenizer and pretrained vocabulary remain in use.

Input policy: one alphabetic word, at most 100 characters. Missing, blank,
multiword, numeric, or punctuation-only inputs return **422**. A word without
a vector in this vocabulary returns **404**, so an unknown word is not silently
presented as a meaningful zero vector. The original letter case is preserved.

## Original classroom endpoint

```bash
curl -X POST 'http://localhost:8000/generate' \
  -H 'Content-Type: application/json' \
  -d '{"start_word":"generating","length":4}'
```

Expected response:

```json
{"generated_text":"generating text based on"}
```

`length` counts the starting word and must be an integer from 1 to 200.
Generation stops early if no next word is available; an unknown start word is
returned by itself. Other paths can vary because generation samples from bigram
probabilities.

## Run locally with uv

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
uv sync --frozen
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Python 3.12 is selected by `.python-version`. `uv.lock` fixes the dependency
versions, including the exact spaCy model release. Run checks with:

```bash
uv run pytest -q
# With the API running in another terminal:
uv run python scripts/smoke_test.py
```

GitHub Actions builds the Linux Docker image, runs the tests with the actual
model, starts the container, and queries it over HTTP. See the repository's
Actions tab for each run's result.

## Files and classroom sources

| File | Purpose / source |
|---|---|
| `app/main.py` | FastAPI structure and `/generate` from the Module 3 activity; adds `/embedding` |
| `app/bigram_model.py` | Tokenization, counting, and sampling adapted from Module 2 Practical 2 |
| `app/embedding_model.py` | `en_core_web_lg` and `nlp(word).vector` from Module 2 Practical 3 |
| `Dockerfile`, `pyproject.toml`, `uv.lock` | Reproducible deployment and dependencies |
| `tests/test_api.py` | Real-model comparisons, invalid inputs, original endpoint checks |
| `scripts/smoke_test.py` | HTTP checks against a running local or Docker server |
| `docs/Assignment1_Probability_Solutions.pdf` | All six theory answers with calculations |

The classroom notebooks remain reference material. The application contains
only the functionality needed for this assignment. Bigram transition weights
are normalized by observed outgoing transitions, and separate corpus samples
are not joined into artificial cross-sample bigrams.

Official implementation references:
[spaCy English models](https://spacy.io/models/en),
[FastAPI Docker deployment](https://fastapi.tiangolo.com/deployment/docker/),
[uv Docker integration](https://docs.astral.sh/uv/guides/integration/docker/).
