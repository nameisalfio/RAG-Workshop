# RAG workshop

<img align="right" src="docs/paradigma-asterisks.svg" alt="" height="96">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/paradigma-logo-dark.svg">
  <img src="docs/paradigma-logo.svg" alt="Paradigma" height="32">
</picture>

A complete Retrieval-Augmented Generation system that answers questions about the documents of a
fictional company, Torrevalle Acque: 195 contracts, addenda, manuals, plant sheets, tickets,
policies, minutes and e-mails. You type a question in a web page, the system finds the passages
that matter, gives them to a language model and shows the answer together with its sources.

This guide follows the order of the workshop: the theory, the notebooks that explain every choice,
then the system started one piece at a time, up to the chat and its validation.

## 1. The theory, and where it lives in the code

The slides are in [`docs/RAG-workshop.html`](docs/RAG-workshop.html). Open the file in a browser
and move with the arrow keys; the page loads its libraries from the internet.

Each part of the slides has a notebook that tries it out and a module that runs it:

| In the slides | In the notebooks | In the code |
|---|---|---|
| The knowledge base: parsing | `00_parsing.ipynb` | `src/ingestion/parsing.py` |
| The knowledge base: chunking | `01_chunking.ipynb` | `src/ingestion/chunking.py` |
| Embeddings | `02_embedding.ipynb` | `src/retrieval/embedding.py` |
| Retrieval: vector database, hybrid search, re-ranking | `03_reranking.ipynb` | `src/retrieval/vector_store.py`, `src/retrieval/reranking.py` |
| Generation: the prompt | `04_rag.ipynb` | `src/rag.py`, `src/llm.py` |
| Evaluation | the Evaluation tab of the web page | `src/evaluation.py` |

```
RAG-Workshop/
├── docs/               the slides
├── notebooks/          one notebook per step: how each piece was chosen
├── raw-documents/      the 195 documents
├── src/
│   ├── conf/config.yaml   every setting
│   ├── ingestion/      offline: documents -> text -> chunks -> vector database
│   ├── retrieval/      online: question -> vectors -> search -> re-ranker
│   ├── rag.py          retrieve, augment, generate
│   ├── llm.py          client of the language model
│   ├── evaluation.py   metrics against the golden set
│   └── api.py          backend of the web page
├── frontend/           the web page
├── vector-db/          the vector database, ready to load
├── evaluation/         golden set and recorded results
└── docker-compose.yml  the four services
```

## 2. Get ready

You need Docker Desktop, about 12 GB of free disk space and 8 GB of memory assigned to Docker.
Tested on macOS (Apple Silicon).

```bash
git clone https://github.com/nameisalfio/RAG-Workshop.git
cd RAG-Workshop
cp .env.example .env
```

The language model is reached through OpenRouter, which offers some models for free. Sign in at
https://openrouter.ai, open **API Keys**, press **New Key**, and paste the key in `.env`:

```
OPENROUTER_API_KEY=sk-or-v1-...
```

The file `.env` stays on your machine: git ignores it. A free account can make 50 requests a day;
when they are used up, the page says so and tells when they start again.

## 3. The notebooks: what each piece does, and why it is set this way

Every setting in `src/conf/config.yaml` was chosen by trying the alternatives on the real
documents. The notebooks show that work, one step at a time. They are optional: the system in
section 4 runs without them.

```bash
conda env create -f environment.yml
conda activate rag-workshop
python -m ipykernel install --user --name rag-workshop --display-name "Python (rag-workshop)"
python -m src.ingestion.parsing     # reads the 195 documents, about 5 minutes
```

Open them in VS Code, or in any Jupyter client, with the kernel **Python (rag-workshop)**.

| Notebook | The question it answers | What was chosen, and why |
|---|---|---|
| `00_parsing.ipynb` | How do we turn a PDF into text? | **Docling**: of five parsers, the only one that keeps tables, columns and headings. |
| `01_chunking.ipynb` | Where do we cut a document? | **On its sections, 300 tokens at most**: an article of a contract stays in one piece. |
| `02_embedding.ipynb` | How does a text become a point in space? | **BGE-M3**: it reads Italian and English, and gives a dense and a sparse vector in one pass. |
| `03_reranking.ipynb` | How do we find the right passages? | **Hybrid search and a re-ranker**: on the golden set Recall goes from 0.60 to 0.86 and MRR from 0.59 to 0.82. |
| `04_rag.ipynb` | What changes when the model reads our documents? | **Retrieve, augment, generate**: from a model that does not know, to an answer with its sources. |

Notebooks 03 and 04 search the vector database: start it and fill it first (steps 1 and 2 below).
Notebook 04 also needs the key in `.env`.

## 4. The pipeline, one command at a time

The system is four services. Starting them one by one shows what each of them does.
`docker compose up --build` starts all of them at once.

### Step 1. The vector database

```bash
docker compose up -d qdrant
```

Qdrant is the database that will hold the passages and their vectors. It has its own dashboard:

[![Open the Qdrant dashboard](https://img.shields.io/badge/Open_the_Qdrant_dashboard-0a2843?style=for-the-badge)](http://localhost:6333/dashboard#/collections)

The page **Collections** is empty: the database is running, and it knows nothing yet.

### Step 2. The ingestion

```bash
docker compose up ingest
```

The first time this builds the Python image, which takes about 8 minutes. Then the job runs and
exits. It takes the cheapest path that works:

1. the database already holds the passages: it does nothing;
2. the file `vector-db/torrevalle_chunks.snapshot` exists: it loads it, in a few seconds;
3. otherwise it parses the documents, cuts them into chunks, embeds them and stores them
   (15 to 30 minutes), then writes the snapshot.

With the repository as it comes, it takes path 2 and prints `The vector store now holds 1754 chunks`.
Now look at the database again:

[![Open the collection](https://img.shields.io/badge/Open_the_collection-246399?style=for-the-badge)](http://localhost:6333/dashboard#/collections/torrevalle_chunks)

The collection `torrevalle_chunks` holds 1754 points. Each point is a passage: its text, the
document it comes from, and two vectors, a dense one for the meaning and a sparse one for the terms.

To see how the passages are arranged in space, open the map:

[![Open the map](https://img.shields.io/badge/Open_the_map-246399?style=for-the-badge)](http://localhost:6333/dashboard#/collections/torrevalle_chunks/visualize)

The panel on the right is an editor. Click inside it, select everything (`Cmd+A` or `Ctrl+A`),
paste the request below, then click the small word **RUN** that appears above its first line.
Nothing is drawn until then.

```json
{
  "limit": 1754,
  "using": "dense",
  "color_by": { "payload": "folder" }
}
```

Passages with close vectors are close on the screen, and the colour is the folder of the document:
contracts, manuals, tickets and so on. Passages of the same kind gather together, which is what a
search by meaning relies on. If the map stays black, use Chrome, Safari or Firefox instead of a
browser embedded in an editor.

To see the database empty again, delete the collection and repeat this step:

```bash
curl -X DELETE http://localhost:6333/collections/torrevalle_chunks
```

### Step 3. The backend

```bash
docker compose up -d backend
docker compose logs -f backend
```

The first time it downloads two models, 2.3 GB each: the embedding model and the re-ranker. It is
ready when the log shows `Application startup complete`; leave the log with `Ctrl+C`.

For every question the backend does this:

1. **Retrieve.** It searches the database by meaning and by exact terms, 50 passages each, and
   fuses the two rankings into one (hybrid search). The re-ranker then reads the question together
   with each of the 50 candidates and keeps the best 10.
2. **Augment.** The 10 passages go into the prompt, each with the name of its document.
3. **Generate.** The language model answers from those passages only.

### Step 4. The web interface

```bash
docker compose up -d frontend
```

[![Open the chat](https://img.shields.io/badge/Open_the_chat-0a2843?style=for-the-badge)](http://localhost:8080)

Write a question, or pick one on the opening screen. Under every answer, **Sources** opens the 10
passages that reached the model, best first, each with its document, section, page and score.

Questions worth trying:

- *Con quanto preavviso possiamo recedere dal contratto CTR-2024-017 con Beta Idroservizi?*
  The contract says 60 days, but an addendum changed it: look at which documents the answer cites.
- *Cosa significa l'allarme E-4012 sulla pompa P7?*
- *Qual è il numero di cellulare del Direttore Generale?*
  The documents do not say: the system should answer that it does not know.

Good to know:

- **Time.** The first time, a question takes from 30 seconds to over a minute: about 20 seconds for
  the re-ranker, which runs on the CPU, and the rest for the free model.
- **A question is paid once.** Passages and answers are saved in `data/cache`: the same question,
  asked again from a new chat, is answered at once and spends no request. A saved answer names the
  date of the day it was written. Delete the folder to start again.
- **Follow-ups.** A chat keeps its thread, so *E il fornitore?* is understood. When you change
  subject, start a **New chat**.
- **Chats** are listed on the left, can be renamed and deleted, and live in your browser only.

### Step 5. The validation

Open the **Evaluation** tab at the top of the chat. It measures the system against a golden set:
20 questions, each with the passages that hold the answer and the correct answer. Five of them
cannot be answered from the documents.

The tab opens directly on the performance: the results were recorded in advance and come with the
repository (`evaluation/results.json`), so there is nothing to wait for and no request is spent.
Two things are measured.

**Retrieval, with metrics.** Did the search bring back the right passages, and near the top? No
language model is involved.

| Metric | What it says |
|---|---|
| Recall | of the passages that hold the answer, how many are among the 10 retrieved |
| Precision | of the 10 retrieved passages, how many hold the answer |
| MRR | how high the first right passage is: 1 when it is always first |

The tab also shows NDCG and Context recall, and the same numbers by type of question.

**Generation, with an LLM as a judge.** Is the answer supported by the passages it was written
from? A second call to the model splits the answer into its claims and counts the ones that the
passages state: that share is the **faithfulness**. Next to it, the tab counts the answers that
state the expected facts, and the questions without an answer where the model said so.

Every number can be measured again live, on your machine: **Measure** for retrieval (about five
minutes the first time), **Ask all** for generation, or a single question from the list
**Questions**. To record a new set of results, for example after a change to the settings:

```bash
docker compose run --rm backend python -m src.evaluation
```

It asks every question and judges every answer: about half an hour and 40 of the 50 daily
requests. A run that stops early is continued by the next one.

## 5. Reference

### Everyday commands

| Goal | Command |
|---|---|
| Start everything | `docker compose up -d` |
| Stop | `docker compose down` |
| Follow the backend log | `docker compose logs -f backend` |
| See what the ingestion did | `docker compose logs ingest` |
| Build the database again from the documents | `docker compose run --rm ingest python -m src.ingestion.main ingestion.force=true` |
| Forget the saved passages and answers | delete the folder `data/cache` |
| Delete the database and the downloaded models | `docker compose down -v` |

### Settings

Everything that can be tuned is in `src/conf/config.yaml`, with a comment on every line. The ones
that matter most:

| Setting | What it changes |
|---|---|
| `llm.model` | the language model |
| `rag.system_prompt` | the instructions given to it |
| `retrieval.mode` | the search: `dense`, `sparse` or `hybrid` |
| `retrieval.candidates` | how many passages the re-ranker reads (fewer is faster) |
| `reranking.enabled`, `reranking.top_k` | whether the re-ranker runs, and how many passages reach the model |
| `chunking.max_tokens` | the size of a chunk |
| `cache.retrieval`, `cache.answers` | what is saved for the questions already asked |

After a change, start again with `docker compose up -d --build`. A change to parsing, chunking or
embedding also needs the database to be built again; after a change to parsing, delete
`data/artifacts/parsed` first. After a change to the search, the Evaluation tab refuses the
recorded results and says so, until they are recorded again.

Three optional lines can go in `.env`:

```
# A Hugging Face cache that already holds the two models (the notebooks download them there):
# Docker uses it and does not download them again. Use the absolute path on your machine.
MODEL_CACHE=/Users/me/.cache/huggingface

# Ports on your machine for the web page and for the dashboard of the vector database.
FRONTEND_PORT=8080
QDRANT_PORT=6333
```

### When something does not work

| What you see | What to do |
|---|---|
| The chat says that the system is not ready | Normal while the backend downloads the models: the page checks again by itself. Follow `docker compose logs -f backend`. |
| The language model does not answer | Free models are often busy. The message offers **Try again**. |
| The daily allowance is used up | The message says when it starts again. Questions already asked still answer, from `data/cache`. |
| The database is empty | `docker compose logs ingest` says why; `docker compose up ingest` fills it. |
