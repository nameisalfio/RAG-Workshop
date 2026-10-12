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

This guide takes you from an empty machine to your first answer.

## What is in the repository

```
RAG-Workshop/
├── docs/                  slides of the workshop: the theory
├── notebooks/             the five notebooks: how each piece was chosen
├── docker-compose.yml     the four services and how they depend on each other
├── Dockerfile             image of the Python services (backend and ingestion)
├── .env.example           template of the file that holds your key
├── frontend/              web interface: static page served by nginx
├── src/                   Python package
│   ├── conf/config.yaml   every setting
│   ├── config.py          the settings as typed objects, and the key read from .env
│   ├── ingestion/         offline: from the documents to the vector database
│   │   ├── main.py        the job that fills the database (python -m src.ingestion.main)
│   │   ├── parsing.py     files to text
│   │   └── chunking.py    text to chunks
│   ├── retrieval/         online: from a question to the passages that answer it
│   │   ├── embedding.py   text to vectors, dense and sparse
│   │   ├── vector_store.py  vector database (Qdrant): dense, sparse and hybrid search
│   │   └── reranking.py   re-ranker (cross-encoder)
│   ├── llm.py             client of the language model
│   ├── rag.py             retrieve, augment, generate
│   ├── cache.py           passages and answers saved for the questions already asked
│   ├── evaluation.py      metrics against the golden set (python -m src.evaluation)
│   └── api.py             backend (HTTP API)
├── raw-documents/         the 195 documents
├── vector-db/             ready-made vector database (snapshot)
├── evaluation/            golden set (questions with their correct answers) and recorded results
├── tests/                 automated tests
└── data/                  created at run time, ignored by git
```

## Start with the theory

The slides of the workshop are in [`docs/RAG-workshop.html`](docs/RAG-workshop.html). Open the file
in a browser and move with the arrow keys; the page loads its libraries from the internet, so you
need a connection.

They explain every idea that this repository puts into practice, in nine parts: the problem (why a
language model alone is not enough), the architecture of a RAG system, the knowledge base (parsing
and chunking), embeddings, retrieval, generation, evaluation, frameworks, and what lies beyond a
basic RAG.

Each part of the theory has its counterpart here:

| In the slides | In the notebooks | In the code |
|---|---|---|
| The knowledge base: parsing | `00_parsing.ipynb` | `src/ingestion/parsing.py` |
| The knowledge base: chunking | `01_chunking.ipynb` | `src/ingestion/chunking.py` |
| Embeddings | `02_embedding.ipynb` | `src/retrieval/embedding.py` |
| Retrieval: vector database, hybrid search, re-ranking | `03_reranking.ipynb` | `src/retrieval/vector_store.py`, `src/retrieval/reranking.py` |
| Generation: the prompt | `04_rag.ipynb` | `src/rag.py`, `src/llm.py` |
| Evaluation | the Evaluation panel of the web page | `src/evaluation.py`, `evaluation/golden_set.yaml` |

## What you need, and where it goes

Tested on macOS (Apple Silicon) with Docker Desktop. Everything runs in containers, so other
systems should behave the same way, but they have not been tried.

| What | Where it comes from | Where it goes | Your action |
|---|---|---|---|
| Docker Desktop | https://www.docker.com/products/docker-desktop | installed on your machine | install it |
| This repository | https://github.com/nameisalfio/RAG-Workshop | any folder | clone it |
| OpenRouter API key | https://openrouter.ai (free account) | the file `.env` in the project folder | create it |
| The documents | in the repository | `raw-documents/` | none |
| Ready-made vector database, 21 MB | in the repository | `vector-db/torrevalle_chunks.snapshot` | none |
| Embedding model `BAAI/bge-m3`, 2.3 GB | Hugging Face | a Docker volume | none, it is downloaded at the first start |
| Re-ranker model `BAAI/bge-reranker-v2-m3`, 2.3 GB | Hugging Face | the same Docker volume | none, it is downloaded at the first start |

You also need about 12 GB of free disk space and 8 GB of memory assigned to Docker
(Docker Desktop, Settings, Resources).

## From zero to the first answer

### 1. Install Docker

Install Docker Desktop, start it and check that it answers:

```bash
docker compose version
```

The command must print a version. If it says that `compose` is not a command, your Docker is too old.

### 2. Get the code

```bash
git clone https://github.com/nameisalfio/RAG-Workshop.git
cd RAG-Workshop
```

Every command below is run from this folder.

### 3. Create the API key

The language model is reached through OpenRouter, which offers some models for free.

1. Open https://openrouter.ai and sign in (a GitHub or Google account is enough).
2. Open the **API Keys** page of your account and press **New Key**.
3. Give it a name. You can leave the credit limit at zero: the models used here cost nothing.
4. Copy the key. It is shown only once.

An account that never bought credits can make 50 requests a day to the free models; the count
starts again at midnight UTC. That is enough to follow the workshop. When it is used up, the page
says so and tells when it resets.

Then create the file `.env` from the template:

```bash
cp .env.example .env
```

Open `.env` in an editor and complete the line with your key:

```
OPENROUTER_API_KEY=sk-or-v1-...
```

The file `.env` stays on your machine: git ignores it. Never put the key anywhere else.

### 4. Start everything

```bash
docker compose up --build
```

The first start is the slow one. In order:

1. **Images are built**: about 8 minutes. The Python image is 4 GB.
2. **The vector database starts** (Qdrant).
3. **The ingestion job runs and exits.** It loads the ready-made database that comes with the
   repository, which takes a few seconds, and prints `The vector store now holds 1754 chunks`.
4. **The backend starts** and downloads the embedding model and the re-ranker model, 2.3 GB each.
   How long this takes depends only on your connection. It is ready when the log shows
   `Application startup complete`.
5. **The web interface starts.**

Later starts skip all of this and take a few seconds: use `docker compose up`.

### 5. Ask a question

Open http://localhost:8080

The page is a chat. Write a question in the box at the bottom and press **Send** (or Enter), or
pick one of the three questions on the opening screen. An answer takes from a few seconds to about
a minute; if the language model does not answer, the message offers **Try again**.

Under every answer, **Sources** opens the ten passages that reached the model, best first, each
with its document, section, page and the relevance score given by the re-ranker. Open a passage
to read its text.

Behind every question there are three steps: a search by meaning and one by exact terms, each
returning 50 passages; the fusion of the two rankings into one (hybrid search); and a re-ranker
that reads the question together with each of the 50 candidates and keeps the best ten. The
re-ranker runs on the CPU: inside Docker it adds about 20 seconds to every answer. With
`retrieval.candidates: 30` in `src/conf/config.yaml` it adds about 12, and finds a little less.

A question is paid once. The passages found and the answer written are saved in `data/cache`,
and the same question, asked again from a new chat, is answered at once without calling the
re-ranker or the language model. A saved answer is the one written the first time: it names the
date of that day. To have every answer written again, set `cache.answers: false` in
`src/conf/config.yaml`, or delete the folder `data/cache`.

A chat keeps its thread: a follow-up such as *E il fornitore?* is searched together with the
question before it, and the model receives the last exchanges. When you change subject, start a
**New chat**, otherwise the previous question still weighs on the search.

The panel on the left lists your chats. Each one can be reopened, renamed (pencil) or deleted
(bin), and the button at the top left of the conversation hides or shows the panel. Chats are
saved in your browser only: another browser, or a private window, starts empty.

While the system is not ready, a notice says so under the top bar and the box stays disabled.
This is normal at the first start, when the backend is downloading the two models; the page
checks again by itself every five seconds.

Questions worth trying:

- *Con quanto preavviso possiamo recedere dal contratto CTR-2024-017 con Beta Idroservizi?*
  The contract says 60 days, but an addendum changed it: look at which documents the answer cites.
- *Cosa significa l'allarme E-4012 sulla pompa P7?*
- *Qual è il numero di cellulare del Direttore Generale?*
  The documents do not say: the system should answer that it does not know.

### 6. Measure the system

The **Evaluation** tab, at the top of the page, measures the system against a golden set: 20
questions, each with the passages that hold the answer and the correct answer. Five of them
cannot be answered from the documents, to check that the system says so.

The panel opens directly on the performance of the system: the results were recorded in advance
and come with the repository, in the file `evaluation/results.json`, so there is nothing to wait
for and no request is spent. The line under the title says when they were recorded. Every number
can also be measured again live, on your machine and with your key:

- **Retrieval**: Recall, Precision, MRR and NDCG over the ten retrieved passages, and Context
  recall (the share of the facts of the correct answer that the retrieved text contains), with a
  table by type of question, and the line above them says which search produced them. **Measure**
  computes them again without calling the language model. The first time the re-ranker reads
  every candidate of every question, which takes about five minutes; after that the passages
  are saved and the numbers come back at once.
- **Generation**: the share of answers that state the expected facts, the share of unanswerable
  questions where the model said so, and faithfulness (how many claims of the answers the passages
  support, according to a second model call that acts as a judge). **Ask all** runs it again live,
  one question at a time: with a free model it takes several minutes and uses part of the daily
  allowance, and **Stop** ends it after the question in progress.
- **Questions**: closed at first. Open the section, then a question, to see its golden passages,
  what was retrieved and the answer; a single question can be asked on its own.

A live run replaces the numbers on screen until the page is loaded again. To record new results,
for example after a change to `src/conf/config.yaml`, run:

```bash
docker compose run --rm backend python -m src.evaluation
```

It asks every question (two model calls each, the second one for the judge), retries the ones the
model does not answer, and rewrites `evaluation/results.json`. With a free model it takes about
half an hour and 40 of the 50 daily requests: `evaluation.judge=false` at the end of the command
halves both, and leaves faithfulness out.

Each answer is saved as soon as it arrives. A run that stops early, because the daily allowance
ended or the model stayed busy, is continued by the next one, which asks only the questions still
without an answer. Add `evaluation.fresh=true` to start again from the first question and to have
the answers saved in `data/cache` written again; without it, a question that was already asked,
here or in the chat, is not sent to the model a second time.

Recorded results carry the search that produced them. After a change to the search
(`retrieval.mode`, `retrieval.candidates`, the re-ranker, `reranking.top_k`) the panel refuses the
old ones and says so, until the command above is run again.

The golden set is the file `evaluation/golden_set.yaml`. After adding or changing questions,
record the results again and restart the backend with `docker compose restart backend`.

## Everyday commands

| Goal | Command |
|---|---|
| Start again | `docker compose up` |
| Start in the background | `docker compose up -d` |
| Stop | `Ctrl+C`, or `docker compose down` |
| Follow the backend log | `docker compose logs -f backend` |
| See why the database is empty | `docker compose logs ingest` |
| Build the vector database again from the documents | `docker compose run --rm ingest python -m src.ingestion.main ingestion.force=true` |
| Forget the saved passages and answers | delete the folder `data/cache` |
| Delete the database and the downloaded models | `docker compose down -v` |

### Look inside the vector database, before and after the ingestion

Qdrant has a dashboard at http://localhost:6333/dashboard. It can show the database empty, then
full, then as a map of its points.

1. **Before.** Delete the collection:

   ```bash
   curl -X DELETE http://localhost:6333/collections/torrevalle_chunks
   ```

   The page **Collections** of the dashboard is now empty, and the chat cannot answer.

2. **Ingestion.** Fill it again; this loads the ready-made database and takes a few seconds:

   ```bash
   docker compose run --rm ingest
   ```

3. **After.** **Collections** shows `torrevalle_chunks` with 1754 points. Open it: the tab
   **Points** lists the passages, each with its text, its document and its two vectors.

4. **The map.** Open the tab **Visualize**. The panel on the right is an editor: click inside it,
   select everything (`Cmd+A`, or `Ctrl+A`), paste the request below, then click the small word
   **RUN** that appears above its first line. Nothing is drawn until then.

   ```json
   {
     "limit": 1754,
     "using": "dense",
     "color_by": { "payload": "folder" }
   }
   ```

   Every point is a passage. Qdrant places them so that passages with close vectors are close on
   the screen, and the colour is the folder the document comes from: contracts, manuals, tickets
   and so on. Passages of the same kind gather together, which is what a search by meaning relies
   on. Click a point to see its passage and its nearest neighbours.

   The request that the page proposes at first draws nothing for this collection: every point
   has two vectors, and `"using": "dense"` says which one to use. If the map stays black, open
   the page in Chrome, Safari or Firefox instead of a browser embedded in an editor.

The passages and the answers saved for the questions already asked are kept through these steps.

### Settings

Everything that can be tuned is in `src/conf/config.yaml`: the language model (`llm.model`), the
instructions given to it (`rag.system_prompt`), the kind of search (`retrieval.mode`: `dense`,
`sparse` or `hybrid`), how many candidates it returns (`retrieval.candidates`), whether the
re-ranker runs (`reranking.enabled`), how many passages reach the model (`reranking.top_k`), the
size of the chunks (`chunking.max_tokens`), what is saved for the questions already asked
(`cache.retrieval`, `cache.answers`) and so on. After a change, start with
`docker compose up --build`. A change to parsing, chunking or embedding also needs the database to
be built again (see the table above); after a change to parsing, delete `data/artifacts/parsed`
first, otherwise the text saved by the previous run is used.

Three optional lines can be added to `.env`:

```
# Reuse a Hugging Face cache that already holds the two models, instead of downloading them
# again inside Docker. Use the absolute path of the cache on your machine.
MODEL_CACHE=/Users/me/.cache/huggingface

# Ports on your machine for the web interface and for the dashboard of the vector database.
FRONTEND_PORT=8080
QDRANT_PORT=6333
```

## How the vector database is filled

The ingestion job takes the cheapest path that works:

1. the database already holds passages: it does nothing;
2. the file `vector-db/torrevalle_chunks.snapshot` exists: it loads it;
3. otherwise it parses the documents in `raw-documents/`, cuts them into chunks, embeds them,
   stores them, and writes the snapshot file. This takes 15 to 30 minutes on a recent laptop.

The snapshot in the repository was produced by path 3. It only fits the configuration it was built
with: same embedding model and same chunking. It holds a dense and a sparse vector for every
passage, so the kind of search can change without building it again. Building the database
again rewrites the file, and git will show it as modified.

## The notebooks (optional)

The folder `notebooks/` shows how each piece was chosen, one step at a time. They run outside
Docker, in a conda environment:

```bash
conda env create -f environment.yml
conda activate rag-workshop
python -m ipykernel install --user --name rag-workshop --display-name "Python (rag-workshop)"
python -m src.ingestion.parsing
```

The last command reads the 195 documents and saves their text in `data/artifacts/parsed`. It takes
about 5 minutes and the notebooks from 01 onwards need it. Then open the notebooks in VS Code, or
in any other Jupyter client, and choose the kernel **Python (rag-workshop)**.

| Notebook | What it shows |
|---|---|
| `00_parsing.ipynb` | five PDF parsers on real pages, and why Docling was chosen |
| `01_chunking.ipynb` | three ways to cut a document, and why we cut on its sections |
| `02_embedding.ipynb` | a text becomes a point in space; cosine similarity |
| `03_reranking.ipynb` | search by meaning, by terms and by both, then the re-ranker, measured on the golden set; it needs the vector database (`docker compose up -d`) |
| `04_rag.ipynb` | from a model alone to retrieve, augment, generate, and what the re-ranker adds; it needs the key in `.env` and the vector database |

After running them, the embedding model and the re-ranker model are in your Hugging Face cache
(`~/.cache/huggingface`): set `MODEL_CACHE` to that path and Docker will not download them a
second time.

