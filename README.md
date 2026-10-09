# RAG workshop

A complete Retrieval-Augmented Generation system that answers questions about the documents of a
fictional company, Torrevalle Acque: 195 contracts, addenda, manuals, plant sheets, tickets,
policies, minutes and e-mails. You type a question in a web page, the system finds the passages
that matter, gives them to a language model and shows the answer together with its sources.

This guide takes you from an empty machine to your first answer.

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
| The knowledge base: parsing | `00_parsing.ipynb` | `src/parsing.py` |
| The knowledge base: chunking | `01_chunking.ipynb` | `src/chunking.py` |
| Embeddings | `02_embedding.ipynb` | `src/embedding.py` |
| Retrieval: vector database and similarity | `03_rag.ipynb` | `src/vector_store.py`, `src/retrieval.py` |
| Generation: the prompt | `03_rag.ipynb` | `src/rag.py`, `src/llm.py` |

Hybrid search, re-ranking and evaluation are explained in the slides and are not built here yet.

## What you need, and where it goes

Tested on macOS (Apple Silicon) with Docker Desktop. Everything runs in containers, so other
systems should behave the same way, but they have not been tried.

| What | Where it comes from | Where it goes | Your action |
|---|---|---|---|
| Docker Desktop | https://www.docker.com/products/docker-desktop | installed on your machine | install it |
| This repository | https://github.com/nameisalfio/RAG-Workshop | any folder | clone it |
| OpenRouter API key | https://openrouter.ai (free account) | the file `.env` in the project folder | create it |
| The documents | in the repository | `raw-documents/` | none |
| Ready-made vector database, 18 MB | in the repository | `vector-db/torrevalle_chunks.snapshot` | none |
| Embedding model `BAAI/bge-m3`, 2.3 GB | Hugging Face | a Docker volume | none, it is downloaded at the first start |

You also need about 10 GB of free disk space and 8 GB of memory assigned to Docker
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
4. **The backend starts** and downloads the embedding model, 2.3 GB. How long this takes depends
   only on your connection. It is ready when the log shows `Application startup complete`.
5. **The web interface starts.**

Later starts skip all of this and take a few seconds: use `docker compose up`.

### 5. Ask a question

Open http://localhost:8080

Under the title, a line tells you the state of the system:

- `1754 passages indexed. Model: ...` means that everything is ready;
- `Waiting for the backend` means that the backend is still starting, usually because it is
  downloading the model. The page checks again by itself every five seconds.

Type a question, or press one of the three examples. An answer takes from a few seconds to about
a minute. Below the answer you find:

- the name of the model that wrote it;
- **Passages given to the model**: the ten passages that were retrieved, best first, each with its
  document, section, page and similarity score. Open one to read its text.

Questions worth trying:

- *Con quanto preavviso possiamo recedere dal contratto CTR-2024-017 con Beta Idroservizi?*
  The contract says 60 days, but an addendum changed it: look at which documents the answer cites.
- *Cosa significa l'allarme E-4012 sulla pompa P7?*
- *Qual è il numero di cellulare del Direttore Generale?*
  The documents do not say: the system should answer that it does not know.

## Everyday commands

| Goal | Command |
|---|---|
| Start again | `docker compose up` |
| Start in the background | `docker compose up -d` |
| Stop | `Ctrl+C`, or `docker compose down` |
| Follow the backend log | `docker compose logs -f backend` |
| See why the database is empty | `docker compose logs ingest` |
| Build the vector database again from the documents | `docker compose run --rm ingest python -m src.ingest ingestion.force=true` |
| Delete the database and the downloaded model | `docker compose down -v` |

The vector database has a dashboard at http://localhost:6333/dashboard, where you can look at the
stored passages and their vectors.

### Settings

Everything that can be tuned is in `src/conf/config.yaml`: the language model (`llm.model`), the
instructions given to it (`rag.system_prompt`), how many passages are retrieved (`reranking.top_k`),
the size of the chunks (`chunking.max_tokens`) and so on. After a change, start with
`docker compose up --build`. A change to parsing, chunking or embedding also needs the database to
be built again (see the table above); after a change to parsing, delete `data/artifacts/parsed`
first, otherwise the text saved by the previous run is used.

Three optional lines can be added to `.env`:

```
# Reuse a Hugging Face cache that already holds the embedding model, instead of downloading it
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
with: same embedding model and same chunking. Building the database again rewrites the file, and
git will show it as modified.

## The notebooks (optional)

The folder `notebooks/` shows how each piece was chosen, one step at a time. They run outside
Docker, in a conda environment:

```bash
conda env create -f environment.yml
conda activate rag-workshop
python -m ipykernel install --user --name rag-workshop --display-name "Python (rag-workshop)"
python -m src.parsing
```

The last command reads the 195 documents and saves their text in `data/artifacts/parsed`. It takes
about 5 minutes and the notebooks from 01 onwards need it. Then open the notebooks in VS Code, or
in any other Jupyter client, and choose the kernel **Python (rag-workshop)**.

| Notebook | What it shows |
|---|---|
| `00_parsing.ipynb` | five PDF parsers on real pages, and why Docling was chosen |
| `01_chunking.ipynb` | three ways to cut a document, and why we cut on its sections |
| `02_embedding.ipynb` | a text becomes a point in space; cosine similarity |
| `03_rag.ipynb` | from a model alone to retrieve, augment, generate; it needs the key in `.env` |

After running them, the embedding model is in your Hugging Face cache (`~/.cache/huggingface`):
set `MODEL_CACHE` to that path and Docker will not download it a second time.

## What is in the repository

```
RAG-Workshop/
├── docs/                  slides of the workshop: the theory
├── notebooks/             the four notebooks: how each piece was chosen
├── docker-compose.yml     the four services and how they depend on each other
├── Dockerfile             image of the Python services (backend and ingestion)
├── .env.example           template of the file that holds your key
├── frontend/              web interface: static page served by nginx
├── src/                   Python package
│   ├── conf/config.yaml   every setting
│   ├── parsing.py         files to text
│   ├── chunking.py        text to chunks
│   ├── embedding.py       chunks to vectors
│   ├── vector_store.py    vector database (Qdrant)
│   ├── retrieval.py       in-memory index used by the notebooks
│   ├── llm.py             client of the language model
│   ├── rag.py             retrieve, augment, generate
│   ├── api.py             backend (HTTP API)
│   └── ingest.py          job that fills the vector database
├── raw-documents/         the 195 documents
├── vector-db/             ready-made vector database (snapshot)
├── tests/                 automated tests
└── data/                  created at run time, ignored by git
```

## When something goes wrong

**`env file .env not found` when starting.**
Step 3 was skipped: create `.env` from `.env.example`.

**The page says `Waiting for the backend` for a long time.**
The backend is downloading the 2.3 GB embedding model. Run `docker compose logs -f backend` and
wait for `Application startup complete`. If you already have the model on your machine, set
`MODEL_CACHE` (see Settings).

**The page says `The vector store is empty`.**
The ingestion job failed. Read its log with `docker compose logs ingest`.

**`No answer: The language model did not answer` under the question.**
Free models are shared by many people and are sometimes busy or rate-limited. The system already
tries a second model by itself; when both fail, wait a little and ask again. If it keeps happening,
pick another free model from https://openrouter.ai/models, write its name in `llm.model` and start
with `docker compose up --build`.

**`HTTP 401` in the error message.**
The key in `.env` is wrong or was deleted. Create a new one (step 3), then run
`docker compose up -d --force-recreate backend`.

**`port is already allocated` when starting.**
Another program uses port 8080 or 6333. Choose other ports with `FRONTEND_PORT` and `QDRANT_PORT`
in `.env`.

**A container is killed, or the machine becomes very slow.**
Docker has too little memory. Assign 8 GB to it in Docker Desktop, Settings, Resources.
