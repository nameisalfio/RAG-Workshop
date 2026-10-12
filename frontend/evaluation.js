// Evaluation panel: measures the system against the golden set kept by the backend.
//
// The panel opens on the results recorded by `python -m src.evaluation`, so that nothing has to be
// waited for. Everything can also be run live: retrieval in one request, without the language model;
// generation one question at a time, so that progress is visible and a failure costs one question
// only. A live run replaces the recorded numbers on screen until the page is loaded again.

import { answerBlock, element } from "./render.js";

const problem = document.querySelector("#evaluation-problem");
const lede = document.querySelector("#evaluation-lede");
const recorded = document.querySelector("#evaluation-recorded");
const questionsCount = document.querySelector("#questions-count");
const runRetrieval = document.querySelector("#run-retrieval");
const retrievalSearch = document.querySelector("#retrieval-search");
const retrievalMetrics = document.querySelector("#retrieval-metrics");
const retrievalTypes = document.querySelector("#retrieval-types");
const runGeneration = document.querySelector("#run-generation");
const stopGeneration = document.querySelector("#stop-generation");
const judge = document.querySelector("#judge");
const progress = document.querySelector("#generation-progress");
const generationMetrics = document.querySelector("#generation-metrics");
const questionList = document.querySelector("#question-list");

const UNANSWERABLE = "unanswerable";

// The golden set, as sent by the backend: { top_k, questions }.
let golden = null;
// The last retrieval report, or null.
let retrieval = null;
// For each question that was asked: the row sent by the backend, or { error }.
let generation = {};
let enabled = false;
let measuring = false;
let asking = false;
let stopRequested = false;
// The question being asked right now, if any.
let currentId = null;
// Questions whose details are open, so that a redraw does not close them.
const openIds = new Set();

/** Read the body of a response; a failed request becomes an Error with the reason given by the backend. */
async function readJson(response) {
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = body && typeof body.detail === "string" ? body.detail : `HTTP ${response.status}`;
    throw new Error(detail);
  }
  if (body === null) {
    throw new Error("The backend returned a reply that is not JSON.");
  }
  return body;
}

/** Show or clear the problem line of the panel. */
function showProblem(text) {
  problem.textContent = text ?? "";
  problem.hidden = !text;
}

/** Say when the numbers on screen were recorded, or that they come from a run made just now. */
function showOrigin(createdAt) {
  if (createdAt === null) {
    recorded.textContent = "Results of a run made now, in this page.";
  } else {
    const when = new Date(createdAt).toLocaleDateString("en-GB", { day: "numeric", month: "long", year: "numeric" });
    recorded.textContent = `Results recorded on ${when}. Measure or ask again to run them live.`;
  }
  recorded.hidden = false;
}

/** Load the recorded results; having none is normal, anything else is reported. */
async function loadRecorded() {
  const response = await fetch("api/evaluation/results");
  if (response.status === 404) return;
  const results = await readJson(response);
  retrieval = results.retrieval;
  generation = { ...results.generation };
  for (const [id, message] of Object.entries(results.errors)) generation[id] = { error: message };
  showOrigin(results.created_at);
}

/** A score between 0 and 1, written with two decimals. */
function score(value) {
  return value.toFixed(2);
}

/** A small label that says a state in words, with a colour that repeats it. */
function chip(text, kind) {
  return element("span", `chip chip-${kind}`, text);
}

/** One tile of a metric: name, value, a bar and a line that says what it measures. */
function metricTile(name, value, meaning) {
  const tile = element("div", "metric");
  const bar = element("progress", "bar");
  bar.max = 1;
  bar.value = value ?? 0;
  bar.setAttribute("aria-hidden", "true");
  tile.append(
    element("dt", "metric-name", name),
    element("dd", "metric-value", value === null ? "n/a" : score(value)),
    element("dd", "metric-bar"),
    element("dd", "metric-meaning", meaning),
  );
  tile.querySelector(".metric-bar").append(bar);
  return tile;
}

function renderRetrieval() {
  retrievalMetrics.replaceChildren();
  retrievalTypes.replaceChildren();
  retrievalSearch.hidden = retrieval === null;
  if (retrieval === null) {
    retrievalMetrics.append(element("p", "meta", "Not measured yet."));
    return;
  }
  retrievalSearch.textContent = `Measured with: ${retrieval.search}.`;
  const k = retrieval.top_k;
  const s = retrieval.summary;
  retrievalMetrics.append(
    metricTile(`Recall@${k}`, s.recall, "golden passages found among the retrieved ones"),
    metricTile(`Precision@${k}`, s.precision, "retrieved passages that are golden ones"),
    metricTile("MRR", s.mrr, "how high the first golden passage ranks"),
    metricTile(`NDCG@${k}`, s.ndcg, "quality of the whole ranking"),
    metricTile("Context recall", s.context_recall, "facts of the correct answer present in the retrieved text"),
  );
  const table = element("table", "types");
  const head = element("tr");
  for (const title of ["Type of question", "Questions", "Recall", "MRR", "Context recall"]) {
    head.append(element("th", "", title));
  }
  table.append(head);
  for (const group of retrieval.by_type) {
    const row = element("tr");
    row.append(
      element("td", "", group.type),
      element("td", "number", String(group.questions)),
      element("td", "number", score(group.recall)),
      element("td", "number", score(group.mrr)),
      element("td", "number", score(group.context_recall)),
    );
    table.append(row);
  }
  retrievalTypes.append(table);
}

function renderGeneration() {
  generationMetrics.replaceChildren();
  const rows = golden.questions.map((question) => [question, generation[question.id]]);
  const answered = rows.filter(([, row]) => row && !row.error);
  const failed = rows.filter(([, row]) => row && row.error).length;
  if (answered.length === 0 && failed === 0) {
    generationMetrics.append(element("p", "meta", "Not run yet."));
    return;
  }
  const share = (items) => (items.length ? items.filter(([, row]) => row.passed).length / items.length : null);
  const answerable = answered.filter(([question]) => question.type !== UNANSWERABLE);
  const unanswerable = answered.filter(([question]) => question.type === UNANSWERABLE);
  const judged = answered.filter(([, row]) => row.faithfulness);
  const faithfulness = judged.length
    ? judged.reduce((sum, [, row]) => sum + row.faithfulness.score, 0) / judged.length
    : null;
  generationMetrics.append(
    metricTile("Correct answers", share(answerable), `answers that state every expected fact (${answerable.length} asked)`),
    metricTile("Abstentions", share(unanswerable), `questions without an answer where the model said so (${unanswerable.length} asked)`),
    metricTile("Faithfulness", faithfulness, `claims supported by the passages, according to the judge (${judged.length} judged)`),
  );
  if (failed) generationMetrics.append(element("p", "meta", `${failed} questions got no answer from the model.`));
}

/** The chip that sums up the retrieval of a question. */
function retrievalChip(question) {
  if (question.type === UNANSWERABLE) return chip("no passage expected", "idle");
  const row = retrieval?.rows.find((item) => item.id === question.id);
  if (!row) return chip("retrieval not measured", "idle");
  return row.first_relevant ? chip(`found at ${row.first_relevant}`, "good") : chip("missed", "bad");
}

/** The chip that sums up the answer to a question. */
function generationChip(question) {
  if (currentId === question.id) return chip("asking", "idle");
  const row = generation[question.id];
  if (!row) return chip("not asked", "idle");
  if (row.error) return chip("no answer", "bad");
  if (question.type === UNANSWERABLE) {
    return row.passed ? chip("said it does not know", "good") : chip("answered anyway", "bad");
  }
  return row.passed ? chip("correct", "good") : chip("facts missing", "bad");
}

/** What is known about a question, shown when its row is opened. */
function questionDetails(question) {
  const box = element("div", "question-details");
  box.append(element("h3", "", "Correct answer"), element("p", "", question.answer));

  const row = retrieval?.rows.find((item) => item.id === question.id);
  if (question.context.length) {
    box.append(element("h3", "", "Golden passages"));
    const list = element("ul", "plain-list");
    for (const passage of question.context) {
      const name = passage.section ? `${passage.document} · ${passage.section}` : passage.document;
      const item = element("li");
      item.append(element("span", "passage-document", name));
      if (row) {
        const found = row.retrieved.some(
          (hit) => hit.relevant && hit.document.startsWith(passage.document) && hit.section.startsWith(passage.section),
        );
        item.append(" ", found ? chip("retrieved", "good") : chip("not retrieved", "bad"));
      }
      list.append(item);
    }
    box.append(list);
  }
  if (row) {
    box.append(element("h3", "", `Retrieved passages (recall ${score(row.recall)}, context recall ${score(row.context_recall)})`));
    const list = element("ol", "plain-list");
    for (const hit of row.retrieved) {
      const name = hit.section ? `${hit.document} · ${hit.section}` : hit.document;
      const item = element("li");
      item.append(element("span", "passage-document", name));
      if (hit.relevant) item.append(" ", chip("golden", "good"));
      list.append(item);
    }
    box.append(list);
  }

  const answer = generation[question.id];
  if (answer && answer.error) {
    box.append(element("h3", "", "Answer of the model"), element("p", "failure", `No answer. ${answer.error}`));
  } else if (answer) {
    box.append(element("h3", "", "Answer of the model"), answerBlock(answer.answer));
    if (answer.facts.length) {
      const facts = element("ul", "plain-list");
      for (const fact of answer.facts) {
        const item = element("li");
        item.append(fact.wordings[0], " ", fact.found ? chip("stated", "good") : chip("missing", "bad"));
        facts.append(item);
      }
      box.append(element("h3", "", "Expected facts"), facts);
    }
    if (answer.faithfulness) {
      const verdict = answer.faithfulness;
      box.append(
        element("p", "meta", `Faithfulness ${score(verdict.score)}: the judge found ${verdict.supported} of ${verdict.claims} claims supported by the passages.`),
      );
    } else if (answer.faithfulness_error) {
      box.append(element("p", "meta", `The judge gave no verdict. ${answer.faithfulness_error}`));
    }
  }

  const ask = element("button", "text-button", answer ? "Ask again" : "Ask this question");
  ask.type = "button";
  ask.disabled = !enabled || asking;
  ask.addEventListener("click", () => askQuestions([question.id]));
  box.append(ask);
  return box;
}

function renderQuestions() {
  questionList.replaceChildren();
  for (const question of golden.questions) {
    const summary = element("summary", "question-summary");
    const chips = element("span", "chips");
    chips.append(chip(question.type, "type"), retrievalChip(question), generationChip(question));
    summary.append(element("span", "question-text", question.question), chips);
    const details = element("details", "question");
    details.open = openIds.has(question.id);
    details.addEventListener("toggle", () => {
      if (details.open) openIds.add(question.id);
      else openIds.delete(question.id);
    });
    details.append(summary, questionDetails(question));
    const item = element("li");
    item.append(details);
    questionList.append(item);
  }
}

/** Enable what can run now and say how many model calls a full run costs. */
function updateControls() {
  const calls = golden ? golden.questions.length * (judge.checked ? 2 : 1) : 0;
  runGeneration.textContent = `Ask all (${calls} model calls)`;
  runRetrieval.disabled = !enabled || !golden || measuring;
  runGeneration.disabled = !enabled || !golden || asking;
  runGeneration.hidden = asking;
  stopGeneration.hidden = !asking;
  judge.disabled = asking;
}

function render() {
  if (golden === null) return;
  const answerable = golden.questions.filter((question) => question.type !== UNANSWERABLE).length;
  lede.textContent =
    `A golden set of ${golden.questions.length} questions, each with the passages that hold the answer and the ` +
    `correct answer; ${golden.questions.length - answerable} of them cannot be answered from the documents.`;
  questionsCount.textContent = `${golden.questions.length} questions`;
  renderRetrieval();
  renderGeneration();
  renderQuestions();
  updateControls();
}

/** Measure retrieval on the whole golden set. */
async function measureRetrieval() {
  measuring = true;
  showProblem(null);
  updateControls();
  try {
    retrieval = await readJson(await fetch("api/evaluation/retrieval", { method: "POST" }));
    showOrigin(null);
  } catch (error) {
    showProblem(`Retrieval could not be measured. ${error.message}`);
  }
  measuring = false;
  render();
}

/** Ask the model the given questions, one after the other, until done or stopped. */
async function askQuestions(ids) {
  asking = true;
  stopRequested = false;
  showProblem(null);
  let done = 0;
  for (const id of ids) {
    if (stopRequested) break;
    currentId = id;
    progress.textContent = `Asking question ${done + 1} of ${ids.length}.`;
    render();
    try {
      const response = await fetch("api/evaluation/generation", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id, judge: judge.checked }),
      });
      generation[id] = await readJson(response);
    } catch (error) {
      generation[id] = { error: error.message };
    }
    showOrigin(null);
    done += 1;
  }
  progress.textContent = stopRequested ? `Stopped after ${done} of ${ids.length} questions.` : "";
  currentId = null;
  asking = false;
  render();
}

runRetrieval.addEventListener("click", measureRetrieval);
runGeneration.addEventListener("click", () => askQuestions(golden.questions.map((question) => question.id)));
stopGeneration.addEventListener("click", () => {
  stopRequested = true;
  progress.textContent = "Stopping after the question in progress.";
});
judge.addEventListener("change", updateControls);

/** Called when the panel is shown: the golden set is fetched the first time. */
export async function openEvaluation() {
  if (golden !== null) {
    render();
    return;
  }
  try {
    golden = await readJson(await fetch("api/evaluation/golden"));
  } catch (error) {
    showProblem(`The golden set could not be loaded. ${error.message}`);
    return;
  }
  try {
    await loadRecorded();
  } catch (error) {
    showProblem(`The recorded results could not be loaded. ${error.message}`);
  }
  render();
}

/** Called when the backend becomes ready or stops being so. */
export function setEvaluationEnabled(value) {
  enabled = value;
  if (golden !== null) render();
}
