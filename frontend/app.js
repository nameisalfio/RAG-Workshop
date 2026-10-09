// Web interface of the RAG workshop: sends a question to the backend and shows the answer with its sources.

const form = document.querySelector("#ask-form");
const input = document.querySelector("#question");
const submit = document.querySelector("#submit");
const progress = document.querySelector("#progress");
const formError = document.querySelector("#form-error");
const statusLine = document.querySelector("#status");
const examples = document.querySelector("#examples");
const result = document.querySelector("#result");
const asked = document.querySelector("#asked");
const answer = document.querySelector("#answer");
const answeredBy = document.querySelector("#answered-by");
const sources = document.querySelector("#sources");

// Pause between two checks of the backend while it is not ready.
const STATUS_RETRY_MS = 5000;

let maxQuestionChars = null;

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

/** Create an element with a class and a text. */
function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

/** Add a line of model text to a node, turning **bold**, *italic* and `code` into elements. */
function appendInline(node, text) {
  const pattern = /(\*\*[^*]+\*\*|\*[^*\s][^*]*\*|`[^`]+`)/g;
  for (const part of text.split(pattern)) {
    if (part.startsWith("**") && part.endsWith("**") && part.length > 4) {
      node.append(element("strong", "", part.slice(2, -2)));
    } else if (part.startsWith("*") && part.endsWith("*") && part.length > 2) {
      node.append(element("em", "", part.slice(1, -1)));
    } else if (part.startsWith("`") && part.endsWith("`") && part.length > 2) {
      node.append(element("code", "", part.slice(1, -1)));
    } else if (part) {
      node.append(part);
    }
  }
}

/** Add a list item to the answer; a new list is started when the kind of list changes. */
function appendListItem(list, ordered, text) {
  let target = list;
  if (!target || target.dataset.ordered !== String(ordered)) {
    target = element(ordered ? "ol" : "ul");
    target.dataset.ordered = String(ordered);
    answer.append(target);
  }
  const item = element("li");
  appendInline(item, text);
  target.append(item);
  return target;
}

/**
 * Show the answer of the model. Models write a small subset of Markdown (headings, lists, emphasis):
 * it is turned into elements one line at a time and never inserted as HTML.
 */
function renderAnswer(text) {
  answer.replaceChildren();
  let list = null;
  for (const raw of text.split("\n")) {
    const line = raw.trim();
    const bullet = line.match(/^([-*\u2022]|\d+[.)])\s+(.*)$/);
    if (bullet) {
      list = appendListItem(list, /^\d/.test(bullet[1]), bullet[2]);
      continue;
    }
    list = null;
    if (!line) continue;
    const heading = line.match(/^#{1,6}\s+(.*)$/);
    const block = element(heading ? "h3" : "p");
    appendInline(block, heading ? heading[1] : line);
    answer.append(block);
  }
}

/** Show the passages that were given to the model, closed, best first. */
function renderSources(items) {
  sources.replaceChildren();
  for (const item of items) {
    const summary = element("summary");
    summary.append(
      element("span", "source-position", `${item.position}.`),
      element("span", "source-document", item.document),
    );
    const details = [item.section, item.page === null ? null : `page ${item.page}`, `score ${item.score.toFixed(3)}`];
    summary.append(element("span", "source-detail", details.filter(Boolean).join(" · ")));
    const entry = element("details");
    entry.append(summary, element("pre", "source-text", item.text));
    const row = element("li");
    row.append(entry);
    sources.append(row);
  }
}

/** Show or clear the error under the question. */
function showFormError(message) {
  formError.textContent = message ?? "";
  formError.hidden = !message;
}

/** Ask the backend and show what comes back; the button stays disabled while the model writes. */
async function ask(question) {
  showFormError(null);
  submit.disabled = true;
  const started = Date.now();
  const describe = () => {
    const seconds = Math.round((Date.now() - started) / 1000);
    progress.textContent = `Searching the documents and writing the answer: ${seconds} s`;
  };
  describe();
  const timer = setInterval(describe, 1000);
  try {
    const response = await fetch("api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const body = await readJson(response);
    asked.textContent = question;
    renderAnswer(body.answer);
    answeredBy.textContent = `Written by ${body.model} from ${body.sources.length} passages.`;
    renderSources(body.sources);
    examples.hidden = true;
    result.hidden = false;
    result.scrollIntoView({ block: "start" });
  } catch (error) {
    showFormError(`No answer: ${error.message}`);
  } finally {
    clearInterval(timer);
    progress.textContent = "";
    submit.disabled = false;
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = input.value.trim();
  if (!question) {
    showFormError("Write a question first.");
    input.focus();
    return;
  }
  if (maxQuestionChars !== null && question.length > maxQuestionChars) {
    showFormError(`The question is longer than ${maxQuestionChars} characters.`);
    input.focus();
    return;
  }
  ask(question);
});

input.addEventListener("keydown", (event) => {
  // Enter sends the question, Shift+Enter starts a new line.
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    form.requestSubmit();
  }
});

for (const button of document.querySelectorAll(".example")) {
  button.addEventListener("click", () => {
    input.value = button.textContent.trim();
    form.requestSubmit();
  });
}

/** Say whether the backend is ready and how much it knows; keep asking until it is ready. */
async function loadStatus() {
  try {
    const health = await readJson(await fetch("api/health"));
    maxQuestionChars = health.max_question_chars;
    if (health.chunks > 0) {
      delete statusLine.dataset.state;
      statusLine.textContent = `${health.chunks} passages indexed. Model: ${health.model}.`;
      return;
    }
    statusLine.dataset.state = "error";
    statusLine.textContent = "The vector store is empty: run the ingestion.";
  } catch (error) {
    statusLine.dataset.state = "error";
    statusLine.textContent =
      "Waiting for the backend. Its first start downloads the embedding model, about 2.3 GB. " +
      `Last reply: ${error.message}`;
  }
  setTimeout(loadStatus, STATUS_RETRY_MS);
}

loadStatus();
