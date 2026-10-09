// Chat interface of the RAG workshop.
//
// The list of chats lives in the browser (see chats.js). Each question goes to the backend together
// with the earlier exchanges of its chat, and comes back as an answer with the passages it was
// written from. The backend itself keeps no memory.

import { historyOf, loadState, newChat, saveState } from "./chats.js";
import { openEvaluation, setEvaluationEnabled } from "./evaluation.js";
import { answerBlock, element, iconButton, sourcesBlock } from "./render.js";

const shell = document.querySelector("#shell");
const sidebarToggle = document.querySelector("#toggle-sidebar");
const backdrop = document.querySelector("#backdrop");
const chatList = document.querySelector("#chat-list");
const chatEmpty = document.querySelector("#chat-empty");
const newChatButton = document.querySelector("#new-chat");
const thread = document.querySelector("#thread");
const welcome = document.querySelector("#welcome");
const composer = document.querySelector("#composer");
const input = document.querySelector("#question");
const send = document.querySelector("#send");
const formError = document.querySelector("#form-error");
const notice = document.querySelector("#notice");
const suggestions = document.querySelectorAll(".suggestion");
const main = document.querySelector("#main");
const evaluation = document.querySelector("#evaluation");
const viewTabs = { chat: document.querySelector("#view-chat"), evaluation: document.querySelector("#view-evaluation") };

// Pause between two checks of the backend while it is not ready.
const READY_RETRY_MS = 5000;
// Longest name a chat can be given, in characters.
const NAME_LENGTH = 80;
// Below this width the list of chats covers the conversation instead of sitting next to it.
// The style sheet switches layout at the same width.
const narrow = window.matchMedia("(max-width: 47.99rem)");

const [state, loadProblem] = loadState();
let storageProblem = loadProblem;
let readinessProblem = null;
let ready = false;
let busy = false;
let maxQuestionChars = null;
// While an answer is being written: { chatId, startedAt }.
let pending = null;
// While a chat is being renamed or deleted: { id, kind: "rename" | "delete" }.
let rowAction = null;

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

/** The chat on screen, or null on the opening screen. */
function activeChat() {
  return state.chats.find((chat) => chat.id === state.activeId) ?? null;
}

/** Show what is wrong, if anything: a backend that is not ready comes before a storage problem. */
function renderNotice() {
  const text = readinessProblem ?? storageProblem;
  notice.textContent = text ?? "";
  notice.hidden = !text;
}

/** Save the chats and report a storage that refuses them. */
function persist() {
  try {
    saveState(state);
    storageProblem = null;
  } catch (error) {
    storageProblem = error.message;
  }
  renderNotice();
}

/** Show or clear the error under the composer. */
function showFormError(text) {
  formError.textContent = text ?? "";
  formError.hidden = !text;
}

/** Enable the controls only when the backend is ready and no answer is being written. */
function updateControls() {
  const blocked = !ready || busy;
  input.disabled = !ready;
  send.disabled = blocked;
  for (const button of suggestions) button.disabled = blocked;
}

/** Let the question field grow with its text, up to the height allowed by the style sheet. */
function fitInput() {
  input.style.height = "auto";
  input.style.height = `${input.scrollHeight}px`;
}

// ---------------------------------------------------------------- views

/** Show the conversation or the evaluation panel. */
function setView(view) {
  main.dataset.view = view;
  thread.hidden = view !== "chat";
  composer.hidden = view !== "chat";
  evaluation.hidden = view !== "evaluation";
  for (const [name, tab] of Object.entries(viewTabs)) {
    tab.setAttribute("aria-pressed", String(name === view));
  }
  if (view === "evaluation") openEvaluation();
}

// ---------------------------------------------------------------- list of chats

/** Open or close the list of chats; on a wide screen the choice is remembered. */
function setSidebar(open, remember) {
  shell.dataset.sidebar = open ? "open" : "closed";
  const label = open ? "Hide the list of chats" : "Show the list of chats";
  sidebarToggle.setAttribute("aria-expanded", String(open));
  sidebarToggle.setAttribute("aria-label", label);
  sidebarToggle.title = label;
  backdrop.hidden = !(open && narrow.matches);
  if (remember && !narrow.matches) {
    state.sidebarOpen = open;
    persist();
  }
}

/** On a narrow screen the list covers the conversation, so it closes once a choice is made. */
function closeSidebarIfNarrow() {
  if (narrow.matches) setSidebar(false, false);
}

/** Leave the rename or delete mode of a row. */
function closeRowAction() {
  rowAction = null;
  renderSidebar();
}

/** Show a chat. */
function openChat(id) {
  state.activeId = id;
  rowAction = null;
  persist();
  setView("chat");
  render(false);
  closeSidebarIfNarrow();
  input.focus();
}

/** Go back to the opening screen; the new chat is created by its first question. */
function startNewChat() {
  state.activeId = null;
  rowAction = null;
  showFormError(null);
  persist();
  setView("chat");
  render(false);
  closeSidebarIfNarrow();
  input.focus();
}

/** Delete a chat for good. An answer still being written for it is dropped when it arrives. */
function deleteChat(id) {
  state.chats = state.chats.filter((chat) => chat.id !== id);
  if (state.activeId === id) state.activeId = null;
  rowAction = null;
  persist();
  render(false);
}

/** The row of a chat while it is being renamed: a field with save and cancel. */
function renameRow(row, chat) {
  const field = element("input", "chat-name");
  field.type = "text";
  field.value = chat.title;
  field.maxLength = NAME_LENGTH;
  field.setAttribute("aria-label", "Name of the chat");
  field.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeRowAction();
  });
  const save = iconButton("check", "Save the name", "chat-action");
  save.type = "submit";
  const cancel = iconButton("close", "Keep the old name", "chat-action");
  cancel.addEventListener("click", closeRowAction);
  const form = element("form", "chat-edit");
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const name = field.value.trim();
    if (!name) {
      field.focus();
      return;
    }
    chat.title = name;
    rowAction = null;
    persist();
    renderSidebar();
  });
  form.append(field, save, cancel);
  row.append(form);
  return row;
}

/** The row of a chat while its deletion waits for a confirmation. */
function deleteRow(row, chat) {
  row.classList.add("chat-row-confirm");
  const confirm = element("button", "text-button text-button-danger", "Delete");
  confirm.type = "button";
  confirm.addEventListener("click", () => deleteChat(chat.id));
  const cancel = element("button", "text-button chat-keep", "Keep");
  cancel.type = "button";
  cancel.addEventListener("click", closeRowAction);
  row.append(element("span", "chat-confirm", "Delete this chat?"), confirm, cancel);
  return row;
}

/** The row of a chat in the list. */
function chatRow(chat) {
  const row = element("li", "chat-row");
  const current = chat.id === state.activeId;
  if (current) row.classList.add("chat-row-active");
  if (rowAction && rowAction.id === chat.id) {
    return rowAction.kind === "rename" ? renameRow(row, chat) : deleteRow(row, chat);
  }
  const open = element("button", "chat-open", chat.title);
  open.type = "button";
  open.title = chat.title;
  if (current) open.setAttribute("aria-current", "true");
  open.addEventListener("click", () => openChat(chat.id));
  const rename = iconButton("pencil", `Rename the chat: ${chat.title}`, "chat-action");
  rename.addEventListener("click", () => {
    rowAction = { id: chat.id, kind: "rename" };
    renderSidebar();
    chatList.querySelector(".chat-name").select();
  });
  const remove = iconButton("trash", `Delete the chat: ${chat.title}`, "chat-action");
  remove.addEventListener("click", () => {
    rowAction = { id: chat.id, kind: "delete" };
    renderSidebar();
    chatList.querySelector(".chat-keep").focus();
  });
  row.append(open, rename, remove);
  return row;
}

/** Draw the list of chats, most recent first. */
function renderSidebar() {
  const chats = [...state.chats].sort((first, second) => second.updatedAt - first.updatedAt);
  chatEmpty.hidden = chats.length > 0;
  chatList.replaceChildren(...chats.map(chatRow));
}

// ---------------------------------------------------------------- conversation

/** A message of the conversation; the author is announced by screen readers only. */
function messageNode(kind, author, ...content) {
  const message = element("article", `message message-${kind}`);
  message.append(element("h2", "visually-hidden", author), ...content);
  return message;
}

/** The message shown while an answer is being written. */
function typingNode() {
  const dots = element("span", "dots");
  dots.append(element("span"), element("span"), element("span"));
  const typing = element("p", "typing");
  typing.append(dots, "Reading the documents", element("span", "elapsed"));
  return messageNode("assistant", "Assistant", typing);
}

/** A question that got no answer, with the way to ask it again. */
function failureNode(chat, index) {
  const retry = element("button", "text-button", "Try again");
  retry.type = "button";
  retry.addEventListener("click", () => {
    if (busy || !ready) return;
    const question = chat.messages[index - 1].text;
    chat.messages.splice(index - 1, 2);
    ask(question);
  });
  const failure = element("div", "failure");
  failure.append(element("p", "", `No answer. ${chat.messages[index].text}`), retry);
  return messageNode("assistant", "Assistant", failure);
}

/** Draw the chat on screen, or the opening screen when there is none. */
function renderThread(animateLast) {
  for (const node of thread.querySelectorAll(".message")) node.remove();
  const chat = activeChat();
  welcome.hidden = chat !== null;
  if (chat === null) return;
  chat.messages.forEach((message, index) => {
    if (message.role === "user") {
      thread.append(messageNode("user", "You", element("p", "bubble", message.text)));
    } else if (message.role === "assistant") {
      const answer = messageNode("assistant", "Assistant", answerBlock(message.text));
      answer.append(sourcesBlock(message.sources));
      thread.append(answer);
    } else {
      thread.append(failureNode(chat, index));
    }
  });
  if (pending && pending.chatId === chat.id) thread.append(typingNode());
  const last = thread.lastElementChild;
  if (animateLast) last.classList.add("message-new");
  last.scrollIntoView({ block: "end" });
}

/** Draw everything that depends on the chats. */
function render(animateLast) {
  renderSidebar();
  renderThread(animateLast);
  updateControls();
}

/** Send a question of the chat on screen and store the answer, or the reason of the failure. */
async function ask(question) {
  let chat = activeChat();
  if (chat === null) {
    chat = newChat(question);
    state.chats.push(chat);
    state.activeId = chat.id;
  }
  const history = historyOf(chat);
  chat.messages.push({ role: "user", text: question });
  chat.updatedAt = Date.now();
  busy = true;
  pending = { chatId: chat.id, startedAt: Date.now() };
  showFormError(null);
  persist();
  render(true);
  const timer = setInterval(() => {
    const elapsed = thread.querySelector(".elapsed");
    if (elapsed) elapsed.textContent = `${Math.round((Date.now() - pending.startedAt) / 1000)} s`;
  }, 1000);

  let message;
  try {
    const response = await fetch("api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, history }),
    });
    const body = await readJson(response);
    message = { role: "assistant", text: body.answer, sources: body.sources };
  } catch (error) {
    message = { role: "failure", text: error.message };
  }
  clearInterval(timer);
  busy = false;
  pending = null;
  // The chat may have been deleted while the answer was being written.
  if (state.chats.includes(chat)) {
    chat.messages.push(message);
    chat.updatedAt = Date.now();
    persist();
  }
  render(state.activeId === chat.id);
  input.focus();
}

// ---------------------------------------------------------------- events

composer.addEventListener("submit", (event) => {
  event.preventDefault();
  if (busy || !ready) return;
  const question = input.value.trim();
  if (!question) {
    showFormError("Write a question first.");
    return;
  }
  if (maxQuestionChars !== null && question.length > maxQuestionChars) {
    showFormError(`The question is longer than ${maxQuestionChars} characters.`);
    return;
  }
  input.value = "";
  fitInput();
  ask(question);
});

input.addEventListener("input", fitInput);

input.addEventListener("keydown", (event) => {
  // Enter sends the question, Shift+Enter starts a new line.
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    composer.requestSubmit();
  }
});

for (const button of suggestions) {
  button.addEventListener("click", () => {
    if (busy || !ready) return;
    ask(button.textContent.trim());
  });
}

newChatButton.addEventListener("click", startNewChat);

for (const [name, tab] of Object.entries(viewTabs)) {
  tab.addEventListener("click", () => setView(name));
}

sidebarToggle.addEventListener("click", () => {
  setSidebar(shell.dataset.sidebar !== "open", true);
});

backdrop.addEventListener("click", () => setSidebar(false, false));

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && narrow.matches && shell.dataset.sidebar === "open") {
    setSidebar(false, false);
    sidebarToggle.focus();
  }
});

narrow.addEventListener("change", () => setSidebar(!narrow.matches && state.sidebarOpen, false));

/** Check whether the backend can answer; while it cannot, say why and keep checking. */
async function checkReady() {
  let problem = null;
  try {
    const health = await readJson(await fetch("api/health"));
    maxQuestionChars = health.max_question_chars;
    if (health.chunks === 0) {
      problem = "The documents are not loaded yet. See the log of the ingestion job.";
    }
  } catch {
    problem =
      "Getting ready. The backend is still starting: the first time it downloads the embedding " +
      "model, which can take several minutes.";
  }
  readinessProblem = problem;
  ready = problem === null;
  renderNotice();
  updateControls();
  setEvaluationEnabled(ready);
  // A golden set that could not be loaded while the backend was starting is fetched now.
  if (ready && main.dataset.view === "evaluation") openEvaluation();
  if (!ready) setTimeout(checkReady, READY_RETRY_MS);
}

setSidebar(!narrow.matches && state.sidebarOpen, false);
render(false);
renderNotice();
checkReady();
