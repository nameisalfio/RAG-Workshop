// Chats kept in the browser: each one is a list of messages, saved in localStorage.
//
// A chat:    { id, title, createdAt, updatedAt, messages }
// A message: { role: "user", text }
//            { role: "assistant", text, sources }
//            { role: "failure", text }      the question before it got no answer

const STORAGE_KEY = "rag-workshop.chats.v1";

// Longest title given to a chat automatically, in characters.
const TITLE_LENGTH = 48;

/** The state the page starts from when nothing was saved. */
function emptyState() {
  return { chats: [], activeId: null, sidebarOpen: true };
}

/**
 * Read the saved chats. Data that cannot be read is reported and left out, never guessed:
 * the second value is the problem to show, or null.
 */
export function loadState() {
  let raw;
  try {
    raw = localStorage.getItem(STORAGE_KEY);
  } catch (error) {
    return [emptyState(), `This browser does not let the page keep chats: ${error.message}`];
  }
  if (raw === null) return [emptyState(), null];
  try {
    const saved = JSON.parse(raw);
    if (!Array.isArray(saved.chats) || saved.chats.some((chat) => !Array.isArray(chat.messages))) {
      throw new Error("unexpected shape");
    }
    const activeId = saved.chats.some((chat) => chat.id === saved.activeId) ? saved.activeId : null;
    return [{ chats: saved.chats, activeId, sidebarOpen: saved.sidebarOpen !== false }, null];
  } catch (error) {
    return [emptyState(), `The saved chats could not be read and were not loaded (${error.message}).`];
  }
}

/** Save the chats; a full or blocked storage is an error for the caller to show. */
export function saveState(state) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  } catch (error) {
    throw new Error(`The chats could not be saved, delete some of them (${error.name}).`);
  }
}

/** A title for a chat, taken from its first question. */
export function titleFrom(question) {
  const line = question.replace(/\s+/g, " ").trim();
  return line.length > TITLE_LENGTH ? `${line.slice(0, TITLE_LENGTH).trimEnd()}…` : line;
}

/** Create a chat that starts with a question. */
export function newChat(question) {
  const now = Date.now();
  return { id: crypto.randomUUID(), title: titleFrom(question), createdAt: now, updatedAt: now, messages: [] };
}

/** The exchanges of a chat that got an answer, oldest first, in the form the backend expects. */
export function historyOf(chat) {
  const turns = [];
  chat.messages.forEach((message, index) => {
    const next = chat.messages[index + 1];
    if (message.role === "user" && next && next.role === "assistant") {
      turns.push({ question: message.text, answer: next.text });
    }
  });
  return turns;
}
