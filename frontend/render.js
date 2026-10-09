// Builders of the pieces of the page. Text coming from the model or from the documents is always
// added as text nodes, never as HTML.

const SVG_NAMESPACE = "http://www.w3.org/2000/svg";

/** Create an element with a class and a text. */
export function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

/** One of the icons defined in the page (see the sprite at the top of index.html). */
export function icon(name) {
  const svg = document.createElementNS(SVG_NAMESPACE, "svg");
  svg.setAttribute("class", "icon");
  svg.setAttribute("aria-hidden", "true");
  const use = document.createElementNS(SVG_NAMESPACE, "use");
  use.setAttribute("href", `#icon-${name}`);
  svg.append(use);
  return svg;
}

/** A button that shows only an icon; the label is what a screen reader announces. */
export function iconButton(name, label, className) {
  const button = element("button", `icon-button ${className}`);
  button.type = "button";
  button.setAttribute("aria-label", label);
  button.title = label;
  button.append(icon(name));
  return button;
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

/** Add a list item to a block; a new list is started when the kind of list changes. */
function appendListItem(block, list, ordered, text) {
  let target = list;
  if (!target || target.dataset.ordered !== String(ordered)) {
    target = element(ordered ? "ol" : "ul");
    target.dataset.ordered = String(ordered);
    block.append(target);
  }
  const item = element("li");
  appendInline(item, text);
  target.append(item);
  return target;
}

/**
 * The block of an answer. Models write a small subset of Markdown (headings, lists, emphasis):
 * it is turned into elements one line at a time.
 */
export function answerBlock(text) {
  const block = element("div", "answer");
  let list = null;
  for (const raw of text.split("\n")) {
    const line = raw.trim();
    const bullet = line.match(/^([-*•]|\d+[.)])\s+(.*)$/);
    if (bullet) {
      list = appendListItem(block, list, /^\d/.test(bullet[1]), bullet[2]);
      continue;
    }
    list = null;
    if (!line) continue;
    const heading = line.match(/^#{1,6}\s+(.*)$/);
    const part = element(heading ? "h3" : "p");
    appendInline(part, heading ? heading[1] : line);
    block.append(part);
  }
  return block;
}

/** The closed list of the passages that were given to the model, best first. */
export function sourcesBlock(items) {
  const list = element("ol", "passages");
  for (const item of items) {
    const summary = element("summary");
    const facts = [item.section, item.page === null ? null : `page ${item.page}`, `score ${item.score.toFixed(2)}`];
    summary.append(
      element("span", "passage-position", `${item.position}.`),
      element("span", "passage-document", item.document),
      element("span", "passage-detail", facts.filter(Boolean).join(" · ")),
    );
    const passage = element("details");
    passage.append(summary, element("pre", "passage-text", item.text));
    const row = element("li");
    row.append(passage);
    list.append(row);
  }
  const block = element("details", "sources");
  block.append(element("summary", "", `Sources: ${items.length} passages`), list);
  return block;
}
