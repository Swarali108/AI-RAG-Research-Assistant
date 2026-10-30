"use strict";

/* Researchly front end: vanilla JS, no build step. Sections:
   utilities · menus & dialogs · layout · profile · chats · messages & streaming
   · modes · sources · insights & inspector · export & saved research · init */

// ===================================================================== utilities

const $ = selector => document.querySelector(selector);

function h(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === false || value === null || value === undefined) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

function icon(name, extra = "") {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("class", ("icon " + extra).trim());
  svg.setAttribute("aria-hidden", "true");
  const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
  use.setAttribute("href", "#i-" + name);
  svg.append(use);
  return svg;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;").replaceAll("'", "&#039;");
}

const store = {
  get(key, fallback) {
    try {
      const raw = localStorage.getItem(key);
      return raw === null ? fallback : JSON.parse(raw);
    } catch (error) {
      return fallback;
    }
  },
  set(key, value) {
    try {
      localStorage.setItem(key, JSON.stringify(value));
      return true;
    } catch (error) {
      return false; // quota or blocked storage: the app keeps working in memory
    }
  },
};

function timeAgo(ms) {
  const seconds = Math.max(0, Math.round((Date.now() - ms) / 1000));
  if (seconds < 45) return "just now";
  const units = [["y", 31536000], ["mo", 2592000], ["w", 604800], ["d", 86400], ["h", 3600], ["m", 60]];
  for (const [label, size] of units) {
    if (seconds >= size) return Math.floor(seconds / size) + label + " ago";
  }
  return "just now";
}

function clockTime(ms) {
  return new Date(ms).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
}

function shortName(name, max = 34) {
  const text = String(name || "");
  return text.length > max ? text.slice(0, max - 1) + "…" : text;
}

function toast(message, kind = "info") {
  const node = h("div", { class: "toast" + (kind === "error" ? " toast-error" : ""), role: kind === "error" ? "alert" : "status" },
    icon(kind === "error" ? "alert" : "check", "icon-sm"), h("span", { text: message }));
  $("#toasts").append(node);
  setTimeout(() => node.remove(), kind === "error" ? 6000 : 3200);
}

const uid = () => (window.crypto && crypto.randomUUID ? crypto.randomUUID()
  : Date.now().toString(36) + Math.random().toString(36).slice(2));

function debounce(fn, wait) {
  let timer = null;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), wait);
  };
}

async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch (error) {
    return false;
  }
}

// Answers are markdown. Sanitize: they quote document and web text.
function renderMarkdown(node, text) {
  if (window.marked && window.DOMPurify) {
    node.innerHTML = DOMPurify.sanitize(marked.parse(String(text || "")));
    node.querySelectorAll("a").forEach(link => { link.target = "_blank"; link.rel = "noopener"; });
  } else {
    node.textContent = text || "";
  }
}

// ================================================================== menus & dialogs

const menuEl = $("#menu");
let menuAnchor = null;

/* items: [{ label, icon, onSelect, danger, checked, disabled } | "sep" | { heading }] */
function openMenu(anchor, items, { align = "start" } = {}) {
  if (menuAnchor === anchor) return closeMenu();
  closeMenu();
  menuEl.replaceChildren();
  for (const item of items) {
    if (item === "sep") { menuEl.append(h("div", { class: "menu-sep", role: "separator" })); continue; }
    if (item.heading) { menuEl.append(h("div", { class: "menu-label", text: item.heading })); continue; }
    const button = h("button", {
      class: "menu-item" + (item.danger ? " danger" : "") + (item.checked ? " checked" : ""),
      type: "button", role: item.checked === undefined ? "menuitem" : "menuitemradio",
      "aria-checked": item.checked === undefined ? null : String(Boolean(item.checked)),
      disabled: item.disabled || null, title: item.title || null,
      onclick: () => { closeMenu(); item.onSelect && item.onSelect(); },
    }, item.icon ? icon(item.icon, "icon-sm") : null, h("span", { text: item.label }),
       item.checked ? icon("check", "icon-sm") : null);
    menuEl.append(button);
  }
  menuEl.hidden = false;
  const rect = anchor.getBoundingClientRect();
  const width = menuEl.offsetWidth;
  const height = menuEl.offsetHeight;
  let left = align === "end" ? rect.right - width : rect.left;
  left = Math.max(8, Math.min(left, window.innerWidth - width - 8));
  let top = rect.bottom + 6;
  if (top + height > window.innerHeight - 8) top = Math.max(8, rect.top - height - 6);
  menuEl.style.left = left + "px";
  menuEl.style.top = top + "px";
  menuAnchor = anchor;
  anchor.setAttribute("aria-expanded", "true");
  const first = menuEl.querySelector(".menu-item:not(:disabled)");
  if (first) first.focus();
}

function closeMenu() {
  if (!menuAnchor) return;
  menuEl.hidden = true;
  menuAnchor.setAttribute("aria-expanded", "false");
  const anchor = menuAnchor;
  menuAnchor = null;
  if (document.activeElement && menuEl.contains(document.activeElement)) anchor.focus();
}

document.addEventListener("click", event => {
  if (menuAnchor && !menuEl.contains(event.target) && !menuAnchor.contains(event.target)) closeMenu();
});
menuEl.addEventListener("keydown", event => {
  const items = [...menuEl.querySelectorAll(".menu-item:not(:disabled)")];
  const index = items.indexOf(document.activeElement);
  if (event.key === "ArrowDown") { event.preventDefault(); items[(index + 1) % items.length]?.focus(); }
  if (event.key === "ArrowUp") { event.preventDefault(); items[(index - 1 + items.length) % items.length]?.focus(); }
  if (event.key === "Home") { event.preventDefault(); items[0]?.focus(); }
  if (event.key === "End") { event.preventDefault(); items[items.length - 1]?.focus(); }
  if (event.key === "Escape") { event.preventDefault(); closeMenu(); }
  if (event.key === "Tab") closeMenu();
});
window.addEventListener("resize", closeMenu);

const modal = $("#modal");

/* Opens the shared <dialog>. actions: [{ label, kind, onClick(close) }]. */
function openModal({ title, body, actions = [], onClose }) {
  closeMenu();
  $("#modalTitle").textContent = title;
  $("#modalBody").replaceChildren(body);
  const foot = $("#modalFoot");
  foot.replaceChildren(...actions.map(action => h("button", {
    class: "btn " + (action.kind === "primary" ? "btn-primary" : action.kind === "danger" ? "btn-danger" : "btn-ghost"),
    type: "button",
    onclick: () => action.onClick ? action.onClick(() => modal.close()) : modal.close(),
  }, action.label)));
  modal.onclose = () => { if (onClose) onClose(); };
  modal.showModal();
  const focusTarget = body.querySelector("input, textarea") || foot.querySelector(".btn-primary");
  if (focusTarget) focusTarget.focus();
}

function field({ label, id, type = "text", value = "", placeholder = "", multiline = false }) {
  const input = multiline
    ? h("textarea", { id, placeholder })
    : h("input", { id, type, placeholder, autocomplete: "off" });
  input.value = value;
  const error = h("div", { class: "error-text", role: "alert" });
  return { node: h("div", { class: "field" }, h("label", { for: id, text: label }), input, error), input, error };
}

/* Small form dialog: one or more fields, a submit action with validation. */
function formModal({ title, fields, submitLabel, onSubmit }) {
  const built = fields.map(field);
  const body = h("div", {}, ...built.map(f => f.node));
  const submit = close => {
    built.forEach(f => { f.error.textContent = ""; });
    const values = built.map(f => f.input.value.trim());
    const problem = onSubmit(values);
    if (problem) {
      built[problem.field || 0].error.textContent = problem.message;
      built[problem.field || 0].input.focus();
      return;
    }
    close();
  };
  built.forEach(f => {
    if (f.input.tagName === "INPUT") {
      f.input.addEventListener("keydown", event => {
        if (event.key === "Enter") { event.preventDefault(); submit(() => modal.close()); }
      });
    }
  });
  openModal({ title, body, actions: [{ label: "Cancel" }, { label: submitLabel, kind: "primary", onClick: submit }] });
}

function confirmModal({ title, message, confirmLabel, onConfirm }) {
  openModal({
    title,
    body: h("p", { class: "muted", text: message }),
    actions: [{ label: "Cancel" }, { label: confirmLabel, kind: "danger", onClick: close => { onConfirm(); close(); } }],
  });
}

// ========================================================================= layout

const app = $("#app");
const wideLayout = () => window.matchMedia("(min-width: 1101px)").matches;
const desktopLeft = () => window.matchMedia("(min-width: 821px)").matches;
const layout = store.get("researchly.layout", { left: true, right: true, modes: true });

function applyLayout() {
  app.classList.toggle("left-collapsed", !layout.left);
  app.classList.toggle("right-collapsed", !layout.right);
  $("#modesBar").classList.toggle("collapsed", !layout.modes);
  const toggle = $("#modesToggle");
  toggle.setAttribute("aria-expanded", String(layout.modes));
  toggle.setAttribute("aria-label", layout.modes ? "Hide answer styles" : "Show answer styles");
  toggle.title = toggle.getAttribute("aria-label");
  toggle.querySelector("use").setAttribute("href", layout.modes ? "#i-chevron-down" : "#i-chevron-up");
  store.set("researchly.layout", layout);
}

function openDrawer(side) {
  app.classList.add(side === "left" ? "left-open" : "right-open");
  $("#backdrop").hidden = false;
}

function closeDrawers() {
  app.classList.remove("left-open", "right-open");
  $("#backdrop").hidden = true;
}

function showLeft() {
  if (desktopLeft()) { layout.left = true; applyLayout(); } else openDrawer("left");
}

function showRight(pane) {
  if (pane) selectPane(pane);
  if (wideLayout()) { layout.right = true; applyLayout(); } else openDrawer("right");
}

$("#collapseLeft").addEventListener("click", () => {
  if (desktopLeft()) { layout.left = false; applyLayout(); } else closeDrawers();
});
$("#collapseRight").addEventListener("click", () => {
  if (wideLayout()) { layout.right = false; applyLayout(); } else closeDrawers();
});
$("#openLeft").addEventListener("click", showLeft);
$("#openRight").addEventListener("click", () => showRight());
$("#backdrop").addEventListener("click", closeDrawers);
$("#modesToggle").addEventListener("click", () => { layout.modes = !layout.modes; applyLayout(); });
document.addEventListener("keydown", event => {
  if (event.key === "Escape" && !modal.open && !menuAnchor) closeDrawers();
});

/* Arrow-key navigation for a tablist (WAI-ARIA tabs pattern). */
function wireTablist(list, onSelect) {
  list.addEventListener("keydown", event => {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    const tabs = [...list.querySelectorAll('[role="tab"]:not([aria-disabled="true"])')];
    const index = tabs.indexOf(document.activeElement);
    let next = index;
    if (event.key === "ArrowRight") next = (index + 1) % tabs.length;
    if (event.key === "ArrowLeft") next = (index - 1 + tabs.length) % tabs.length;
    if (event.key === "Home") next = 0;
    if (event.key === "End") next = tabs.length - 1;
    event.preventDefault();
    tabs[next].focus();
    onSelect(tabs[next]);
  });
}

function selectView(name) {
  document.querySelectorAll(".view-tab").forEach(tab => {
    const active = tab.dataset.view === name;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-selected", String(active));
    tab.tabIndex = active ? 0 : -1;
  });
  $("#chatView").hidden = name !== "chat";
  $("#inspectorView").hidden = name !== "inspector";
  $("#chatView").classList.toggle("active", name === "chat");
  $("#inspectorView").classList.toggle("active", name === "inspector");
}

document.querySelectorAll(".view-tab").forEach(tab => tab.addEventListener("click", () => selectView(tab.dataset.view)));
wireTablist($(".view-tabs"), tab => selectView(tab.dataset.view));

function selectPane(name) {
  document.querySelectorAll(".right-tab").forEach(tab => {
    if (tab.getAttribute("aria-disabled") === "true") return;
    const active = tab.dataset.pane === name;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-selected", String(active));
    tab.tabIndex = active ? 0 : -1;
  });
  $("#knowledgePane").hidden = name !== "knowledge";
  $("#insightsPane").hidden = name !== "insights";
  $("#knowledgePane").classList.toggle("active", name === "knowledge");
  $("#insightsPane").classList.toggle("active", name === "insights");
  if (name === "insights") refreshSessionInsight();
}

document.querySelectorAll(".right-tab").forEach(tab => tab.addEventListener("click", () => {
  if (tab.getAttribute("aria-disabled") === "true") {
    toast("Explore is coming soon: topic maps and related reading for your sources.");
    return;
  }
  selectPane(tab.dataset.pane);
}));
wireTablist($(".right-tabs"), tab => selectPane(tab.dataset.pane));

// ======================================================================== profile

let userName = store.get("researchly.name", null) ?? (() => {
  try { return localStorage.getItem("ragUserName") || ""; } catch (error) { return ""; } // earlier versions' key
})();

function renderProfile() {
  const display = userName || "Researcher";
  $("#heroName").textContent = display;
  $("#profileName").textContent = display;
  $("#profileAvatar").textContent = display.trim().charAt(0).toUpperCase() || "R";
}

$("#openSettings").addEventListener("click", () => {
  const name = field({ label: "Your name (used in the greeting)", id: "settingsName", value: userName, placeholder: "Researcher" });
  name.input.addEventListener("input", () => {
    userName = name.input.value.trim();
    store.set("researchly.name", userName);
    renderProfile();
  });
  const body = h("div", {}, name.node,
    h("p", { class: "muted", text: "Your profile, chats, feedback and saved research are stored in this browser only. There is no account or email." }),
    h("button", {
      class: "btn btn-outline", type: "button",
      onclick: () => confirmModal({
        title: "Delete all chats?",
        message: "This removes every chat in this browser. It can't be undone.",
        confirmLabel: "Delete all",
        onConfirm: () => { chats = []; saveChats(); newChat(); toast("All chats deleted."); },
      }),
    }, icon("trash", "icon-sm"), "Delete all chats"));
  openModal({ title: "Settings", body, actions: [{ label: "Done", kind: "primary" }] });
});

// ========================================================================== chats

/* chat: { id, title, createdAt, updatedAt, sources: [names], turns: [turn] }
   turn: { role: "user"|"assistant", content, time, mode?, meta? } */
let chats = store.get("researchly.chats", null) ?? migrateOldChats();
let currentChatId = null;
let chatFilter = "";
let showAllChats = false;
const CHATS_VISIBLE = 6;

function migrateOldChats() {
  // Earlier versions stored { id, title, turns } under "ragChats".
  const old = store.get("ragChats", []);
  return Array.isArray(old) ? old.map(chat => {
    const created = parseInt(chat.id, 36) || Date.now();
    return { ...chat, createdAt: created, updatedAt: created, sources: chat.sources || [] };
  }) : [];
}

function saveChats() {
  // No cap: if the storage quota fills, drop the oldest from storage only.
  for (let keep = chats.length; keep >= 0; keep--) {
    if (store.set("researchly.chats", chats.slice(0, keep))) return;
  }
}

const currentChat = () => chats.find(chat => chat.id === currentChatId) || null;

function renderChatList() {
  const list = $("#chatList");
  const query = chatFilter.toLowerCase();
  const matches = chats.filter(chat => !query || chat.title.toLowerCase().includes(query)
    || chat.turns.some(turn => (turn.content || "").toLowerCase().includes(query)));
  const visible = showAllChats || query ? matches : matches.slice(0, CHATS_VISIBLE);

  list.replaceChildren();
  if (!chats.length) list.append(h("div", { class: "empty", text: "No chats yet. Ask your first question!" }));
  else if (!matches.length) list.append(h("div", { class: "empty", text: "No chats match your search." }));

  for (const chat of visible) {
    const sourceCount = (chat.sources || []).length;
    const menuBtn = h("button", { class: "icon-btn", type: "button", "aria-label": "Chat options for " + chat.title,
      "aria-haspopup": "menu", "aria-expanded": "false" }, icon("more"));
    menuBtn.addEventListener("click", event => {
      event.stopPropagation();
      openMenu(menuBtn, [
        { label: "Rename", icon: "edit", onSelect: () => renameChat(chat.id) },
        { label: "Delete", icon: "trash", danger: true, onSelect: () => deleteChat(chat.id) },
      ], { align: "end" });
    });
    list.append(h("div", { class: "chat-item" + (chat.id === currentChatId ? " active" : "") },
      h("button", { class: "chat-open", type: "button", title: chat.title, onclick: () => openChat(chat.id),
        "aria-current": chat.id === currentChatId ? "true" : null },
        icon("message"),
        h("span", { class: "chat-meta" },
          h("strong", { text: chat.title }),
          h("span", { text: `${sourceCount} source${sourceCount === 1 ? "" : "s"} • ${timeAgo(chat.updatedAt || chat.createdAt)}` }))),
      menuBtn));
  }

  const more = $("#showMoreChats");
  more.hidden = Boolean(query) || matches.length <= CHATS_VISIBLE;
  more.querySelector("span").textContent = showAllChats ? "Show less" : `Show more (${matches.length - CHATS_VISIBLE})`;
  more.querySelector("use").setAttribute("href", showAllChats ? "#i-chevron-up" : "#i-chevron-right");
  renderTitle();
}

function renderTitle() {
  const chat = currentChat();
  $("#chatTitle").textContent = chat ? chat.title : "New research";
  document.title = (chat ? chat.title + " · " : "") + "Researchly";
}

$("#showMoreChats").addEventListener("click", () => { showAllChats = !showAllChats; renderChatList(); });
$("#chatSearchToggle").addEventListener("click", () => {
  const wrap = $("#chatSearchWrap");
  wrap.hidden = !wrap.hidden;
  $("#chatSearchToggle").setAttribute("aria-expanded", String(!wrap.hidden));
  if (!wrap.hidden) $("#chatSearch").focus();
  else { chatFilter = ""; $("#chatSearch").value = ""; renderChatList(); }
});
$("#chatSearch").addEventListener("input", debounce(event => { chatFilter = event.target.value.trim(); renderChatList(); }, 120));

function renameChat(chatId) {
  const chat = chats.find(item => item.id === chatId);
  if (!chat) return;
  formModal({
    title: "Rename chat",
    fields: [{ label: "Title", id: "renameInput", value: chat.title }],
    submitLabel: "Save",
    onSubmit: ([title]) => {
      if (!title) return { message: "Title can't be empty." };
      chat.title = title.slice(0, 120);
      saveChats();
      renderChatList();
      return null;
    },
  });
}

function deleteChat(chatId) {
  const chat = chats.find(item => item.id === chatId);
  if (!chat) return;
  confirmModal({
    title: "Delete chat?",
    message: `“${shortName(chat.title, 60)}” will be removed from this browser.`,
    confirmLabel: "Delete",
    onConfirm: () => {
      chats = chats.filter(item => item.id !== chatId);
      saveChats();
      if (currentChatId === chatId) newChat();
      else renderChatList();
      toast("Chat deleted.");
    },
  });
}

function newChat() {
  stopStreamingUi();
  currentChatId = null;
  renderMessages();
  updateInspector(null);
  setConfidence(null);
  renderAnswerInsight(null);
  renderChatList();
  selectView("chat");
  setNav("home");
  $("#question").value = "";
  autoGrow();
  $("#question").focus();
  if (!desktopLeft()) closeDrawers();
}

function openChat(chatId) {
  if (streaming) { toast("Wait for the current answer to finish first."); return; }
  currentChatId = chatId;
  renderMessages();
  const lastAnswer = [...(currentChat()?.turns || [])].reverse().find(turn => turn.role === "assistant");
  setConfidence(lastAnswer?.meta?.confidence || null);
  renderAnswerInsight(lastAnswer?.meta || null);
  renderChatList();
  selectView("chat");
  if (!desktopLeft()) closeDrawers();
}

$("#newChat").addEventListener("click", newChat);
$("#chatTitleBtn").addEventListener("click", event => {
  const chat = currentChat();
  const items = [];
  if (chat) items.push({ label: "Rename this chat", icon: "edit", onSelect: () => renameChat(chat.id) }, "sep");
  const others = chats.filter(item => item.id !== currentChatId).slice(0, 8);
  if (others.length) {
    items.push({ heading: "Switch to" });
    others.forEach(item => items.push({ label: item.title, icon: "message", onSelect: () => openChat(item.id) }));
    items.push("sep");
  }
  items.push({ label: "New research", icon: "plus", onSelect: newChat });
  openMenu(event.currentTarget, items);
});

// ============================================================ messages & streaming

const messagesEl = $("#messages");
let streaming = false;
let followStream = true;
let lastScrollTop = 0;
let lastScrollHeight = 0;

function scrollToBottom() {
  messagesEl.scrollTo({ top: messagesEl.scrollHeight, behavior: "instant" });
}

// Follow the streaming answer only while the reader hasn't scrolled away.
messagesEl.addEventListener("scroll", () => {
  const top = messagesEl.scrollTop;
  const height = messagesEl.scrollHeight;
  const atBottom = height - top - messagesEl.clientHeight < 4;
  // A drop while content got shorter is the browser clamping after a re-render.
  if (top < lastScrollTop - 2 && height >= lastScrollHeight) followStream = false;
  else if (atBottom) followStream = true;
  lastScrollTop = top;
  lastScrollHeight = height;
}, { passive: true });

function renderMessages() {
  messagesEl.replaceChildren();
  const chat = currentChat();
  const turns = chat ? chat.turns : [];
  $("#hero").hidden = turns.length > 0;
  if (!turns.length) {
    messagesEl.append(h("div", { class: "welcome" },
      h("strong", { text: sources.length ? "Your sources are ready. Ask anything." : "Start by adding a source" }),
      sources.length
        ? "Try a quick prompt below, or pick a suggested question on the right."
        : "Drop a PDF, DOCX, TXT, Markdown file or image into the Knowledge panel, or add a URL, YouTube link or pasted text. Every answer cites the exact page it came from."));
    return;
  }
  const lastAssistant = turns.map(turn => turn.role).lastIndexOf("assistant");
  turns.forEach((turn, index) => {
    if (turn.role === "user") messagesEl.append(userMessage(turn));
    else messagesEl.append(assistantMessage(turn, index, { latest: index === lastAssistant }).row);
  });
  scrollToBottom();
}

function userMessage(turn) {
  return h("div", { class: "msg msg-user" },
    h("div", { class: "bubble-user" }, turn.content,
      turn.time ? h("time", { datetime: new Date(turn.time).toISOString(), text: clockTime(turn.time) }) : null),
    h("span", { class: "avatar avatar-sm", "aria-hidden": "true", text: (userName || "R").charAt(0).toUpperCase() }));
}

/* Builds an assistant message. Returns handles used while streaming. */
function assistantMessage(turn, index, { latest = false, pending = false } = {}) {
  const head = h("div", { class: "answer-head" });
  const notices = h("div");
  const body = h("div", { class: "markdown" });
  const footer = h("div");
  const card = h("article", { class: "answer-card", "aria-label": "Answer" }, head, notices, body, footer);
  const row = h("div", { class: "msg msg-assistant", "data-index": String(index) },
    h("span", { class: "bot-avatar", "aria-hidden": "true" }, icon("sparkle")), card);

  head.append(h("span", { class: "tag" }, modeIcon(turn.mode), modeLabel(turn.mode)));
  if (pending) {
    body.append(h("div", { class: "typing", "aria-label": "Thinking" }, h("span"), h("span"), h("span")));
  } else {
    renderMarkdown(body, turn.content);
  }
  if (turn.meta) renderMeta({ head, notices }, turn.meta);
  if (!pending) renderFooter(footer, turn, index, latest);
  return { row, head, notices, body, footer };
}

function renderMeta({ head, notices }, meta) {
  const conf = meta.confidence;
  if (conf && head.childElementCount < 2) {
    const pct = Math.round((Number(conf.score) || 0) * 100);
    head.append(h("span", { class: "tag tag-" + (conf.label || "low"),
      title: "Retrieval confidence: semantic similarity + keyword coverage of the retrieved passages" },
      icon("target", "icon-sm"), `Confidence ${pct}% · ${conf.label || "low"}`));
    if (meta.route) head.append(h("span", { class: "tag" }, icon(meta.route === "Web Search" ? "globe" : "file-text", "icon-sm"), meta.route));
  }
  notices.replaceChildren();
  (meta.warnings || []).forEach(item => {
    const reason = String(item.message || "could not be read").trim().replace(/[.]?$/, ".");
    notices.append(h("div", { class: "notice notice-warn", role: "status" }, icon("alert", "icon-sm"),
      h("span", { text: `Skipped ${item.source}: ${reason} Answering from your other sources.` })));
  });
  if (conf && conf.warning) {
    notices.append(h("div", { class: "notice notice-warn" }, icon("alert", "icon-sm"), h("span", { text: conf.warning })));
  }
}

function renderFooter(footer, turn, index, latest) {
  footer.replaceChildren();
  const meta = turn.meta || {};

  // Source chips from the real retrieval metadata.
  const citations = meta.citations || [];
  const web = meta.web || [];
  if (citations.length || web.length) {
    const chips = h("div", { class: "source-chips" }, h("span", { text: "Sources:" }));
    const seen = new Set();
    citations.forEach((cite, i) => {
      const key = cite.source + "#" + cite.page;
      if (seen.has(key)) return;
      seen.add(key);
      chips.append(h("button", { class: "source-chip", type: "button", title: `${cite.source}, page ${cite.page}`,
        onclick: () => showCitation(cite, i + 1) },
        h("span", { class: "num", text: String(seen.size) }),
        h("span", { text: `${shortName(displaySourceName(cite.source), 28)} (p. ${cite.page})` })));
    });
    web.forEach(result => {
      chips.append(h("button", { class: "source-chip", type: "button", title: result.title || "Web source",
        onclick: () => showWebResult(result) }, icon("globe", "icon-sm"), h("span", { text: "Web Source" })));
    });
    footer.append(chips);
  }

  if (latest && (meta.followups || []).length) {
    footer.append(h("div", { class: "followups", "aria-label": "Follow-up questions" },
      ...meta.followups.slice(0, 3).map(question => h("button", { class: "quick", type: "button",
        onclick: () => sendQuestion(question) }, question))));
  }

  const details = h("div", { class: "details-line", hidden: true });
  if (meta.metrics) {
    const m = meta.metrics;
    const tokens = (m.prompt_tokens || 0) + (m.completion_tokens || 0);
    details.append(
      h("span", { text: `⏱ ${m.latency_ms} ms` }),
      h("span", { text: `🔢 ${tokens} tokens` }),
      h("span", { text: `💸 $${(Number(m.request_cost_usd) || 0).toFixed(5)}` }),
      h("span", { text: `🔎 ${m.retrieval_mode || ""}` }));
  }

  const key = feedbackKey(index);
  const feedback = feedbackStore[key];
  const up = h("button", { class: "icon-btn", type: "button", "aria-label": "Good answer", title: "Good answer",
    "aria-pressed": String(feedback === "up") }, icon("thumbs-up", "icon-sm"));
  const down = h("button", { class: "icon-btn", type: "button", "aria-label": "Bad answer", title: "Bad answer",
    "aria-pressed": String(feedback === "down") }, icon("thumbs-down", "icon-sm"));
  up.addEventListener("click", () => setFeedback(key, "up", up, down));
  down.addEventListener("click", () => setFeedback(key, "down", up, down));
  const more = h("button", { class: "icon-btn", type: "button", "aria-label": "More answer options",
    "aria-haspopup": "menu", "aria-expanded": "false" }, icon("more", "icon-sm"));
  more.addEventListener("click", () => openMenu(more, [
    { label: details.hidden ? "Show details (latency, tokens, cost)" : "Hide details", icon: "info",
      disabled: !meta.metrics, onSelect: () => { details.hidden = !details.hidden; } },
    { label: "Save to research", icon: "bookmark", onSelect: () => saveResearch(index) },
    { label: "Open RAG Inspector", icon: "eye", disabled: !latest, title: latest ? null : "Inspector shows the latest answer",
      onSelect: () => selectView("inspector") },
  ], { align: "start" }));

  footer.append(h("div", { class: "answer-actions" },
    h("button", { class: "icon-btn", type: "button", "aria-label": "Copy answer", title: "Copy answer",
      onclick: async () => toast(await copyText(turn.content) ? "Answer copied." : "Couldn't copy: clipboard blocked.", "info") },
      icon("copy", "icon-sm")),
    up, down,
    h("button", { class: "icon-btn", type: "button", "aria-label": "Regenerate answer", title: latest ? "Regenerate" : "Only the latest answer can be regenerated",
      disabled: !latest || null, onclick: () => regenerate(index) }, icon("refresh", "icon-sm")),
    more,
    turn.time ? h("time", { datetime: new Date(turn.time).toISOString(), text: clockTime(turn.time) }) : null),
    details);
}

function displaySourceName(source) {
  return String(source).startsWith("youtube:") ? "YouTube " + String(source).slice(8) : source;
}

function showCitation(cite, number) {
  const body = h("div", {},
    h("p", { class: "muted" }, `[${number}] ${displaySourceName(cite.source)} · page ${cite.page}` +
      (cite.section ? ` · ${cite.section}` : "") + ` · score ${Number(cite.score || 0).toFixed(3)}`));
  const preview = h("pre", { class: "chunk-preview" });
  preview.innerHTML = highlightTerms(cite.preview || "", cite.matched_terms);
  body.append(preview);
  openModal({ title: "Source passage", body, actions: [{ label: "Close", kind: "primary" }] });
}

function showWebResult(result) {
  const body = h("div", {},
    h("p", {}, h("strong", { text: result.title || "Web result" })),
    h("p", { class: "muted", text: result.snippet || "" }),
    result.url ? h("a", { href: result.url, target: "_blank", rel: "noopener" }, "Open source ", icon("external", "icon-sm")) : null);
  openModal({ title: "Web source", body, actions: [{ label: "Close", kind: "primary" }] });
}

function highlightTerms(text, terms) {
  let safe = escapeHtml(text);
  const seen = new Set();
  (terms || []).forEach(term => {
    const word = String(term).trim();
    if (word.length < 2 || seen.has(word.toLowerCase()) || !/^[a-z0-9]+$/i.test(word)) return;
    seen.add(word.toLowerCase());
    safe = safe.replace(new RegExp("(" + word + ")", "gi"), "<mark>$1</mark>");
  });
  return safe;
}

// Feedback (thumbs) — stored per chat + answer in this browser.
const feedbackStore = store.get("researchly.feedback", {});
const feedbackKey = index => (currentChatId || "draft") + ":" + index;

function setFeedback(key, value, up, down) {
  feedbackStore[key] = feedbackStore[key] === value ? undefined : value;
  store.set("researchly.feedback", feedbackStore);
  up.setAttribute("aria-pressed", String(feedbackStore[key] === "up"));
  down.setAttribute("aria-pressed", String(feedbackStore[key] === "down"));
  if (feedbackStore[key]) toast("Thanks for the feedback!");
}

// ---- Ask / stream -------------------------------------------------------------

let wordQueue = [];
let streamTimer = null;
let streamTarget = null;
let streamText = "";

function stopStreamingUi() {
  clearInterval(streamTimer);
  streamTimer = null;
  wordQueue = [];
  streamTarget = null;
}

function enqueueWords(text) {
  wordQueue.push(...(String(text).match(/\S+\s*|\s+/g) || []));
  if (!streamTimer) streamTimer = setInterval(flushWord, 22);
}

function flushWord() {
  if (!wordQueue.length) { clearInterval(streamTimer); streamTimer = null; return; }
  // Drain faster when far behind so long answers don't lag the server.
  const take = Math.max(1, Math.floor(wordQueue.length / 40));
  streamText += wordQueue.splice(0, take).join("");
  if (streamTarget) renderMarkdown(streamTarget, streamText);
  $("#inspectAnswer").textContent = streamText;
  if (followStream) scrollToBottom();
}

function waitForQueue() {
  return new Promise(resolve => {
    const check = () => (wordQueue.length ? setTimeout(check, 30) : resolve());
    check();
  });
}

function setBusy(busy) {
  streaming = busy;
  $("#sendBtn").disabled = busy;
  document.querySelectorAll(".quick, .suggestion").forEach(button => { button.disabled = busy; });
}

async function sendQuestion(rawQuestion, { regenerateIndex = null } = {}) {
  const question = String(rawQuestion || "").trim();
  if (streaming) return;
  if (!question) { $("#question").focus(); return; }
  const usable = sources.filter(source => source.status !== "failed");
  if (!usable.length) {
    toast(sources.length ? "None of your sources could be read. Add another source." : "Add a source first: a file, URL, YouTube link or pasted text.", "error");
    showRight("knowledge");
    return;
  }

  selectView("chat");
  let chat = currentChat();
  if (!chat) {
    chat = { id: Date.now().toString(36), title: shortName(question, 70), createdAt: Date.now(), updatedAt: Date.now(), sources: [], turns: [] };
    chats.unshift(chat);
    currentChatId = chat.id;
  }
  chat.sources = [...new Set([...(chat.sources || []), ...usable.map(source => source.name)])];

  // History = everything before this question (for regenerate: before the replaced answer).
  if (regenerateIndex !== null) chat.turns.splice(regenerateIndex, 1);
  const history = chat.turns.map(({ role, content }) => ({ role, content }));
  if (regenerateIndex === null) chat.turns.push({ role: "user", content: question, time: Date.now() });
  else history.pop(); // the question itself is resent as `question`
  chat.updatedAt = Date.now();
  chats = [chat, ...chats.filter(item => item.id !== chat.id)];
  saveChats();
  renderChatList();
  renderMessages();

  const turn = { role: "assistant", content: "", mode: currentMode, time: Date.now(), meta: { followups: [] } };
  const view = assistantMessage(turn, chat.turns.length, { pending: true });
  messagesEl.append(view.row);
  followStream = true;
  scrollToBottom();
  setBusy(true);
  streamText = "";
  streamTarget = null;
  $("#inspectAnswer").textContent = "Streaming answer...";

  const form = new FormData();
  usable.forEach(source => {
    if (source.file) form.append("files", source.file, source.file.name);
    if (source.kind === "url") form.append("url", source.value);
    if (source.kind === "youtube") form.append("youtube_url", source.value);
  });
  form.append("question", question);
  form.append("answer_mode", currentMode);
  form.append("use_web", webSearch ? "true" : "false");
  form.append("history", JSON.stringify(history));

  try {
    const response = await fetch("/api/research/stream", { method: "POST", body: form });
    if (!response.ok || !response.body) {
      const data = await response.json().catch(() => ({}));
      throw new Error(typeof data.detail === "string" ? data.detail : "The request failed. Please try again.");
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let finished = false;
    while (!finished) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const events = buffer.split("\n\n");
      buffer = events.pop() || "";
      for (const eventText of events) finished = handleEvent(eventText, turn, view) || finished;
    }
    if (buffer.trim()) handleEvent(buffer, turn, view);
    await waitForQueue();
    if (!turn.content) throw new Error("The model returned an empty answer. Please try again.");
    renderMarkdown(view.body, turn.content);
    chat.turns.push(turn);
    chat.updatedAt = Date.now();
    saveChats();
    renderChatList();
    // Re-render so action rows reflect the new "latest" answer, but leave the
    // reader where they are if they scrolled up during the stream.
    const keepScroll = followStream ? null : messagesEl.scrollTop;
    renderMessages();
    if (keepScroll !== null) messagesEl.scrollTop = keepScroll;
  } catch (error) {
    stopStreamingUi();
    view.body.replaceChildren(h("div", { class: "notice notice-error", role: "alert" }, icon("alert", "icon-sm"),
      h("span", { text: error.message || "Something went wrong." }),
      h("button", { class: "btn btn-ghost", type: "button", onclick: () => {
        view.row.remove();
        const index = chat.turns.length - 1;
        if (chat.turns[index] && chat.turns[index].role === "user") {
          chat.turns.splice(index, 1);
          sendQuestion(question);
        }
      } }, icon("refresh", "icon-sm"), "Retry")));
  } finally {
    setBusy(false);
  }
}

/* Returns true when the stream reports it's finished. */
function handleEvent(eventText, turn, view) {
  const lines = eventText.split("\n");
  const eventLine = lines.find(line => line.startsWith("event:"));
  const dataLine = lines.find(line => line.startsWith("data:"));
  if (!eventLine || !dataLine) return false;
  const name = eventLine.slice(6).trim();
  const payload = JSON.parse(dataLine.slice(5).trim());

  if (name === "trace") {
    turn.meta = {
      ...turn.meta,
      confidence: payload.confidence,
      route: payload.route,
      warnings: payload.warnings || [],
      citations: (payload.retrieved_chunks || []).map(chunk => ({
        source: chunk.source, page: chunk.page, section: chunk.section, score: chunk.score,
        preview: chunk.preview, matched_terms: chunk.matched_terms })),
      web: (payload.web_results || []).map(result => ({ title: result.title, url: result.url, snippet: result.snippet })),
    };
    renderMeta(view, turn.meta);
    updateInspector(payload);
    setConfidence(payload.confidence);
    renderAnswerInsight(turn.meta, payload);
  }
  if (name === "chunk") {
    if (!streamTarget) { view.body.replaceChildren(); streamTarget = view.body; }
    enqueueWords(payload.text);
  }
  if (name === "done") {
    turn.content = payload.answer || streamText;
    turn.meta.metrics = payload.metrics;
    turn.meta.followups = payload.follow_up_questions || [];
    $("#inspectAnswer").textContent = turn.content;
    renderAnswerInsight(turn.meta);
    showInspectorMetrics(payload.metrics);
    return true;
  }
  if (name === "error") throw new Error(payload.message || "Streaming failed.");
  return false;
}

function regenerate(index) {
  const chat = currentChat();
  if (!chat || streaming) return;
  const question = [...chat.turns.slice(0, index)].reverse().find(turn => turn.role === "user");
  if (question) sendQuestion(question.content, { regenerateIndex: index });
}

// Composer
const questionInput = $("#question");

function autoGrow() {
  questionInput.style.height = "auto";
  questionInput.style.height = Math.min(questionInput.scrollHeight, 200) + "px";
}

function submitComposer() {
  const question = questionInput.value;
  if (!question.trim() || streaming) return;
  questionInput.value = "";
  autoGrow();
  sendQuestion(question);
}

questionInput.addEventListener("input", autoGrow);
questionInput.addEventListener("keydown", event => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    submitComposer();
  }
});
$("#sendBtn").addEventListener("click", submitComposer);
document.querySelectorAll("#quickPrompts .quick").forEach(button => {
  button.addEventListener("click", () => sendQuestion(button.textContent.trim()));
});

let webSearch = false;
function setWebSearch(on) {
  webSearch = on;
  $("#webChip").setAttribute("aria-pressed", String(on));
  document.querySelector('.nav-item[data-nav="web"]').setAttribute("aria-pressed", String(on));
}
$("#webChip").addEventListener("click", () => setWebSearch(!webSearch));

// ========================================================================= modes

const MODES = [
  { value: "Research Mode", title: "Research", sub: "Detailed & professional", icon: "🔬" },
  { value: "Beginner Mode", title: "Beginner", sub: "Simple explanations", icon: "🌱" },
  { value: "Interview Mode", title: "Interview", sub: "Q&A practice", icon: "🎯" },
  { value: "Professor Mode", title: "Professor", sub: "In-depth teaching", icon: "🎓" },
  { value: "30-Second Summary", title: "30-second", sub: "Quick summary", icon: "⚡" },
  { value: "Bestie Mode", title: "Bestie", sub: "Casual & fun", icon: "✨" },
];
let currentMode = store.get("researchly.mode", "Research Mode");
if (!MODES.some(mode => mode.value === currentMode)) currentMode = "Research Mode";

const modeOf = value => MODES.find(mode => mode.value === value) || MODES[0];
const modeLabel = value => (value === "30-Second Summary" ? value : modeOf(value).title + " Mode");
const modeIcon = value => h("span", { "aria-hidden": "true", text: modeOf(value).icon });

function setMode(value) {
  currentMode = value;
  store.set("researchly.mode", value);
  document.querySelectorAll(".mode-card").forEach(card => {
    const checked = card.dataset.mode === value;
    card.setAttribute("aria-checked", String(checked));
    card.tabIndex = checked ? 0 : -1;
  });
  document.querySelectorAll(".mode-name").forEach(node => { node.textContent = modeLabel(value); });
}

function renderModeCards() {
  const wrap = $("#modeCards");
  wrap.replaceChildren(...MODES.map(mode => h("button", {
    class: "mode-card", type: "button", role: "radio", "data-mode": mode.value, "aria-checked": "false",
    onclick: () => setMode(mode.value),
  }, h("span", { class: "mode-icon", "aria-hidden": "true", text: mode.icon }),
     h("span", {}, h("strong", { text: mode.title }), h("small", { text: mode.sub })))));
  wrap.addEventListener("keydown", event => {
    if (!["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(event.key)) return;
    event.preventDefault();
    const index = MODES.findIndex(mode => mode.value === currentMode);
    const step = event.key === "ArrowRight" || event.key === "ArrowDown" ? 1 : -1;
    const next = MODES[(index + step + MODES.length) % MODES.length];
    setMode(next.value);
    wrap.querySelector(`[data-mode="${next.value}"]`).focus();
  });
}

function openModeMenu(event) {
  openMenu(event.currentTarget, MODES.map(mode => ({
    label: `${mode.icon}  ${mode.title} — ${mode.sub}`, checked: mode.value === currentMode,
    onSelect: () => setMode(mode.value),
  })));
}
$("#modePill").addEventListener("click", openModeMenu);
$("#modeDropdown").addEventListener("click", openModeMenu);

// ======================================================================= sources

/* source: { id, kind: file|text|url|youtube, name, value?, file?, status: processing|ready|failed,
             pages, chunks, message } — kept in memory (File objects can't be persisted). */
let sources = [];
let analyzeController = null;
const insights = { pages: 0, chunks: 0, topics: [], questions: [], loading: false };
let showAllTopics = false;

const IMAGE_EXT = /\.(png|jpe?g|webp|bmp|tiff)$/i;
const SUPPORTED_EXT = /\.(pdf|docx|txt|md|markdown|png|jpe?g|webp|bmp|tiff)$/i;

function addFiles(fileList, kind = "file") {
  let added = 0;
  for (const file of fileList) {
    if (!SUPPORTED_EXT.test(file.name)) {
      toast(`${file.name}: unsupported type. Use PDF, DOCX, TXT, Markdown or an image.`, "error");
      continue;
    }
    if (sources.some(source => source.file && source.file.name === file.name && source.file.size === file.size)) continue;
    sources.push({ id: uid(), kind, name: file.name, file, status: "processing", pages: 0, chunks: 0, message: "" });
    added += 1;
  }
  if (added) sourcesChanged();
}

function addLink(kind, value) {
  if (sources.some(source => source.kind === kind && source.value === value)) {
    toast("That link is already in your sources.");
    return;
  }
  sources.push({ id: uid(), kind, name: value, value, status: "processing", pages: 0, chunks: 0, message: "" });
  sourcesChanged();
}

function removeSource(id) {
  sources = sources.filter(source => source.id !== id);
  sourcesChanged();
}

function sourcesChanged() {
  renderSources();
  if (!currentChat() || !currentChat().turns.length) renderMessages();
  scheduleAnalyze();
}

const scheduleAnalyze = debounce(() => analyzeSources(), 350);

async function analyzeSources({ refreshQuestions = false } = {}) {
  if (analyzeController) analyzeController.abort();
  const pending = sources.filter(source => source.status !== "failed" || source.retry);
  if (!pending.length) {
    Object.assign(insights, { pages: 0, chunks: 0, topics: [], questions: [], loading: false });
    renderInsights();
    return;
  }
  // Everything is re-sent (insights cover the whole set), but only sources not
  // yet ready show "Processing…".
  pending.forEach(source => {
    if (source.status !== "ready") source.status = "processing";
    source.retry = false;
  });
  insights.loading = true;
  renderSources();
  renderInsights();

  const controller = new AbortController();
  analyzeController = controller;
  const form = new FormData();
  pending.forEach(source => {
    if (source.file) form.append("files", source.file, source.file.name);
    if (source.kind === "url") form.append("url", source.value);
    if (source.kind === "youtube") form.append("youtube_url", source.value);
  });
  form.append("refresh_questions", refreshQuestions ? "true" : "false");

  try {
    const response = await fetch("/api/sources/analyze", { method: "POST", body: form, signal: controller.signal });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Analysis failed.");
    const byInput = new Map((data.sources || []).map(item => [item.input, item]));
    pending.forEach(source => {
      const report = byInput.get(source.file ? source.file.name : source.value);
      if (!report) return;
      source.status = report.status;
      source.pages = report.pages;
      source.chunks = report.chunks;
      source.message = report.message || "";
      if (report.name && source.kind === "youtube") source.name = displaySourceName(report.name);
    });
    Object.assign(insights, { pages: data.pages || 0, chunks: data.chunks || 0,
      topics: data.topics || [], questions: data.questions || [], loading: false });
    const failed = pending.filter(source => source.status === "failed");
    if (failed.length) toast(`${failed.length} source${failed.length > 1 ? "s" : ""} couldn't be read: ${failed.map(s => s.name).join(", ")}`, "error");
  } catch (error) {
    if (error.name === "AbortError") return;
    pending.forEach(source => {
      if (source.status === "processing") {
        source.status = "failed";
        source.message = "Couldn't analyze: " + (error.message || "network error") + ". Use ⋯ → Retry.";
      }
    });
    insights.loading = false;
    toast("Couldn't analyze your sources. You can retry from the source menu.", "error");
  } finally {
    if (analyzeController === controller) analyzeController = null;
    renderSources();
    renderInsights();
    if (!currentChat() || !currentChat().turns.length) renderMessages();
  }
}

function sourceIcon(source) {
  if (source.kind === "url") return ["web", "globe"];
  if (source.kind === "youtube") return ["yt", "youtube"];
  if (source.kind === "text") return ["", "file-text"];
  if (/\.pdf$/i.test(source.name)) return ["pdf", "file-text"];
  if (IMAGE_EXT.test(source.name)) return ["img", "image"];
  return ["", "file"];
}

function sourceSubtitle(source) {
  if (source.status === "failed") return source.message || "Couldn't be read.";
  if (source.status === "processing") return source.kind === "url" ? "Fetching page…" : source.kind === "youtube" ? "Fetching transcript…" : "Reading & indexing…";
  const kind = source.kind === "url" ? "Web source · " : source.kind === "youtube" ? "Transcript · " : "";
  return `${kind}${source.pages} page${source.pages === 1 ? "" : "s"} • ${source.chunks} chunk${source.chunks === 1 ? "" : "s"}`;
}

function renderSources() {
  const list = $("#sourceList");
  list.replaceChildren();
  $("#sourceCount").textContent = `(${sources.length})`;
  $("#sourcesChipLabel").textContent = sources.length ? `Sources (${sources.length})` : "Sources";
  if (!sources.length) {
    list.append(h("li", { class: "empty", text: "No sources yet. Your library is empty: add a file, link or pasted text above." }));
    return;
  }
  for (const source of sources) {
    const [tone, glyph] = sourceIcon(source);
    const badge = source.status === "ready"
      ? h("span", { class: "badge badge-ready" }, "Ready")
      : source.status === "failed"
        ? h("span", { class: "badge badge-failed" }, "Failed")
        : h("span", { class: "badge badge-processing" }, h("span", { class: "spinner", "aria-hidden": "true" }), "Processing…");
    const menuBtn = h("button", { class: "icon-btn icon-btn-sm", type: "button", "aria-label": "Options for " + source.name,
      "aria-haspopup": "menu", "aria-expanded": "false" }, icon("more"));
    menuBtn.addEventListener("click", () => {
      const items = [];
      if (source.status === "failed") items.push({ label: "Retry", icon: "refresh", onSelect: () => { source.retry = true; source.status = "processing"; renderSources(); analyzeSources(); } });
      if (source.kind === "url") items.push({ label: "Open link", icon: "external", onSelect: () => window.open(source.value, "_blank", "noopener") });
      items.push({ label: "Remove", icon: "trash", danger: true, onSelect: () => removeSource(source.id) });
      openMenu(menuBtn, items, { align: "end" });
    });
    list.append(h("li", { class: "source-item" },
      h("span", { class: "source-icon " + tone, "aria-hidden": "true" }, icon(glyph, "icon-sm")),
      h("span", { class: "source-main" },
        h("strong", { title: source.name, text: source.name }),
        h("span", { class: source.status === "failed" ? "error" : "", title: sourceSubtitle(source), text: sourceSubtitle(source) })),
      badge, menuBtn));
  }
}

const fileInput = $("#fileInput");
const pickFiles = () => fileInput.click();
fileInput.addEventListener("change", () => { addFiles([...fileInput.files]); fileInput.value = ""; });
$("#attachBtn").addEventListener("click", pickFiles);

const dropzone = $("#dropzone");
dropzone.addEventListener("click", pickFiles);
dropzone.addEventListener("keydown", event => {
  if (event.key === "Enter" || event.key === " ") { event.preventDefault(); pickFiles(); }
});
["dragenter", "dragover"].forEach(type => dropzone.addEventListener(type, event => {
  event.preventDefault();
  dropzone.classList.add("dragging");
}));
["dragleave", "drop"].forEach(type => dropzone.addEventListener(type, event => {
  event.preventDefault();
  dropzone.classList.remove("dragging");
}));
dropzone.addEventListener("drop", event => addFiles([...(event.dataTransfer?.files || [])]));
// Dropping a file anywhere else shouldn't navigate away from the app.
window.addEventListener("dragover", event => event.preventDefault());
window.addEventListener("drop", event => {
  event.preventDefault();
  if (!dropzone.contains(event.target) && event.dataTransfer?.files?.length) addFiles([...event.dataTransfer.files]);
});

$("#addUrlBtn").addEventListener("click", () => formModal({
  title: "Add a web page",
  fields: [{ label: "Page URL", id: "urlField", type: "url", placeholder: "https://example.com/article" }],
  submitLabel: "Add URL",
  onSubmit: ([value]) => {
    if (!/^https?:\/\/\S+\.\S+/i.test(value)) return { message: "Enter a full URL starting with http:// or https://" };
    addLink("url", value);
    return null;
  },
}));

$("#addYoutubeBtn").addEventListener("click", () => formModal({
  title: "Add a YouTube video",
  fields: [{ label: "YouTube link (its transcript is used)", id: "ytField", type: "url", placeholder: "https://www.youtube.com/watch?v=…" }],
  submitLabel: "Add video",
  onSubmit: ([value]) => {
    if (!/(youtube\.com\/(watch\?v=|embed\/|shorts\/)|youtu\.be\/)[\w-]{11}/i.test(value)) return { message: "That doesn't look like a YouTube video link." };
    addLink("youtube", value);
    return null;
  },
}));

let pastedCount = 0;
$("#pasteTextBtn").addEventListener("click", () => formModal({
  title: "Paste text",
  fields: [
    { label: "Name (optional)", id: "pasteName", placeholder: "Lecture notes" },
    { label: "Text", id: "pasteBody", multiline: true, placeholder: "Paste notes, an article or any text…" },
  ],
  submitLabel: "Add text",
  onSubmit: ([name, text]) => {
    if (text.length < 20) return { field: 1, message: "Paste at least a sentence or two." };
    pastedCount += 1;
    const safeName = (name || `Pasted text ${pastedCount}`).replace(/[\\/:*?"<>|]+/g, " ").slice(0, 60).trim();
    // The backend already ingests .txt files, so pasted text travels as one.
    addFiles([new File([text], safeName + ".txt", { type: "text/plain" })], "text");
    return null;
  },
}));

$("#manageSources").addEventListener("click", event => openMenu(event.currentTarget, [
  { label: "Add files…", icon: "upload", onSelect: pickFiles },
  { label: "Re-analyze sources", icon: "refresh", disabled: !sources.length,
    onSelect: () => { sources.forEach(s => { s.retry = true; s.status = "processing"; }); analyzeSources(); } },
  "sep",
  { label: "Clear all sources", icon: "trash", danger: true, disabled: !sources.length,
    onSelect: () => confirmModal({ title: "Clear all sources?", message: "Your chats are kept; only the source list is cleared.",
      confirmLabel: "Clear all", onConfirm: () => { sources = []; sourcesChanged(); } }) },
], { align: "end" }));
$("#sourcesChip").addEventListener("click", () => showRight("knowledge"));

// ============================================================ insights & inspector

function renderInsights() {
  $("#statPages").textContent = insights.pages;
  $("#statChunks").textContent = insights.chunks;
  $("#statTopics").textContent = insights.topics.length;

  const topicsEl = $("#topics");
  topicsEl.replaceChildren();
  if (insights.loading && !insights.topics.length) {
    topicsEl.append(h("span", { class: "muted", text: "Finding key topics…" }));
  } else if (!insights.topics.length) {
    topicsEl.append(h("span", { class: "muted", text: "Topics appear once a source is ready." }));
  } else {
    const limit = showAllTopics ? insights.topics.length : 6;
    insights.topics.slice(0, limit).forEach(topic => topicsEl.append(h("button", { class: "topic", type: "button",
      title: `Ask about ${topic}`, onclick: () => { questionInput.value = `Explain ${topic} using my sources.`; autoGrow(); questionInput.focus(); } }, topic)));
    if (insights.topics.length > 6) {
      topicsEl.append(h("button", { class: "topic topic-more", type: "button", onclick: () => { showAllTopics = !showAllTopics; renderInsights(); } },
        showAllTopics ? "Show less" : `+${insights.topics.length - 6} more`));
    }
  }

  const list = $("#suggestions");
  list.replaceChildren();
  if (insights.loading) {
    for (let i = 0; i < 3; i++) list.append(h("li", { class: "skeleton", "aria-hidden": "true" }));
  } else if (!insights.questions.length) {
    list.append(h("li", { class: "muted", text: "Suggested questions appear once a source is ready." }));
  } else {
    insights.questions.forEach(question => list.append(h("li", {},
      h("button", { class: "suggestion", type: "button", disabled: streaming || null, onclick: () => sendQuestion(question) },
        icon("chevron-right", "icon-sm"), h("span", { text: question }), icon("send", "icon-sm")))));
  }
  $("#refreshQuestions").disabled = insights.loading || !sources.some(source => source.status === "ready");
}

$("#refreshQuestions").addEventListener("click", async event => {
  const button = event.currentTarget;
  button.classList.add("is-spinning");
  await analyzeSources({ refreshQuestions: true });
  button.classList.remove("is-spinning");
});
$("#viewDetails").addEventListener("click", () => selectPane("insights"));
$("#openInspector").addEventListener("click", () => { selectView("inspector"); if (!wideLayout()) closeDrawers(); });

function setConfidence(confidence) {
  $("#statConfidence").textContent = confidence ? Math.round((Number(confidence.score) || 0) * 100) + "%" : "—";
}

function renderAnswerInsight(meta, trace) {
  const card = $("#answerInsight");
  if (!meta || !meta.confidence) {
    card.replaceChildren(h("p", { class: "muted", text: "Ask a question to see how the answer was grounded." }));
    return;
  }
  const conf = meta.confidence;
  const pct = value => Math.round((Number(value) || 0) * 100) + "%";
  const rows = [
    ["Confidence", `${pct(conf.score)} (${conf.label})`],
    ["Semantic signal", pct(conf.semantic_signal)],
    ["Keyword coverage", pct(conf.lexical_signal)],
    ["Route", meta.route || "Document RAG"],
    ["Passages used", String((meta.citations || []).length + (meta.web || []).length)],
  ];
  if (trace && trace.embedding_summary) rows.push(["Retrieval", trace.embedding_summary.semantic_active ? "Hybrid (semantic + BM25)" : "BM25 keyword fallback"]);
  if (meta.metrics) rows.push(["Latency", meta.metrics.latency_ms + " ms"], ["Cost", "$" + (Number(meta.metrics.request_cost_usd) || 0).toFixed(5)]);
  card.replaceChildren(
    h("div", { class: "meter", role: "img", "aria-label": "Confidence " + pct(conf.score) }, h("div", { style: `width:${pct(conf.score)}` })),
    h("dl", {}, ...rows.flatMap(([label, value]) => [h("dt", { text: label }), h("dd", { text: value })])),
    h("p", { class: "muted", text: "Confidence = 60% semantic similarity + 40% keyword coverage of the top passages. Computed from retrieval signals, not by an AI judge." }));
}

async function refreshSessionInsight() {
  const card = $("#sessionInsight");
  try {
    const response = await fetch("/api/metrics");
    if (!response.ok) throw new Error();
    const m = await response.json();
    card.replaceChildren(h("dl", {},
      ...[["Questions answered", m.requests], ["Hybrid retrievals", m.semantic_requests], ["BM25 fallbacks", m.fallback_requests],
        ["Generation model", m.generation_model], ["Embedding model", m.embedding_model], ["Errors", m.errors]]
        .flatMap(([label, value]) => [h("dt", { text: label }), h("dd", { text: String(value ?? "—") })])),
      h("p", { class: "muted", text: "Counters for this server instance; they reset when it restarts." }));
  } catch (error) {
    card.replaceChildren(h("p", { class: "muted", text: "Couldn't load session metrics. " },
      h("button", { class: "link-btn", type: "button", onclick: refreshSessionInsight, text: "Retry" })));
  }
}

function updateInspector(trace) {
  if (!trace) {
    $("#inspectQuestion").textContent = "Ask a question to populate the inspector.";
    $("#inspectEmbeddings").textContent = "Waiting for query vector terms.";
    $("#inspectChunks").textContent = "No retrieved chunks yet.";
    $("#inspectPrompt").textContent = "The full context-injected prompt will appear here.";
    $("#inspectAnswer").textContent = "The generated answer will appear here.";
    return;
  }
  $("#inspectQuestion").textContent = trace.question;
  $("#inspectEmbeddings").textContent = JSON.stringify({ ...trace.embedding_summary, confidence: trace.confidence }, null, 2);
  $("#inspectPrompt").textContent = trace.prompt;
  const chunksEl = $("#inspectChunks");
  chunksEl.replaceChildren();
  (trace.retrieved_chunks || []).forEach((chunk, index) => {
    const pre = h("pre");
    pre.innerHTML = highlightTerms(chunk.preview, chunk.matched_terms);
    chunksEl.append(h("div", { class: "score-row" },
      h("strong", { text: `Chunk ${index + 1}: ${displaySourceName(chunk.source)} · Page ${chunk.page}` }),
      h("div", { class: "muted", text: `Score ${Number(chunk.score).toFixed(3)} · Matched: ${(chunk.matched_terms || []).join(", ") || "no exact keyword overlap"}` }),
      h("div", { class: "score-track" }, h("div", { class: "score-fill", style: `width:${Math.max(4, Math.round(Number(chunk.score || 0) * 100))}%` })),
      pre));
  });
  if (!chunksEl.childElementCount) chunksEl.textContent = "No chunks were retrieved.";
}

function showInspectorMetrics(metrics) {
  if (!metrics) return;
  $("#inspectChunks").prepend(h("div", { class: "score-row" },
    h("strong", { text: "📊 Request metrics" }),
    h("div", { class: "muted", text: `Latency ${metrics.latency_ms} ms · Prompt ${metrics.prompt_tokens} tok · Completion ${metrics.completion_tokens} tok · Embed ${metrics.embedding_tokens} tok · Cost $${(Number(metrics.request_cost_usd) || 0).toFixed(5)} · ${metrics.retrieval_mode || ""}` })));
}

// ===================================================== export, share, saved research

$("#shareBtn").addEventListener("click", async () => {
  const data = { title: "Researchly · AI Research Assistant", text: "Try my AI research assistant.", url: location.href };
  try {
    if (navigator.share) { await navigator.share(data); return; }
  } catch (error) {
    if (error.name === "AbortError") return;
  }
  toast(await copyText(location.href) ? "Link copied to clipboard." : "Couldn't copy the link.", "info");
});

function chatAsMarkdown(chat) {
  const lines = [`# ${chat.title}`, "", `_Exported from Researchly on ${new Date().toLocaleString()}_`, ""];
  if ((chat.sources || []).length) lines.push(`**Sources:** ${chat.sources.join(", ")}`, "");
  chat.turns.forEach(turn => {
    if (turn.role === "user") lines.push(`## 🙋 ${turn.content}`, "");
    else {
      lines.push(`**${modeLabel(turn.mode)}**`, "", turn.content, "");
      const cites = (turn.meta?.citations || []).map(c => `${displaySourceName(c.source)} (p. ${c.page})`);
      if (cites.length) lines.push(`> Sources: ${[...new Set(cites)].join("; ")}`, "");
    }
  });
  return lines.join("\n");
}

function download(filename, text, type) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const link = h("a", { href: url, download: filename });
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

$("#exportBtn").addEventListener("click", event => {
  const chat = currentChat();
  const empty = !chat || !chat.turns.length;
  openMenu(event.currentTarget, [
    { label: "Markdown (.md)", icon: "file-text", disabled: empty, title: empty ? "Ask a question first" : null,
      onSelect: () => download((chat.title.replace(/[^\w -]+/g, "").trim() || "research") + ".md", chatAsMarkdown(chat), "text/markdown") },
    { label: "PDF (print dialog)", icon: "printer", disabled: empty, title: empty ? "Ask a question first" : null,
      onSelect: () => { selectView("chat"); setTimeout(() => window.print(), 50); } },
  ], { align: "end" });
});

$("#moreBtn").addEventListener("click", event => {
  const chat = currentChat();
  openMenu(event.currentTarget, [
    { label: "Open RAG Inspector", icon: "eye", onSelect: () => selectView("inspector") },
    { label: "Saved research", icon: "bookmark", onSelect: openSavedResearch },
    "sep",
    { label: layout.left ? "Hide sidebar" : "Show sidebar", icon: "chevrons-left", onSelect: () => (layout.left && desktopLeft() ? (layout.left = false, applyLayout()) : showLeft()) },
    { label: "Show knowledge panel", icon: "layers", onSelect: () => showRight("knowledge") },
    "sep",
    { label: "Delete this chat", icon: "trash", danger: true, disabled: !chat, onSelect: () => chat && deleteChat(chat.id) },
  ], { align: "end" });
});

let saved = store.get("researchly.saved", []);

function saveResearch(index) {
  const chat = currentChat();
  const turn = chat?.turns[index];
  if (!turn) return;
  const question = [...chat.turns.slice(0, index)].reverse().find(item => item.role === "user");
  if (saved.some(item => item.chatId === chat.id && item.answer === turn.content)) { toast("Already in Saved Research."); return; }
  saved.unshift({ id: uid(), chatId: chat.id, question: question?.content || chat.title,
    answer: turn.content, mode: turn.mode, savedAt: Date.now() });
  store.set("researchly.saved", saved);
  toast("Saved to Saved Research.");
}

function openSavedResearch() {
  const body = h("div");
  const render = () => {
    body.replaceChildren();
    if (!saved.length) {
      body.append(h("p", { class: "muted", text: "Nothing saved yet. Use ⋯ → “Save to research” under any answer." }));
      return;
    }
    saved.forEach(item => body.append(h("div", { class: "saved-item" },
      h("strong", { text: item.question }),
      h("p", { text: shortName(item.answer.replace(/[#*_`>]/g, ""), 220) }),
      h("div", { class: "row" },
        h("button", { class: "btn btn-ghost", type: "button", disabled: !chats.some(chat => chat.id === item.chatId) || null,
          title: chats.some(chat => chat.id === item.chatId) ? null : "The original chat was deleted",
          onclick: () => { modal.close(); openChat(item.chatId); } }, icon("message", "icon-sm"), "Open chat"),
        h("button", { class: "btn btn-ghost", type: "button", onclick: async () => toast(await copyText(item.answer) ? "Copied." : "Couldn't copy.") }, icon("copy", "icon-sm"), "Copy"),
        h("button", { class: "btn btn-ghost", type: "button", onclick: () => {
          saved = saved.filter(entry => entry.id !== item.id);
          store.set("researchly.saved", saved);
          render();
        } }, icon("trash", "icon-sm"), "Remove")))));
  };
  render();
  openModal({ title: "Saved research", body, actions: [{ label: "Close", kind: "primary" }] });
}

// =========================================================================== nav

function setNav(name) {
  document.querySelectorAll(".nav-item[data-nav]").forEach(item => {
    if (item.dataset.nav !== "web") item.classList.toggle("active", item.dataset.nav === name);
  });
}

document.querySelectorAll(".nav-item").forEach(item => item.addEventListener("click", () => {
  const target = item.dataset.nav;
  if (target === "web") {
    setWebSearch(!webSearch);
    toast(webSearch ? "Web search is on for your next questions." : "Web search is off.");
    return;
  }
  setNav(target);
  if (target === "home") { selectView("chat"); questionInput.focus(); }
  if (target === "recent") {
    const wrap = $("#chatSearchWrap");
    if (wrap.hidden) $("#chatSearchToggle").click();
    else $("#chatSearch").focus();
    return; // keep the drawer open so the list is visible
  }
  if (target === "library") showRight("knowledge");
  if (target === "saved") openSavedResearch();
  if (!desktopLeft()) closeDrawers();
}));

// ========================================================================== init

renderProfile();
renderModeCards();
setMode(currentMode);
applyLayout();
renderSources();
renderInsights();
renderChatList();
renderMessages();
autoGrow();
// Keep "N min ago" labels fresh.
setInterval(renderChatList, 60000);
