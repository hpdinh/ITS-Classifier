// assets/static/chat.js

// --- Simple chat state on the client ---
const chatState = {
  messages: [],   // {role: "user"|"assistant", content: string}
  sending: false,
};

function fixChatLinks() {
  // Select only links inside the chat log
  document.querySelectorAll("#chat-log a").forEach(link => {
    link.setAttribute("target", "_blank");
    link.setAttribute("rel", "noopener noreferrer");
  });
}

function appendMessage(role, content) {
  const chatLog = document.getElementById("chat-log");
  const wrapper = document.createElement("div");
  wrapper.className =
    (role === "user" ? "text-right" : "text-left") + " mb-2";

  const bubble = document.createElement("span");
  bubble.className =
    "inline-block px-4 py-2 rounded-lg " +
    (role === "user"
      ? "bg-blue-500 text-white"
      : "bg-gray-200 text-gray-800");
  if (role === "assistant") {
    bubble.innerHTML = marked.parse(content);
  } else {
    bubble.textContent = content;
  }

  wrapper.appendChild(bubble);
  chatLog.appendChild(wrapper);
  chatLog.scrollTop = chatLog.scrollHeight;

  fixChatLinks()
}

function setSending(isSending) {
  chatState.sending = isSending;
  const btn = document.getElementById("chat-send-btn");
  const input = document.getElementById("chat-input");
  if (!btn || !input) return;
  btn.disabled = isSending;
  input.disabled = isSending;
  btn.textContent = isSending ? "Sending..." : "Send";
}

function handleChatKey(event) {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendMessage();
  }
}

function autoResize(textarea) {
  textarea.style.height = "auto";
  textarea.style.height = Math.min(textarea.scrollHeight, 100) + "px";
}

// Small sanitizer
function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

async function sendMessage() {
  const chatInput = document.getElementById("chat-input");
  const message = chatInput.value.trim();
  if (!message || chatState.sending) return;

  appendMessage("user", message);
  chatInput.value = "";
  chatInput.style.height = "auto";

  setSending(true);
  try {
    const resp = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),   // 🔑 only send raw user msg
    });

    if (!resp.ok) {
      const text = await resp.text();
      appendMessage("assistant", `⚠️ LLM error: ${resp.status} ${text}`);
      return;
    }

    const data = await resp.json();
    appendMessage("assistant", data.reply || "(empty response)");

  } catch (e) {
    appendMessage("assistant", "⚠️ Network error calling /chat");
  } finally {
    setSending(false);
  }
}


async function resetToForm() {
  try {
    await fetch("/reset", { method: "POST" });
  } catch (e) {
    console.warn("Reset failed", e);
  }
  window.location.href = "/";
}

function showPredictions() {
  const predEl = document.getElementById("initial-prediction");
  const ticketEl = document.getElementById("initial-ticket");

  if (!ticketEl) return;

  if (predEl && predEl.dataset.predictions) {
    const pretty = predEl.dataset.predictions.split("|").join(", ");
    appendMessage("assistant", `Top Predictions: ${pretty}`);
  }
}

function showTickets() {
  const ticketEl = document.getElementById("initial-tickets");
  if (!ticketEl) return;

  if (ticketEl.dataset.tickets) {
    const items = JSON.parse(ticketEl.dataset.tickets);
    const mdList = items.map(t => `- ${t}`).join("\n");
    appendMessage("assistant", `**Related Tickets:**\n\n${mdList}`);
  }
}

function showTicketSummary() {
  const summaryEl = document.getElementById("initial-ticket-summary");
  if (!summaryEl) return;

  if (summaryEl.dataset.ticketSummary) {
    appendMessage("assistant", `**Ticket Summary:**\n\n${summaryEl.dataset.ticketSummary}`);
  }
}



function loadHistory() {
  const histEl = document.getElementById("chat-history");
  if (!histEl) return;

  const history = JSON.parse(histEl.dataset.history || "[]");

  for (const msg of history) {
    // Only append if explicitly marked show_msg=true
    if (msg.show_msg) {
      if (msg.role === "assistant") {
        appendMessage("assistant", msg.content);
      } else if (msg.role === "user") {
        appendMessage("user", msg.content);
      }
    }
  }
}


async function kickoffInit() {
  const initFlag = document.getElementById("chat-init-flag");
  if (!initFlag || initFlag.dataset.done === "true") return; // already done
  setSending(true)
  const resp = await fetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message: "INIT" }),
  });
  const data = await resp.json();
  if (data.reply) {
    appendMessage("assistant", data.reply);
  }
  setSending(false)
}

window.addEventListener("DOMContentLoaded", () => {
  showTicketSummary();
  showPredictions();  // show the fast "Top Predictions"
  showTickets();      // show the fast "Related Predictions"
  loadHistory();      // replay assistant replies only
  kickoffInit();
});