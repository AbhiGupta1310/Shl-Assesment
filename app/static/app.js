const messageList = document.querySelector("#messageList");
const recommendationList = document.querySelector("#recommendationList");
const recCount = document.querySelector("#recCount");
const chatForm = document.querySelector("#chatForm");
const input = document.querySelector("#messageInput");
const sendButton = document.querySelector("#sendButton");
const resetButton = document.querySelector("#resetButton");
const statusPill = document.querySelector("#statusPill");

let messages = [];
let busy = false;

const welcomeText =
  "Welcome! I am your SHL assessment consultant. Describe the role you're hiring for, and I will shortlist the best assessments for you.";

const typeLabels = {
  "A": "Ability",
  "B": "Situational",
  "C": "Competency",
  "K": "Skills",
  "P": "Personality",
  "S": "Simulation",
  "D": "Development"
};

function setStatus(label, state = "ready") {
  statusPill.className = "status-pill";
  if (state === "busy") statusPill.classList.add("busy");
  if (state === "error") statusPill.classList.add("error");
  statusPill.querySelector("span:last-child").textContent = label;
}

function escapeHtml(value) {
  if (!value) return "";
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function buildHtmlTable(rows) {
  // Filter out table separator lines (e.g., |---|---|)
  const cleanRows = rows.filter(row => !row.includes('---') && row.trim() !== '');
  if (cleanRows.length === 0) return '';
  
  let tableHtml = '<div class="table-responsive"><table class="markdown-table">';
  
  // Header row
  const headerCols = cleanRows[0]
    .split('|')
    .map(c => c.trim())
    .filter((c, idx, arr) => idx > 0 && idx < arr.length - 1);
    
  tableHtml += '<thead><tr>';
  headerCols.forEach(col => {
    tableHtml += `<th>${col}</th>`;
  });
  tableHtml += '</tr></thead><tbody>';
  
  // Body rows
  for (let i = 1; i < cleanRows.length; i++) {
    const cols = cleanRows[i]
      .split('|')
      .map(c => c.trim())
      .filter((c, idx, arr) => idx > 0 && idx < arr.length - 1);
      
    tableHtml += '<tr>';
    cols.forEach(col => {
      let cellContent = col;
      // Handle links inside tables
      cellContent = cellContent.replace(/&lt;(https?:\/\/.*?)&gt;/g, '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>');
      cellContent = cellContent.replace(/\[(.*?)\]\((.*?)\)/g, (match, linkText, url) => {
        const rawUrl = url.replaceAll("&amp;", "&");
        return `<a href="${rawUrl}" target="_blank" rel="noopener noreferrer">${linkText}</a>`;
      });
      tableHtml += `<td>${cellContent}</td>`;
    });
    tableHtml += '</tr>';
  }
  
  tableHtml += '</tbody></table></div>';
  return tableHtml;
}

function parseMarkdown(text) {
  let html = escapeHtml(text);

  // 1. Parse tables
  const lines = html.split('\n');
  let inTable = false;
  let tableRows = [];
  let processedLines = [];

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    if (line.startsWith('|') && line.endsWith('|')) {
      if (!inTable) {
        inTable = true;
        tableRows = [];
      }
      tableRows.push(lines[i]); // Keep original formatting to avoid stripping column bounds
    } else {
      if (inTable) {
        processedLines.push(buildHtmlTable(tableRows));
        inTable = false;
      }
      processedLines.push(lines[i]);
    }
  }
  if (inTable) {
    processedLines.push(buildHtmlTable(tableRows));
  }
  
  html = processedLines.join('\n');

  // 2. Headers
  html = html.replace(/^### (.*?)$/gm, '<h3>$1</h3>');
  html = html.replace(/^## (.*?)$/gm, '<h2>$1</h2>');
  html = html.replace(/^# (.*?)$/gm, '<h1>$1</h1>');

  // 3. Bold
  html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

  // 4. Bullet lists
  html = html.replace(/^\s*[-*]\s+(.*?)$/gm, '<li>$1</li>');
  html = html.replace(/^\s*\d+\.\s+(.*?)$/gm, '<li>$1</li>');

  // 5. Links
  html = html.replace(/\[(.*?)\]\((.*?)\)/g, (match, linkText, url) => {
    const rawUrl = url.replaceAll("&amp;", "&");
    return `<a href="${rawUrl}" target="_blank" rel="noopener noreferrer">${linkText}</a>`;
  });
  html = html.replace(/&lt;(https?:\/\/.*?)&gt;/g, (match, url) => {
    const rawUrl = url.replaceAll("&amp;", "&");
    return `<a href="${rawUrl}" target="_blank" rel="noopener noreferrer">${rawUrl}</a>`;
  });

  return html;
}

function addMessage(role, content, transient = false) {
  const wrapper = document.createElement("article");
  wrapper.className = `message ${role}`;
  if (transient) {
    wrapper.dataset.transient = "true";
  }

  const meta = document.createElement("div");
  meta.className = "message-meta";
  meta.textContent = role === "user" ? "You" : "Assistant";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  if (transient) {
    bubble.innerHTML = '<span class="typing"><span></span><span></span><span></span></span>';
  } else {
    bubble.innerHTML = parseMarkdown(content);
  }

  wrapper.append(meta, bubble);
  messageList.appendChild(wrapper);
  messageList.scrollTop = messageList.scrollHeight;
}

function removeTransientMessage() {
  const transient = messageList.querySelector("[data-transient='true']");
  if (transient) {
    transient.remove();
  }
}

function renderRecommendations(recommendations) {
  recCount.textContent = String(recommendations.length);

  if (!recommendations.length) {
    recommendationList.innerHTML = `
      <div class="empty-state">
        <p>No shortlist yet.</p>
        <span>Describe your hiring needs to see recommended assessments.</span>
      </div>
    `;
    return;
  }

  recommendationList.innerHTML = recommendations
    .map((rec) => {
      const name = escapeHtml(rec.name || "Untitled assessment");
      const url = escapeHtml(rec.url || "#");
      const type = escapeHtml(rec.test_type || "K");
      const label = typeLabels[type] || type;
      return `
        <a class="rec-card" href="${url}" target="_blank" rel="noopener noreferrer">
          <div class="rec-topline">
            <p class="rec-name">${name}</p>
            <span class="type-chip" data-type="${type}">${label}</span>
          </div>
          <p class="rec-url">${url}</p>
        </a>
      `;
    })
    .join("");
}

function setBusy(nextBusy) {
  busy = nextBusy;
  sendButton.disabled = busy;
  resetButton.disabled = busy;
  input.disabled = busy;
  setStatus(busy ? "Thinking" : "Ready", busy ? "busy" : "ready");
}

function resizeInput() {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 140)}px`;
}

async function sendMessage(text) {
  const content = text.trim();
  if (!content || busy) {
    return;
  }

  messages.push({ role: "user", content });
  addMessage("user", content);
  input.value = "";
  resizeInput();
  setBusy(true);
  addMessage("assistant", "", true);

  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages }),
    });

    if (!response.ok) {
      throw new Error(`Request failed with ${response.status}`);
    }

    const data = await response.json();
    removeTransientMessage();
    messages.push({ role: "assistant", content: data.reply });
    addMessage("assistant", data.reply);
    renderRecommendations(data.recommendations || []);
    setStatus(data.end_of_conversation ? "Complete" : "Ready");
  } catch (error) {
    removeTransientMessage();
    addMessage("assistant", "I could not reach the chat service. Check that the FastAPI server is running and try again.");
    setStatus("Error", "error");
  } finally {
    setBusy(false);
    input.focus();
  }
}

chatForm.addEventListener("submit", (event) => {
  event.preventDefault();
  sendMessage(input.value);
});

input.addEventListener("input", resizeInput);

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    chatForm.requestSubmit();
  }
});

resetButton.addEventListener("click", () => {
  messages = [];
  messageList.innerHTML = "";
  renderRecommendations([]);
  addMessage("assistant", welcomeText);
  setStatus("Ready");
  input.focus();
});

document.querySelectorAll("[data-prompt]").forEach((button) => {
  button.addEventListener("click", () => {
    input.value = button.dataset.prompt;
    resizeInput();
    input.focus();
  });
});

addMessage("assistant", welcomeText);
renderRecommendations([]);
resizeInput();
