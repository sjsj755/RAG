"use strict";

const STATUS_LABEL = {
  pending: "待处理",
  processing: "处理中",
  completed: "已完成",
  failed: "失败",
};

async function request(url, options = {}) {
  const resp = await fetch(url, options);
  let data = {};
  try {
    data = await resp.json();
  } catch (_) {
    /* 非 JSON 响应，保留空对象 */
  }
  if (!resp.ok) {
    const detail = Array.isArray(data.detail)
      ? data.detail.map((d) => d.msg).join("; ")
      : data.detail || `HTTP ${resp.status}`;
    throw new Error(detail);
  }
  return data;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function formatSize(bytes) {
  if (bytes == null || bytes === 0) return "-";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function formatTime(value) {
  if (!value) return "-";
  return new Date(value).toLocaleString();
}

function setMsg(elementId, text, type = "") {
  const el = document.getElementById(elementId);
  el.textContent = text || "";
  el.className = "msg" + (type ? ` ${type}` : "");
}

function renderResultItems(items) {
  return items
    .map((item) => {
      const hasFragment =
        item.fragment && item.fragment !== item.text;
      const body = hasFragment
        ? `
          <div class="fragment">${escapeHtml(item.fragment)}</div>
          <details class="result-context">
            <summary>查看完整上下文</summary>
            <p class="result-text">${escapeHtml(item.text)}</p>
          </details>`
        : `<p class="result-text">${escapeHtml(item.text)}</p>`;
      return `
        <div class="result-item">
          <div class="result-meta">
            <span>相关度 ${item.score == null ? "-" : `${(item.score * 100).toFixed(1)}%`}</span>
            <span>页码 ${item.page_num ?? "-"}</span>
            <span>类型 ${escapeHtml(item.type ?? "-")}</span>
          </div>
          ${body}
        </div>`;
    })
    .join("");
}

let documents = [];

async function refreshDocuments() {
  try {
    documents = await request("/api/v1/documents");
  } catch (err) {
    setMsg("doc-msg", `加载文档失败：${err.message}`, "error");
    return;
  }
  renderDocumentTable();
  renderSearchOptions();
}

function renderDocumentTable() {
  const tbody = document.getElementById("doc-tbody");
  if (documents.length === 0) {
    tbody.innerHTML = '<tr><td colspan="6" class="empty">暂无文档，请先上传 PDF</td></tr>';
    return;
  }

  tbody.innerHTML = documents
    .map((doc) => {
      const statusClass = `badge-${doc.status}`;
      const canIndex = doc.status === "pending" || doc.status === "failed";
      return `
        <tr>
          <td>${escapeHtml(doc.filename)}</td>
          <td>${formatSize(doc.file_size)}</td>
          <td><span class="badge ${statusClass}">${STATUS_LABEL[doc.status] ?? doc.status}</span></td>
          <td>${doc.total_chunks ?? 0}</td>
          <td>${formatTime(doc.created_at)}</td>
          <td style="white-space:nowrap">
            <button class="link" data-action="index" data-id="${doc.id}" ${canIndex ? "" : "disabled"}>索引</button>
            <button class="link" data-action="select" data-id="${doc.id}">检索</button>
            <button class="link danger" data-action="delete" data-id="${doc.id}">删除</button>
          </td>
        </tr>`;
    })
    .join("");
}

function renderSearchOptions() {
  const select = document.getElementById("search-doc");
  const previous = select.value;
  select.innerHTML =
    '<option value="">选择文档…</option>' +
    documents
      .map((doc) => `<option value="${doc.id}">${escapeHtml(doc.filename)}</option>`)
      .join("");
  if (previous && documents.some((doc) => doc.id === previous)) {
    select.value = previous;
  }
}

document.getElementById("upload-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const fileInput = document.getElementById("file-input");
  const btn = document.getElementById("upload-btn");
  const files = Array.from(fileInput.files || []);
  if (!files.length) return;

  const formData = new FormData();
  for (const file of files) {
    formData.append("files", file);
  }

  btn.disabled = true;
  setMsg("upload-msg", "上传中…");
  const resultsBox = document.getElementById("upload-results");
  resultsBox.innerHTML = "";
  try {
    const data = await request("/api/v1/upload/batch", {
      method: "POST",
      body: formData,
    });
    resultsBox.innerHTML = data.results
      .map((item) => {
        const ok = item.status === "uploaded";
        return `
          <div class="result-item">
            <div class="result-meta">
              <span>${ok ? "✅ 成功" : "❌ 失败"}</span>
              <span>${escapeHtml(item.filename)}</span>
              ${item.doc_id ? `<span>${escapeHtml(item.doc_id)}</span>` : ""}
              ${item.file_size ? `<span>${formatSize(item.file_size)}</span>` : ""}
            </div>
            ${item.error ? `<p class="result-text">${escapeHtml(item.error)}</p>` : ""}
          </div>`;
      })
      .join("");
    setMsg(
      "upload-msg",
      `上传完成：成功 ${data.succeeded} / 失败 ${data.failed}`,
      data.failed ? "error" : "ok"
    );
    fileInput.value = "";
    await refreshDocuments();
  } catch (err) {
    setMsg("upload-msg", `上传失败：${err.message}`, "error");
  } finally {
    btn.disabled = false;
  }
});

document.getElementById("doc-tbody").addEventListener("click", async (event) => {
  const btn = event.target.closest("button[data-action]");
  if (!btn) return;
  const { action, id } = btn.dataset;

  try {
    if (action === "index") {
      const result = await request(`/api/v1/index/${id}`, { method: "POST" });
      setMsg("doc-msg", result.message || "索引任务已提交", "ok");
      await refreshDocuments();
    } else if (action === "select") {
      document.getElementById("search-doc").value = id;
      document.getElementById("search-query").focus();
      document.getElementById("search-msg").textContent = "";
    } else if (action === "delete") {
      if (!confirm(`确定删除文档 ${id} 及其索引？`)) return;
      await request(`/api/v1/documents/${id}`, { method: "DELETE" });
      setMsg("doc-msg", "文档已删除", "ok");
      await refreshDocuments();
    }
  } catch (err) {
    setMsg("doc-msg", err.message, "error");
  }
});

document.getElementById("search-btn").addEventListener("click", async () => {
  const docId = document.getElementById("search-doc").value;
  const query = document.getElementById("search-query").value.trim();
  const topK = Number(document.getElementById("search-topk").value || 5);
  const resultsBox = document.getElementById("results");
  const answerBox = document.getElementById("answer-area");

  if (!docId) {
    setMsg("search-msg", "请先选择文档", "error");
    return;
  }
  if (!query) {
    setMsg("search-msg", "请输入检索问题", "error");
    return;
  }

  document.getElementById("search-btn").disabled = true;
  answerBox.innerHTML = "";
  setMsg("search-msg", "检索中…");
  try {
    const data = await request(`/api/v1/search/${docId}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, top_k: topK }),
    });
    const confidenceText =
      data.confidence == null ? "-" : `${(data.confidence * 100).toFixed(1)}%`;
    if (data.refused) {
      setMsg(
        "search-msg",
        `未找到足够可靠的答案（置信度 ${confidenceText}），以下为低置信度片段，请人工判断`,
        "warn"
      );
      resultsBox.innerHTML = `
        <details class="result-details">
          <summary>查看低置信度片段（${data.results.length} 条）</summary>
          ${renderResultItems(data.results)}
        </details>`;
    } else {
      setMsg(
        "search-msg",
        `找到 ${data.total} 个相关片段（置信度 ${confidenceText}）`,
        "ok"
      );
      resultsBox.innerHTML = renderResultItems(data.results);
    }
  } catch (err) {
    setMsg("search-msg", `检索失败：${err.message}`, "error");
  } finally {
    document.getElementById("search-btn").disabled = false;
  }
});

document.getElementById("answer-btn").addEventListener("click", async () => {
  const docId = document.getElementById("search-doc").value;
  const query = document.getElementById("search-query").value.trim();
  const topK = Number(document.getElementById("search-topk").value || 5);
  const answerBox = document.getElementById("answer-area");
  const resultsBox = document.getElementById("results");
  const btn = document.getElementById("answer-btn");

  if (!docId) {
    setMsg("search-msg", "请先选择文档", "error");
    return;
  }
  if (!query) {
    setMsg("search-msg", "请输入问题", "error");
    return;
  }

  btn.disabled = true;
  resultsBox.innerHTML = "";
  answerBox.innerHTML = "";
  setMsg("search-msg", "生成答案中…");

  let answerHtml = "";
  let sources = [];
  let refused = false;
  let refusalReason = "";
  let confidenceText = "-";

  const render = () => {
    answerBox.innerHTML = `
      <div class="answer-box">
        ${
          refused
            ? `<p class="msg warn">${escapeHtml(refusalReason)}（置信度 ${confidenceText}）</p>`
            : ""
        }
        ${answerHtml ? `<div class="answer-text">${answerHtml}</div>` : ""}
        ${
          sources.length
            ? `<div class="answer-sources">引用：${sources
                .map(
                  (s) =>
                    `<span class="source-chip" title="${escapeHtml(
                      (s.text || "").slice(0, 120)
                    )}">第 ${s.page_num ?? "-"} 页 [${s.index}]</span>`
                )
                .join("")}</div>`
            : ""
        }
      </div>`;
  };

  try {
    const resp = await fetch(`/api/v1/answer/${docId}?stream=true`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, top_k: topK }),
    });
    if (!resp.ok) {
      const data = await resp.json().catch(() => ({}));
      throw new Error(data.detail || `HTTP ${resp.status}`);
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let sep;
      while ((sep = buffer.indexOf("\n\n")) >= 0) {
        const block = buffer.slice(0, sep);
        buffer = buffer.slice(sep + 2);
        for (const line of block.split("\n")) {
          if (!line.startsWith("data: ")) continue;
          let event;
          try {
            event = JSON.parse(line.slice(6));
          } catch (_) {
            continue;
          }
          if (event.type === "sources") {
            sources = event.sources || [];
            confidenceText =
              event.confidence == null
                ? "-"
                : `${(event.confidence * 100).toFixed(1)}%`;
            render();
          } else if (event.type === "answer") {
            answerHtml += escapeHtml(event.content || "");
            render();
          } else if (event.type === "refused") {
            refused = true;
            refusalReason = event.reason || "";
            confidenceText =
              event.confidence == null
                ? "-"
                : `${(event.confidence * 100).toFixed(1)}%`;
            render();
          } else if (event.type === "error") {
            throw new Error(event.detail || "答案生成失败");
          }
        }
      }
    }
    setMsg(
      "search-msg",
      refused ? "已拒绝生成（检索置信度不足）" : "答案生成完成",
      refused ? "warn" : "ok"
    );
  } catch (err) {
    setMsg("search-msg", `答案生成失败：${err.message}`, "error");
    answerBox.innerHTML = `<div class="answer-box"><p class="msg error">${escapeHtml(err.message)}</p></div>`;
  } finally {
    btn.disabled = false;
  }
});

async function checkHealth() {
  try {
    const data = await request("/health");
    const el = document.getElementById("health");
    el.textContent = `服务正常 · ${data.service}`;
    el.className = "badge badge-completed";
  } catch (_) {
    const el = document.getElementById("health");
    el.textContent = "服务不可用";
    el.className = "badge badge-failed";
  }
}

refreshDocuments();
checkHealth();
setInterval(refreshDocuments, 3000);
