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
  if (!fileInput.files.length) return;

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  btn.disabled = true;
  setMsg("upload-msg", "上传中…");
  try {
    const doc = await request("/api/v1/upload", { method: "POST", body: formData });
    setMsg("upload-msg", `上传成功：${doc.filename}（${doc.id}）`, "ok");
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

  if (!docId) {
    setMsg("search-msg", "请先选择文档", "error");
    return;
  }
  if (!query) {
    setMsg("search-msg", "请输入检索问题", "error");
    return;
  }

  document.getElementById("search-btn").disabled = true;
  setMsg("search-msg", "检索中…");
  try {
    const data = await request(`/api/v1/search/${docId}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, top_k: topK }),
    });
    setMsg("search-msg", `找到 ${data.total} 个相关片段`, "ok");
    resultsBox.innerHTML = data.results
      .map(
        (item) => `
          <div class="result-item">
            <div class="result-meta">
              <span>相关度 ${item.score == null ? "-" : `${(item.score * 100).toFixed(1)}%`}</span>
              <span>页码 ${item.page_num ?? "-"}</span>
              <span>类型 ${escapeHtml(item.type ?? "-")}</span>
            </div>
            <p class="result-text">${escapeHtml(item.text)}</p>
          </div>`
      )
      .join("");
  } catch (err) {
    setMsg("search-msg", `检索失败：${err.message}`, "error");
  } finally {
    document.getElementById("search-btn").disabled = false;
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
