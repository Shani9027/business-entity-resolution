(() => {
  "use strict";

  const state = { selectedId: "", matchPage: 1, matchPageSize: 50, totalPages: 0 };
  const summary = document.getElementById("summary");
  const message = document.getElementById("message");
  const search = document.getElementById("search");
  const suggestions = document.getElementById("suggestions");
  const selected = document.getElementById("selected-business");
  const matches = document.getElementById("matches");
  const matchMessage = document.getElementById("match-message");

  function formatNumber(value) {
    return Number(value).toLocaleString();
  }

  function setMessage(value, isError = false) {
    message.textContent = value;
    message.className = isError ? "message error" : "message";
  }

  async function request(path) {
    const response = await fetch(path);
    let payload;
    try { payload = await response.json(); } catch (_) { payload = {}; }
    if (!response.ok) throw new Error(payload.error || "The server returned an error.");
    return payload;
  }

  function renderMeta(parent, label, value) {
    const item = document.createElement("div");
    const key = document.createElement("span");
    key.className = "meta-label";
    key.textContent = label;
    const text = document.createElement("strong");
    text.textContent = value;
    item.append(key, text);
    parent.append(item);
  }

  function renderSearchResults(data) {
    suggestions.replaceChildren();
    data.results.forEach((item) => {
      const option = document.createElement("button");
      option.type = "button";
      option.className = "suggestion";
      const name = document.createElement("strong");
      name.textContent = item.business_name;
      const id = document.createElement("span");
      id.className = "entity-meta";
      id.textContent = item.entity_id;
      option.append(name, id);
      option.onclick = () => {
        state.selectedId = item.entity_id;
        state.matchPage = 1;
        search.value = item.business_name;
        suggestions.replaceChildren();
        loadSelected();
      };
      suggestions.append(option);
    });
    if (!data.results.length && search.value.trim()) {
      const empty = document.createElement("p");
      empty.className = "muted suggestion-empty";
      empty.textContent = "No Source-1 business found.";
      suggestions.append(empty);
    }
  }

  async function searchBusinesses() {
    const query = search.value.trim();
    if (!query) {
      suggestions.replaceChildren();
      return;
    }
    try {
      renderSearchResults(await request(`/api/search?q=${encodeURIComponent(query)}&page_size=8`));
    } catch (error) {
      suggestions.replaceChildren();
      setMessage(`${error.message} Please retry.`, true);
    }
  }

  function renderSelected(data) {
    selected.style.display = "block";
    selected.replaceChildren();
    const heading = document.createElement("div");
    heading.className = "selected-heading";
    const name = document.createElement("h2");
    name.textContent = data.source1_business_name;
    const id = document.createElement("span");
    id.className = "entity-meta";
    id.textContent = data.source1_entity_id;
    heading.append(name, id);
    const details = document.createElement("div");
    details.className = "selected-meta";
    renderMeta(details, "Address", data.source1_business_address);
    renderMeta(details, "Country", data.source1_country);
    renderMeta(details, "Predicted matches", formatNumber(data.match_count));
    const badge = document.createElement("span");
    badge.className = `status ${data.status}`;
    badge.textContent = data.status;
    heading.append(badge);
    selected.append(heading, details);
  }

  function renderMatches(data) {
    matches.replaceChildren();
    state.totalPages = data.total_pages;
    document.getElementById("match-page-label").textContent =
      data.total_pages ? `Page ${data.page} of ${data.total_pages}` : "";
    document.getElementById("match-previous").disabled = data.page <= 1;
    document.getElementById("match-next").disabled =
      data.page >= data.total_pages || data.total_pages === 0;
    if (!data.results.length) {
      matchMessage.textContent = "No matches were predicted for this business.";
      return;
    }
    matchMessage.textContent = `${formatNumber(data.total_records)} predicted match${data.total_records === 1 ? "" : "es"}`;
    data.results.forEach((item) => {
      const row = document.createElement("tr");
      const name = document.createElement("td");
      const strong = document.createElement("strong");
      strong.textContent = item.business_name;
      const id = document.createElement("span");
      id.className = "entity-meta";
      id.textContent = item.entity_id;
      name.append(strong, id);
      const source = document.createElement("td");
      source.textContent = item.source;
      const address = document.createElement("td");
      address.textContent = item.business_address;
      const country = document.createElement("td");
      country.textContent = item.country;
      row.append(name, source, address, country);
      matches.append(row);
    });
  }

  async function loadSelected() {
    if (!state.selectedId) return;
    setMessage("Loading business matches...");
    try {
      const data = await request(`/api/results/${encodeURIComponent(state.selectedId)}`);
      renderSelected(data);
      const params = new URLSearchParams({
        page: String(state.matchPage),
        page_size: String(state.matchPageSize),
      });
      renderMatches(await request(`/api/results/${encodeURIComponent(state.selectedId)}/matches?${params}`));
      setMessage("");
    } catch (error) {
      setMessage(`${error.message} Please retry.`, true);
    }
  }

  function renderSummary(data) {
    summary.replaceChildren();
    [
      ["Total Records", formatNumber(data.total_records)],
      ["Matched Records", formatNumber(data.matched_records)],
      ["Unmatched Records", formatNumber(data.unmatched_records)],
      ["Match Rate", `${(data.match_rate * 100).toFixed(1)}%`],
    ].forEach(([label, value]) => {
      const card = document.createElement("article");
      card.className = "summary-card";
      const title = document.createElement("span");
      title.className = "summary-label";
      title.textContent = label;
      const number = document.createElement("strong");
      number.className = "summary-value";
      number.textContent = value;
      card.append(title, number);
      summary.append(card);
    });
  }

  async function loadSummary() {
    try { renderSummary(await request("/api/summary")); }
    catch (error) {
      summary.replaceChildren();
      setMessage(`${error.message} Please retry.`, true);
    }
  }

  let debounceTimer;
  search.addEventListener("input", () => {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(searchBusinesses, 300);
  });
  document.getElementById("find-matches").addEventListener("click", searchBusinesses);
  document.getElementById("match-previous").addEventListener("click", () => {
    if (state.matchPage > 1) { state.matchPage -= 1; loadSelected(); }
  });
  document.getElementById("match-next").addEventListener("click", () => {
    if (state.matchPage < state.totalPages) { state.matchPage += 1; loadSelected(); }
  });
  document.addEventListener("click", (event) => {
    if (!event.target.closest(".search-box")) suggestions.replaceChildren();
  });

  loadSummary();
})();
