// ============================================================
// CineGraph dashboard — fetches precomputed results.json and
// renders every section; wires the fuzzy advisor form to the
// live /api/predict endpoint.
// ============================================================

const fmtMoney = (n) => {
  if (n === null || n === undefined || isNaN(n)) return "—";
  if (Math.abs(n) >= 1e9) return `$${(n / 1e9).toFixed(2)}B`;
  if (Math.abs(n) >= 1e6) return `$${(n / 1e6).toFixed(1)}M`;
  if (Math.abs(n) >= 1e3) return `$${(n / 1e3).toFixed(0)}K`;
  return `$${n.toFixed(0)}`;
};
const fmtNum = (n, decimals = 0) =>
  n === null || n === undefined || isNaN(n) ? "—" : Number(n).toFixed(decimals);
const escapeHtml = (s) =>
  String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

let RESULTS = null;

async function loadResults() {
  try {
    const res = await fetch("data/results.json");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    RESULTS = await res.json();
    renderAll();
  } catch (err) {
    console.error("Failed to load results.json:", err);
    document.querySelectorAll(".loading-card").forEach((el) => {
      el.textContent = "Couldn't load results.json — run 'python src/export_results.py' first.";
    });
  }
}

function renderAll() {
  renderMarquee();
  renderHero();
  renderRegression();
  renderFuzzyStatic();
  renderAssociationRules();
  renderGraph();
  renderAnomalies();
}

// ---------------- Marquee ----------------
function renderMarquee() {
  const items = [];
  const rules = (RESULTS.association_rules && RESULTS.association_rules.person_rules) || [];
  rules.slice(0, 3).forEach((r) => {
    const who = r.antecedents.join(", ").replace(/^(Actor_|Director_)/, "");
    items.push(`${who.toUpperCase()}: ${(r.confidence * 100).toFixed(0)}% HIGH-SUCCESS RATE`);
  });
  const flop = RESULTS.anomalies && RESULTS.anomalies.flops && RESULTS.anomalies.flops[0];
  if (flop) items.push(`${flop.title.toUpperCase()}: ${fmtMoney(flop.budget)} BUDGET \u2192 ${fmtMoney(flop.revenue)}`);
  const sleeper = RESULTS.anomalies && RESULTS.anomalies.sleeper_hits && RESULTS.anomalies.sleeper_hits[0];
  if (sleeper) items.push(`${sleeper.title.toUpperCase()}: ${fmtMoney(sleeper.budget)} BUDGET \u2192 ${fmtMoney(sleeper.revenue)} SLEEPER HIT`);
  const overview = RESULTS.overview;
  if (overview) items.push(`${overview.total_movies} FILMS ANALYZED ACROSS ${overview.date_range[0]}\u2013${overview.date_range[1]}`);

  if (items.length === 0) items.push("WELCOME TO CINEGRAPH");

  const track = document.getElementById("marqueeTrack");
  const html = items.map((t) => `<span class="marquee-item">${escapeHtml(t)}</span>`).join("");
  track.innerHTML = html + html; // duplicate for seamless scroll loop
}

// ---------------- Hero ----------------
function renderHero() {
  const o = RESULTS.overview;
  if (!o) return;
  const blocks = document.querySelectorAll("#heroStats .stat-number");
  blocks[0].textContent = o.total_movies.toLocaleString();
  blocks[1].textContent = `${o.date_range[0]}–${o.date_range[1]}`;
  blocks[2].textContent = fmtMoney(o.total_revenue_tracked);
}

// ---------------- Regression ----------------
function renderRegression() {
  const r = RESULTS.regression;
  if (!r) return;

  const cardsEl = document.getElementById("regressionCards");
  cardsEl.innerHTML = `
    <div class="metric-card">
      <h4>Hindsight model</h4>
      <div class="metric-row"><span>R&sup2;</span><span class="metric-value">${fmtNum(r.hindsight.r2, 3)}</span></div>
      <div class="metric-row"><span>Mean error</span><span class="metric-value">${fmtMoney(r.hindsight.mae)}</span></div>
      <div class="metric-row"><span>Includes</span><span>vote_count, popularity</span></div>
    </div>
    <div class="metric-card dim">
      <h4>Pre-release model (honest)</h4>
      <div class="metric-row"><span>R&sup2;</span><span class="metric-value">${fmtNum(r.pre_release.r2, 3)}</span></div>
      <div class="metric-row"><span>Mean error</span><span class="metric-value">${fmtMoney(r.pre_release.mae)}</span></div>
      <div class="metric-row"><span>Includes</span><span>budget, runtime, genre only</span></div>
    </div>
  `;

  const maxImportance = Math.max(...r.pre_release.top_features.map((f) => f.importance));
  const barsEl = document.getElementById("featureBars");
  barsEl.innerHTML = r.pre_release.top_features
    .map(
      (f) => `
    <div class="feature-bar-row">
      <span>${escapeHtml(f.name.replace("genre_", "Genre: "))}</span>
      <div class="feature-bar-track"><div class="feature-bar-fill" style="width:${(f.importance / maxImportance) * 100}%"></div></div>
      <span class="feature-bar-value">${(f.importance * 100).toFixed(1)}%</span>
    </div>`
    )
    .join("");
}

// ---------------- Fuzzy (static parts: genre bars, genre select) ----------------
function renderFuzzyStatic() {
  const f = RESULTS.fuzzy;
  if (!f) return;

  const select = document.getElementById("genreSelect");
  select.innerHTML = f.genre_popularity
    .map((g) => `<option value="${escapeHtml(g.genre)}">${escapeHtml(g.genre)}</option>`)
    .join("");

  const maxScore = Math.max(...f.genre_popularity.map((g) => g.score));
  const genreEl = document.getElementById("genrePopularity");
  genreEl.innerHTML = f.genre_popularity
    .slice(0, 12)
    .map(
      (g) => `
    <div class="genre-bar-row">
      <span>${escapeHtml(g.genre)}</span>
      <div class="genre-bar-track"><div class="genre-bar-fill" style="width:${(g.score / maxScore) * 100}%"></div></div>
      <span class="genre-bar-value">${fmtNum(g.score, 0)}</span>
    </div>`
    )
    .join("");
}

// ---------------- Association rules ----------------
function ruleCard(rule) {
  const cleanName = (s) => s.replace(/^(Genre_|Actor_|Director_)/, "");
  const ant = rule.antecedents.map(cleanName).join(" + ");
  const cons = rule.consequents.map(cleanName).join(" + ");
  return `
    <div class="rule-card">
      <p class="rule-line">IF <b>${escapeHtml(ant)}</b> <span class="rule-arrow">&rarr;</span> THEN <b>${escapeHtml(cons)}</b></p>
      <div class="rule-stats">
        <span>Support <b>${fmtNum(rule.support, 3)}</b></span>
        <span>Confidence <b>${fmtNum(rule.confidence * 100, 0)}%</b></span>
        <span>Lift <b>${fmtNum(rule.lift, 2)}&times;</b></span>
      </div>
    </div>`;
}

function renderAssociationRules() {
  const a = RESULTS.association_rules;
  if (!a) return;

  const genreEl = document.getElementById("genreRules");
  genreEl.innerHTML = a.genre_rules.length
    ? a.genre_rules.slice(0, 8).map(ruleCard).join("")
    : `<p class="panel-note">No genre-level rules met the support threshold.</p>`;

  const personEl = document.getElementById("personRules");
  personEl.innerHTML = a.person_rules.length
    ? a.person_rules.slice(0, 8).map(ruleCard).join("")
    : `<p class="panel-note">No individual actor/director rules met the (lower) support threshold on this run — a legitimate finding in itself: genre and budget carry more signal than any one person.</p>`;
}

// ---------------- Graph ----------------
function renderGraph() {
  const g = RESULTS.graph;
  if (!g) return;

  const bridgesEl = document.getElementById("bridgesList");
  bridgesEl.innerHTML = g.top_bridges
    .slice(0, 10)
    .map((p) => `<li><span>${escapeHtml(p.person)}</span><span class="rank-score">${fmtNum(p.score, 4)}</span></li>`)
    .join("");

  const collabEl = document.getElementById("collaboratorsList");
  collabEl.innerHTML = g.top_collaborators
    .slice(0, 10)
    .map((p) => `<li><span>${escapeHtml(p.person)}</span><span class="rank-score">${p.collaborations}</span></li>`)
    .join("");

  const commEl = document.getElementById("communitiesGrid");
  commEl.innerHTML = g.communities
    .map(
      (c, i) => `
    <div class="community-card">
      <h4>Community ${i + 1} &middot; ${c.size} people</h4>
      <ul>${c.sample_members.slice(0, 6).map((m) => `<li>${escapeHtml(m.replace(/^(Actor: |Director: )/, ""))}</li>`).join("")}</ul>
    </div>`
    )
    .join("");
}

// ---------------- Anomalies ----------------
function movieCard(m, direction) {
  const predicted = m.predicted_revenue;
  const arrow = direction === "up" ? "&uarr;" : "&darr;";
  const cls = direction === "up" ? "up" : "down";
  return `
    <div class="movie-card">
      <div>
        <span class="movie-card-title">${escapeHtml(m.title)}</span>
        <span class="movie-card-year"> (${m.year})</span>
      </div>
      <div class="movie-card-figures">
        Budget ${fmtMoney(m.budget)} &middot; Actual ${fmtMoney(m.revenue)}
        ${predicted !== undefined && predicted !== null ? ` &middot; Predicted ${fmtMoney(predicted)} <span class="${cls}">${arrow}</span>` : ""}
      </div>
    </div>`;
}

function renderAnomalies() {
  const a = RESULTS.anomalies;
  if (!a) return;

  document.getElementById("sleeperHits").innerHTML = a.sleeper_hits.map((m) => movieCard(m, "up")).join("");
  document.getElementById("flops").innerHTML = a.flops.map((m) => movieCard(m, "down")).join("");
  document.getElementById("outliers").innerHTML = a.outliers
    .map(
      (m) => `
    <div class="movie-card">
      <div><span class="movie-card-title">${escapeHtml(m.title)}</span><span class="movie-card-year"> (${m.year})</span></div>
      <div class="movie-card-figures">Budget ${fmtMoney(m.budget)} &middot; Revenue ${fmtMoney(m.revenue)} &middot; Rating ${fmtNum(m.vote_average, 1)}</div>
    </div>`
    )
    .join("");
}

// ---------------- Live fuzzy advisor ----------------
function setupAdvisorForm() {
  const budgetSlider = document.getElementById("budgetSlider");
  const budgetDisplay = document.getElementById("budgetDisplay");
  const studiosSlider = document.getElementById("studiosSlider");
  const studiosDisplay = document.getElementById("studiosDisplay");
  const form = document.getElementById("advisorForm");
  const resultEl = document.getElementById("advisorResult");

  budgetSlider.addEventListener("input", () => {
    budgetDisplay.textContent = `$${Number(budgetSlider.value).toLocaleString()}`;
  });
  studiosSlider.addEventListener("input", () => {
    studiosDisplay.textContent = studiosSlider.value;
  });

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const submitBtn = form.querySelector("button[type=submit]");
    submitBtn.disabled = true;
    submitBtn.textContent = "Running…";
    resultEl.innerHTML = `<p class="advisor-placeholder">Computing live prediction&hellip;</p>`;

    const payload = {
      budget: Number(budgetSlider.value),
      genre: document.getElementById("genreSelect").value,
      studios: Number(studiosSlider.value),
    };

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();

      if (!res.ok || data.error) {
        resultEl.innerHTML = `<p class="result-error">Prediction failed: ${escapeHtml(data.error || "unknown error")}${data.hint ? `<br><small>${escapeHtml(data.hint)}</small>` : ""}</p>`;
      } else {
        const catClass = data.category.toLowerCase();
        resultEl.innerHTML = `
          <div class="result-display">
            <span class="result-score">${fmtNum(data.score, 1)}</span>
            <span class="result-category ${catClass}">${escapeHtml(data.category)}</span>
          </div>
          <p class="result-detail">
            Genre popularity score: ${fmtNum(data.genre_popularity, 1)}/100
            ${data.genre_recognized ? "" : " (genre not seen in training data — used neutral default)"}
          </p>`;
      }
    } catch (err) {
      resultEl.innerHTML = `<p class="result-error">Couldn't reach the prediction server. Is the Node backend running?</p>`;
    } finally {
      submitBtn.disabled = false;
      submitBtn.textContent = "Run the Advisor";
    }
  });
}

// ---------------- Init ----------------
document.addEventListener("DOMContentLoaded", () => {
  setupAdvisorForm();
  loadResults();
});
