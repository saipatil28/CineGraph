# CineGraph — Movie Industry Intelligence Platform

A full-stack data mining and machine learning platform that analyzes ~4,800 films to uncover what actually predicts box-office success — combining regression, fuzzy logic, association rule mining, graph/network analysis, and anomaly detection into one interactive dashboard.


---

## What this does

Given a dataset of movies (budget, revenue, genre, cast, crew, ratings), CineGraph answers five different questions using five different techniques:

| Technique | Question it answers |
|---|---|
| **Regression** | How well can we predict a movie's revenue — before vs. after it's released? |
| **Fuzzy Logic** | What's the "success likelihood" of a movie, reasoned the way a studio executive would? |
| **Association Rule Mining** | Which genre/cast/director combinations show up together far more than chance? |
| **Graph Analysis** | Who are the most-connected people in the industry, and what collaboration circles exist? |
| **Anomaly Detection** | Which movies were the biggest sleeper hits and flops relative to expectations? |

All five techniques run on the same real dataset and feed into one dashboard, rather than being five disconnected demos.

---

## Key findings

- **Pre-release revenue prediction (honest model, no hindsight features):** R² ≈ 0.52 — mostly driven by budget
- **Hindsight model (includes post-release popularity):** R² ≈ 0.72 — inflated, and explicitly flagged as such in the dashboard, since `vote_count` is only known *after* release
- **Association rules correctly discovered, with no prior knowledge fed in:** Steven Spielberg's films land in the top revenue quartile 85% of the time; Tom Cruise's films, 83%
- **Sleeper hits found:** *The Passion of the Christ* ($30M budget → $612M revenue), *My Big Fat Greek Wedding*, *Slumdog Millionaire*
- **Notable flops:** *Foodfight!* ($65M budget → $73K revenue)

---

## Tech stack

**Analysis (Python)**
- `pandas`, `numpy` — data cleaning and manipulation
- `scikit-learn` — regression (Linear + Random Forest) and Isolation Forest anomaly detection
- `scikit-fuzzy` — fuzzy inference system for the success advisor
- `mlxtend` — Apriori association rule mining
- `networkx` — collaboration graph, centrality, community detection

**Backend**
- `Express.js` (Node) — serves the dashboard and exposes one live endpoint (`/api/predict`) that bridges to the Python fuzzy-logic code via subprocess

**Frontend**
- Vanilla HTML/CSS/JS — no framework, no build step
- Custom cinema/trade-report visual design (dark theme, marquee ticker, film-strip motifs)

---

## Project structure
Claude finished the response

Orchestrated deployment strategy balancing hybrid stack complexities
Orchestrated deployment strategy balancing hybrid stack complexities
Great — first let's do a quick sanity check that the advisor is fully working (screenshot if you have a second), then let's get this onto GitHub, and after that we'll deploy it live. Let's do this in stages so nothing gets lost like the file issues earlier.

Stage 1: Clean up .gitignore before committing anything
Your current .gitignore is missing a few things that would otherwise bloat your repo — node_modules (huge, regenerable) and the .pkl model files (large binary, also regenerable by re-running your scripts). Open .gitignore in VS Code and replace its contents with:

venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
data/raw/*.csv
data/processed/*.csv
data/processed/*.pkl
.env
.vscode/
.DS_Store
node_modules/
backend/node_modules/
Save it (Ctrl+S — same reminder as always).

Note: frontend/data/results.json will still get committed — that's intentional, since your static dashboard needs it to actually show data when deployed.

Stage 2: Initialize git and push to GitHub
In your terminal, from the CineGraph root folder (not inside backend):

powershell
cd C:\Users\ASUS\Desktop\CineGraph
git init
git add .
git commit -m "Initial commit: CineGraph movie industry intelligence platform"
Then create a new empty repository on GitHub:

Go to github.com → click the + icon (top right) → New repository
Name it CineGraph (or whatever you like)
Do NOT check "Add a README" or ".gitignore" (you already have your own) — leave it empty
Click Create repository
GitHub will show you commands to connect your local repo — copy the ones under "…or push an existing repository from the command line," which look like:

powershell
git remote add origin https://github.com/saipatil28/CineGraph.git
git branch -M main
git push -u origin main
Run those. Paste me the output — once your code is safely on GitHub, we'll tackle deployment (which needs a slightly different plan since your project mixes Node + Python, not just a static site).

give me git ignore

venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
data/raw/*.csv
data/processed/*.csv
data/processed/*.pkl
.env
.vscode/
.DS_Store
node_modules/
backend/node_modules/
How to update it:

Click .gitignore in your VS Code Explorer sidebar (it's in the CineGraph root)
Select all existing text (Ctrl+A) and delete it
Paste the content above
Save it — Ctrl+S (check the tab has no unsaved dot after)
Once that's saved, continue with the git commands from before:

powershell
cd C:\Users\ASUS\Desktop\CineGraph
git init
git add .
git commit -m "Initial commit: CineGraph movie industry intelligence platform"
Paste me the output of these once you run them.

give me redme file for this project

Here's a comprehensive README reflecting everything you actually built, including the honest caveats you found along the way — those make it more credible, not less.

markdown
# 🎬 CineGraph — Movie Industry Intelligence Platform

A full-stack data mining and machine learning platform that analyzes ~4,800 films to uncover what actually predicts box-office success — combining regression, fuzzy logic, association rule mining, graph/network analysis, and anomaly detection into one interactive dashboard.

---

## What this does

Given a dataset of movies (budget, revenue, genre, cast, crew, ratings), CineGraph answers five different questions using five different techniques:

| Technique | Question it answers |
|---|---|
| **Regression** | How well can we predict a movie's revenue — before vs. after it's released? |
| **Fuzzy Logic** | What's the "success likelihood" of a movie, reasoned the way a studio executive would? |
| **Association Rule Mining** | Which genre/cast/director combinations show up together far more than chance? |
| **Graph Analysis** | Who are the most-connected people in the industry, and what collaboration circles exist? |
| **Anomaly Detection** | Which movies were the biggest sleeper hits and flops relative to expectations? |

All five techniques run on the same real dataset and feed into one dashboard, rather than being five disconnected demos.

---

## Key findings

- **Pre-release revenue prediction (honest model, no hindsight features):** R² ≈ 0.52 — mostly driven by budget
- **Hindsight model (includes post-release popularity):** R² ≈ 0.72 — inflated, and explicitly flagged as such in the dashboard, since `vote_count` is only known *after* release
- **Association rules correctly discovered, with no prior knowledge fed in:** Steven Spielberg's films land in the top revenue quartile 85% of the time; Tom Cruise's films, 83%
- **Sleeper hits found:** *The Passion of the Christ* ($30M budget → $612M revenue), *My Big Fat Greek Wedding*, *Slumdog Millionaire*
- **Notable flops:** *Foodfight!* ($65M budget → $73K revenue)

---

## Tech stack

**Analysis (Python)**
- `pandas`, `numpy` — data cleaning and manipulation
- `scikit-learn` — regression (Linear + Random Forest) and Isolation Forest anomaly detection
- `scikit-fuzzy` — fuzzy inference system for the success advisor
- `mlxtend` — Apriori association rule mining
- `networkx` — collaboration graph, centrality, community detection

**Backend**
- `Express.js` (Node) — serves the dashboard and exposes one live endpoint (`/api/predict`) that bridges to the Python fuzzy-logic code via subprocess

**Frontend**
- Vanilla HTML/CSS/JS — no framework, no build step
- Custom cinema/trade-report visual design (dark theme, marquee ticker, film-strip motifs)

---

## Project Structure

```text
CineGraph/
├── data/
│   ├── raw/                      # tmdb_5000_movies.csv, tmdb_5000_credits.csv
│   └── processed/                # cleaned data + trained models (generated, gitignored)
│
├── src/
│   ├── data_prep.py              # cleans and merges raw data
│   ├── regression.py             # hindsight vs. pre-release revenue models
│   ├── fuzzy_logic.py            # success-likelihood advisor
│   ├── association_rules.py      # genre/cast/director pattern mining
│   ├── graph_analysis.py         # collaboration network + centrality
│   ├── anomaly_detection.py      # sleeper hits, flops, statistical outliers
│   ├── export_results.py         # exports analytics to frontend/data/results.json
│   └── predict_service.py        # CLI bridge called by the Node backend
│
├── backend/
│   ├── server.js                 # Express server + live prediction endpoint
│   └── package.json
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   ├── script.js
│   └── data/
│       └── results.json          # precomputed analytics results (generated)
│
├── requirements.txt
└── README.md
```


---

## Setup

### 1. Python environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt
```

### 2. Get the data
Download [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) from Kaggle and place `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` into `data/raw/`.

### 3. Run the analysis pipeline (in order)
```bash
python src/data_prep.py
python src/regression.py
python src/export_results.py
```
The other modules (`fuzzy_logic.py`, `association_rules.py`, `graph_analysis.py`, `anomaly_detection.py`) can also be run standalone for their own console output.

### 4. Start the dashboard
```bash
cd backend
npm install
$env:PYTHON_BIN = "path\to\your\venv\Scripts\python.exe"   # Windows PowerShell
node server.js
```
Open **http://localhost:3000**.

---

## Honest limitations

- **Pre-release R² of 0.52** means over half of revenue variance is *not* explained by budget/runtime/genre alone — real box-office outcomes depend on factors this dataset doesn't capture (marketing spend, critical reception, cultural timing).
- **Association rules for individual actors/directors** are based on modest sample sizes (a handful of films each) and should be read as suggestive patterns, not statistically robust claims.
- **Graph community detection** on a dense collaboration network tends to produce a mix of tight, interpretable clusters and looser genre/era-based groupings — both are shown, but only the tighter ones represent literal repeat-collaborator ensembles.
- **The dataset caps around 2017**, so recent films aren't represented.

## Credits

Built with `pandas`, `scikit-learn`, `scikit-fuzzy`, `mlxtend`, `networkx`, and `Express.js`. Dataset: [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) via Kaggle.
