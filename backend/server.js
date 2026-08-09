/**
 * server.js
 * ----------
 * Serves the CineGraph dashboard (static files + precomputed results.json)
 * and exposes one live endpoint, POST /api/predict, which calls into the
 * existing Python fuzzy-logic code via predict_service.py.
 *
 * All real analysis logic stays in Python where it was built and tested -
 * this server's job is narrow: serve the UI, and bridge one interactive
 * feature to the Python side. It does not reimplement any ML/fuzzy logic.
 *
 * Run:
 *     npm install
 *     npm start
 * Then open http://localhost:3000
 *
 * If "python" isn't on your PATH (e.g. some Linux/Mac setups need
 * "python3" instead), set the PYTHON_BIN environment variable:
 *     PYTHON_BIN=python3 npm start
 */

const express = require("express");
const path = require("path");
const { execFile } = require("child_process");

const app = express();
const PORT = process.env.PORT || 3000;
const PYTHON_BIN = process.env.PYTHON_BIN || "python";

const FRONTEND_DIR = path.join(__dirname, "..", "frontend");
const PREDICT_SCRIPT = path.join(__dirname, "..", "src", "predict_service.py");

app.use(express.json());
app.use(express.static(FRONTEND_DIR));

app.post("/api/predict", (req, res) => {
  const { budget, genre, studios } = req.body;

  if (budget === undefined || genre === undefined || studios === undefined) {
    return res.status(400).json({ error: "budget, genre, and studios are all required" });
  }
  if (isNaN(Number(budget)) || isNaN(Number(studios))) {
    return res.status(400).json({ error: "budget and studios must be numbers" });
  }

  const args = [PREDICT_SCRIPT, String(budget), String(genre), String(studios)];

  execFile(PYTHON_BIN, args, { timeout: 15000 }, (error, stdout, stderr) => {
    if (error) {
      console.error("predict_service.py failed:", error.message, stderr);
      return res.status(500).json({
        error: "Prediction service failed to run.",
        detail: stderr || error.message,
        hint: `Make sure '${PYTHON_BIN}' is on PATH and points to the venv with pandas/scikit-fuzzy installed. Set PYTHON_BIN env var to override.`,
      });
    }

    try {
      const result = JSON.parse(stdout.trim().split("\n").pop());
      if (result.error) {
        return res.status(500).json(result);
      }
      return res.json(result);
    } catch (parseErr) {
      console.error("Could not parse predict_service.py output:", stdout);
      return res.status(500).json({
        error: "Prediction service returned unexpected output.",
        raw_output: stdout,
      });
    }
  });
});

app.get("/api/health", (req, res) => {
  res.json({ status: "ok" });
});

app.listen(PORT, () => {
  console.log(`CineGraph server running at http://localhost:${PORT}`);
  console.log(`Using Python binary: ${PYTHON_BIN} (override with PYTHON_BIN env var)`);
});
