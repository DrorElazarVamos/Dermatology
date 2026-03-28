"use strict";

// ── Risk styling maps ────────────────────────────────────────────────────────
const RISK_COLOR = {
    high:   "var(--red)",
    medium: "var(--orange)",
    low:    "var(--green)",
};

const RISK_TEXT = {
    high:   "⚠  High Risk",
    medium: "◉  Moderate Risk",
    low:    "✓  Low Risk",
};

// ── State ────────────────────────────────────────────────────────────────────
let selectedFile = null;

// ── File input wiring ────────────────────────────────────────────────────────
document.getElementById("camera-input") .addEventListener("change", onFileSelected);
document.getElementById("gallery-input").addEventListener("change", onFileSelected);

function onFileSelected(e) {
    const file = e.target.files[0];
    if (!file) return;
    selectedFile = file;
    loadPreview(file);
    e.target.value = "";   // allow re-selecting the same file later
}

// ── Drag-and-drop support (desktop / iPad) ───────────────────────────────────
const dropZone = document.getElementById("drop-zone");

dropZone.addEventListener("dragover", e => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
});

dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));

dropZone.addEventListener("drop", e => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) {
        selectedFile = file;
        loadPreview(file);
    } else {
        showToast("Please drop an image file.");
    }
});

// ── Preview ───────────────────────────────────────────────────────────────────
function loadPreview(file) {
    const reader = new FileReader();
    reader.onload = ev => {
        document.getElementById("preview-img").src = ev.target.result;
        show("preview-section");
        hide("upload-section");
        hide("results-section");
        hide("loading-section");
    };
    reader.readAsDataURL(file);
}

// ── Analyse ───────────────────────────────────────────────────────────────────
async function analyze() {
    if (!selectedFile) return;

    hide("preview-section");
    show("loading-section");

    try {
        const form = new FormData();
        form.append("file", selectedFile);

        const resp = await fetch("/predict", { method: "POST", body: form });

        if (!resp.ok) {
            let detail = `Server error (${resp.status})`;
            try { detail = (await resp.json()).detail ?? detail; } catch (_) {}
            throw new Error(detail);
        }

        const data = await resp.json();
        renderResults(data);

    } catch (err) {
        hide("loading-section");
        show("preview-section");
        showToast(err.message);
    }
}

// ── Render results ────────────────────────────────────────────────────────────
function renderResults(data) {
    hide("loading-section");

    // Hero card
    const hero = document.getElementById("result-hero");
    hero.className = `result-hero risk-${data.risk}`;

    const labelEl = document.getElementById("result-label");
    labelEl.textContent = data.label;
    labelEl.style.color = RISK_COLOR[data.risk];

    document.getElementById("result-conf").textContent =
        `Confidence: ${data.confidence.toFixed(1)} %`;

    const badge = document.getElementById("result-badge");
    badge.textContent  = RISK_TEXT[data.risk];
    badge.className    = `result-badge badge-${data.risk}`;

    // Score bars
    const list = document.getElementById("scores-list");
    list.innerHTML = "";

    data.all_scores.forEach(({ label, confidence, risk }) => {
        const row = document.createElement("div");
        row.className = "score-row";
        row.innerHTML = `
            <span class="score-name" title="${label}">${label}</span>
            <div class="score-bar-bg">
                <div class="score-bar-fill" data-width="${confidence}"
                     style="background:${RISK_COLOR[risk]}"></div>
            </div>
            <span class="score-pct">${confidence.toFixed(1)}%</span>
        `;
        list.appendChild(row);
    });

    show("results-section");

    // Animate bars after paint (double rAF ensures layout is complete)
    requestAnimationFrame(() => requestAnimationFrame(() => {
        list.querySelectorAll(".score-bar-fill").forEach(el => {
            el.style.width = `${el.dataset.width}%`;
        });
    }));
}

// ── Reset ─────────────────────────────────────────────────────────────────────
function resetApp() {
    selectedFile = null;
    document.getElementById("preview-img").src = "";
    document.getElementById("scores-list").innerHTML = "";
    hide("preview-section");
    hide("results-section");
    hide("loading-section");
    show("upload-section");
}

// ── Toast notification ────────────────────────────────────────────────────────
let _toastTimer = null;

function showToast(msg) {
    const toast = document.getElementById("toast");
    toast.textContent = msg;
    toast.classList.remove("hidden");
    clearTimeout(_toastTimer);
    _toastTimer = setTimeout(() => toast.classList.add("hidden"), 4500);
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function show(id) { document.getElementById(id).classList.remove("hidden"); }
function hide(id) { document.getElementById(id).classList.add("hidden"); }
