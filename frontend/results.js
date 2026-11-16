(function () {
    // Demo login check (not real security). Real sign-in comes later.
    if (localStorage.getItem("preprocessaiLoggedIn") !== "true") {
        window.location.href = "login.html";
        return;
    }

    const SPEC_KEYS = ["imputer", "encoder", "scaler", "outlier", "imbalance"];
    const SCORE_KEYS = ["performance", "efficiency", "simplicity", "robustness", "explainability"];
    const LABELS = {
        imputer: "Missing values",
        encoder: "Categories to numbers",
        scaler: "Scaling",
        outlier: "Outliers",
        imbalance: "Class imbalance",
        performance: "Performance",
        efficiency: "Efficiency",
        simplicity: "Simplicity",
        robustness: "Robustness",
        explainability: "Explainability"
    };

    let result = null;
    try {
        result = JSON.parse(sessionStorage.getItem("preprocessaiResult"));
    } catch (ignored) {
        result = null;
    }

    const noResult = document.getElementById("noResult");
    const resultArea = document.getElementById("resultArea");

    if (!result || !result.recommendation_id) {
        noResult.hidden = false;
        return;
    }
    resultArea.hidden = false;

    function make(tag, text, className) {
        const node = document.createElement(tag);
        if (text !== undefined && text !== null) {
            node.textContent = text;
        }
        if (className) {
            node.className = className;
        }
        return node;
    }

    function number(value) {
        const n = Number(value);
        return value === null || value === undefined || Number.isNaN(n) ? "n/a" : n.toFixed(2);
    }

    function sameSpec(row) {
        return SPEC_KEYS.every(function (key) {
            return row[key] === result.recommended_spec[key];
        });
    }

    // Dataset line
    document.getElementById("datasetInfo").textContent =
        "File: " + (result.fileName || "your dataset") +
        "  |  Target: " + (result.target || "?") +
        "  |  Task: " + (result.task || "?");

    // Recommended pipeline
    const specGrid = document.getElementById("specGrid");
    const specCards = {}
    SPEC_KEYS.forEach(function (key) {
        const card = make("div", null, "spec-item");
        specCards[key] = card
        card.appendChild(make("span", LABELS[key], "spec-label"));
        card.appendChild(make("strong", String((result.recommended_spec || {})[key] ?? "n/a")));
        specGrid.appendChild(card);
    });

    // Scores as bars
    const scoresBox = document.getElementById("scores");
    SCORE_KEYS.forEach(function (key) {
        const value = (result.scores || {})[key];
        const row = make("div", null, "score-row");
        row.appendChild(make("span", LABELS[key], "score-name"));
        const track = make("div", null, "score-track");
        const fill = make("div", null, "score-fill");
        const percent = Math.max(0, Math.min(1, Number(value) || 0)) * 100;
        fill.style.width = percent + "%";
        track.appendChild(fill);
        row.appendChild(track);
        row.appendChild(make("span", number(value), "score-value"));
        scoresBox.appendChild(row);
    });

    const STATUS_TEXT = { applied: "Applied", skipped: "Skipped", not_needed: "Not needed", not_applicable: "Not applicable" };
    function applyStepStatus(steps) {
        if (!steps) { return; }
        SPEC_KEYS.forEach(function (key) {
            const info = steps[key];
            const card = specCards[key];
            if (!info || !card) { return; }
            if (info.status === "not_needed" || info.status === "not_applicable") {
                card.querySelector("strong").textContent = STATUS_TEXT[info.status];
                card.classList.add("step-inactive");
            }
            card.appendChild(make("span", STATUS_TEXT[info.status] || info.status, "step-badge step-" + info.status));
            if (info.note) { card.appendChild(make("small", info.note, "step-note")); }
        });
    }

    // Explanation
    async function loadExplanation() {
        const box = document.getElementById("explanation");
        try {
            const response = await fetch(API_URL + "/explain", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ recommendation_id: result.recommendation_id })
            });

            if (response.status === 404) {
                box.textContent = "This recommendation has expired. Please upload your file again.";
                return;
            }
            if (!response.ok) {
                box.textContent = "The explanation could not be loaded (error " + response.status + ").";
                return;
            }

            const data = await response.json();
            applyStepStatus(data.steps);
            box.textContent = "";
            box.appendChild(make("p", data.summary || ""));

            [["Reasons", data.reasons], ["Trade-offs", data.tradeoffs]].forEach(function (pair) {
                const items = Array.isArray(pair[1]) ? pair[1] : [];
                if (items.length === 0) {
                    return;
                }
                box.appendChild(make("h3", pair[0]));
                const list = make("ul");
                items.forEach(function (text) {
                    list.appendChild(make("li", text));
                });
                box.appendChild(list);
            });
        } catch (error) {
            box.textContent = "Cannot reach the API at " + API_URL + ". Is it running?";
        }
    }
    loadExplanation();

    // Pareto table
    const front = Array.isArray(result.pareto_front) ? result.pareto_front : [];
    document.getElementById("paretoHint").textContent =
        front.length + " pipelines are Pareto-optimal: none of them is beaten by another pipeline on all five goals. " +
        "Only these are shown. Pipelines that are beaten on every goal are not sent by the server. " +
        "Points that look close in this two-score view can still differ in the other three scores. The red point is the recommended one.";
    const table = document.getElementById("paretoTable");
    const columns = SPEC_KEYS.concat(SCORE_KEYS);
    const headRow = make("tr");
    columns.forEach(function (key) {
        headRow.appendChild(make("th", LABELS[key]));
    });
    table.appendChild(make("thead")).appendChild(headRow);
    const body = make("tbody");
    front.forEach(function (row) {
        const tr = make("tr", null, sameSpec(row) ? "chosen" : "");
        columns.forEach(function (key) {
            const isScore = SCORE_KEYS.indexOf(key) !== -1;
            tr.appendChild(make("td", isScore ? number(row[key]) : String(row[key] ?? "n/a")));
        });
        body.appendChild(tr);
    });
    table.appendChild(body);

    // Pareto chart (two scores at a time)
    const xAxis = document.getElementById("xAxis");
    const yAxis = document.getElementById("yAxis");
    SCORE_KEYS.forEach(function (key) {
        xAxis.appendChild(new Option(LABELS[key], key));
        yAxis.appendChild(new Option(LABELS[key], key));
    });
    xAxis.value = "performance";
    yAxis.value = "efficiency";

    let chart = null;
    function drawChart() {
        if (typeof Chart === "undefined") {
            document.querySelector(".chart-wrap").hidden = true;
            document.querySelector(".axis-controls").hidden = true;
            document.getElementById("chartNote").hidden = false;
            return;
        }
        const xKey = xAxis.value;
        const yKey = yAxis.value;
        const others = [];
        const chosen = [];

        front.forEach(function (row) {
            if (row[xKey] === null || row[xKey] === undefined || row[yKey] === null || row[yKey] === undefined) {
                return;
            }
            const point = {
                x: Number(row[xKey]),
                y: Number(row[yKey]),
                label: SPEC_KEYS.map(function (key) { return row[key]; }).join(" / "),
                row: row
            };
            (sameSpec(row) ? chosen : others).push(point);
        });
                const allPoints = others.concat(chosen);
        function axisRange(field) {
            const values = allPoints.map(function (p) { return p[field]; });
            const low = Math.min.apply(null, values);
            const high = Math.max.apply(null, values);
            const pad = Math.max((high - low) * 0.2, 0.01);
            return { min: Math.max(0, low - pad), max: Math.min(1, high + pad) };
        }
        const xRange = allPoints.length ? axisRange("x") : { min: 0, max: 1 };
        const yRange = allPoints.length ? axisRange("y") : { min: 0, max: 1 };
        
        if (chart) {
            chart.destroy();
        }
        chart = new Chart(document.getElementById("paretoChart"), {
            type: "scatter",
            data: {
                datasets: [
                    { label: "Other Pareto-optimal pipelines", data: others, backgroundColor: "#818cf8", pointRadius: 6 },
                    { label: "Recommended", data: chosen, backgroundColor: "#dc2626", pointRadius: 9 }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { title: { display: true, text: LABELS[xKey] }, min: xRange.min, max: xRange.max },
                    y: { title: { display: true, text: LABELS[yKey] }, min: yRange.min, max: yRange.max }
                },
                plugins: {
                    tooltip: {
                        callbacks: {
                            label: function (context) {
                                const row = context.raw.row;
                                return [context.raw.label].concat(SCORE_KEYS.map(function (key) {
                                    return LABELS[key] + ": " + number(row[key]);
                                }));
                            }
                        }
                    }
                }
            }
        });
    }
    xAxis.addEventListener("change", drawChart);
    yAxis.addEventListener("change", drawChart);
    drawChart();

    // Similar datasets
    const neighbours = Array.isArray(result.neighbours) ? result.neighbours : [];
    const neighbourTable = document.getElementById("neighboursTable");
    let neighbourColumns = ["dataset_id", "distance"].filter(function (key) {
        return neighbours.length > 0 && key in neighbours[0];
    });
    if (neighbourColumns.length === 0 && neighbours.length > 0) {
        neighbourColumns = Object.keys(neighbours[0]).slice(0, 3);
    }
    if (neighbours.length === 0) {
        neighbourTable.parentElement.appendChild(make("p", "No similar datasets were returned.", "hint"));
    } else {
        const neighbourHead = make("tr");
        neighbourColumns.forEach(function (key) {
            neighbourHead.appendChild(make("th", key === "dataset_id" ? "Dataset" : key === "distance" ? "Distance (smaller = more similar)" : key));
        });
        neighbourTable.appendChild(make("thead")).appendChild(neighbourHead);
        const neighbourBody = make("tbody");
        neighbours.forEach(function (row) {
            const tr = make("tr");
            neighbourColumns.forEach(function (key) {
                const value = row[key];
                tr.appendChild(make("td", typeof value === "number" ? number(value) : String(value ?? "n/a")));
            });
            neighbourBody.appendChild(tr);
        });
        neighbourTable.appendChild(neighbourBody);
    }
})();
