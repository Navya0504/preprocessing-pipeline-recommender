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
    SPEC_KEYS.forEach(function (key) {
        const card = make("div", null, "spec-item");
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
                label: SPEC_KEYS.map(function (key) { return row[key]; }).join(" / ")
            };
            (sameSpec(row) ? chosen : others).push(point);
        });

        if (chart) {
            chart.destroy();
        }
        chart = new Chart(document.getElementById("paretoChart"), {
            type: "scatter",
            data: {
                datasets: [
                    { label: "Other options", data: others, backgroundColor: "#818cf8", pointRadius: 6 },
                    { label: "Recommended", data: chosen, backgroundColor: "#dc2626", pointRadius: 9 }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { title: { display: true, text: LABELS[xKey] }, min: 0, max: 1 },
                    y: { title: { display: true, text: LABELS[yKey] }, min: 0, max: 1 }
                },
                plugins: {
                    tooltip: {
                        callbacks: {
                            label: function (context) {
                                return context.raw.label + " (" + context.raw.x.toFixed(2) + ", " + context.raw.y.toFixed(2) + ")";
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
