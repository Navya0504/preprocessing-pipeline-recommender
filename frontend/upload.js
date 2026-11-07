(function () {
    // Demo login check (not real security). Real sign-in comes later.
    if (localStorage.getItem("preprocessaiLoggedIn") !== "true") {
        window.location.href = "login.html";
        return;
    }

    const fileInput = document.getElementById("datasetFile");
    const fileName = document.getElementById("fileName");
    const targetSelect = document.getElementById("targetColumn");
    const taskSelect = document.getElementById("task");
    const analyzeButton = document.getElementById("analyzeButton");
    const statusMessage = document.getElementById("statusMessage");

    function setStatus(text, isError) {
        statusMessage.textContent = text;
        statusMessage.classList.toggle("error", Boolean(isError));
    }

    // Read the column names from the first line of the CSV.
    function parseHeader(text) {
        text = text.replace(/^﻿/, "");
        const columns = [];
        let current = "";
        let inQuotes = false;

        for (let i = 0; i < text.length; i++) {
            const ch = text[i];
            if (inQuotes) {
                if (ch === '"' && text[i + 1] === '"') {
                    current += '"';
                    i++;
                } else if (ch === '"') {
                    inQuotes = false;
                } else {
                    current += ch;
                }
            } else if (ch === '"') {
                inQuotes = true;
            } else if (ch === ",") {
                columns.push(current.trim());
                current = "";
            } else if (ch === "\n" || ch === "\r") {
                break;
            } else {
                current += ch;
            }
        }
        columns.push(current.trim());
        return columns.filter(function (name) { return name !== ""; });
    }

    function resetTarget(message) {
        targetSelect.innerHTML = "";
        const option = document.createElement("option");
        option.value = "";
        option.textContent = message;
        targetSelect.appendChild(option);
        targetSelect.disabled = true;
    }

    fileInput.addEventListener("change", async function () {
        const file = fileInput.files[0];
        setStatus("");

        if (!file) {
            fileName.textContent = "No file selected";
            resetTarget("Choose a CSV file first");
            return;
        }

        if (!file.name.toLowerCase().endsWith(".csv")) {
            fileName.textContent = "No valid CSV selected";
            setStatus("Please select a CSV file.", true);
            fileInput.value = "";
            resetTarget("Choose a CSV file first");
            return;
        }

        fileName.textContent = "Selected: " + file.name;

        const headerText = await file.slice(0, 65536).text();
        const columns = parseHeader(headerText);

        if (columns.length === 0) {
            resetTarget("No columns found");
            setStatus("This file has no header row.", true);
            return;
        }

        if (columns.length === 1 && /[;\t|]/.test(columns[0])) {
            setStatus("This file does not look comma-separated. Please save it as a normal CSV.", true);
        }

        targetSelect.innerHTML = "";
        columns.forEach(function (name) {
            const option = document.createElement("option");
            option.value = name;
            option.textContent = name;
            targetSelect.appendChild(option);
        });
        // The target is usually the last column.
        targetSelect.value = columns[columns.length - 1];
        targetSelect.disabled = false;
    });

    // Turn an error response from the API into a readable message.
    async function readError(response) {
        try {
            const body = await response.json();
            if (typeof body.detail === "string") {
                return body.detail;
            }
            if (Array.isArray(body.detail)) {
                return body.detail.map(function (item) { return item.msg; }).join(" ");
            }
        } catch (ignored) {
            // The body was not JSON.
        }
        return "The server returned an error (" + response.status + ").";
    }

    analyzeButton.addEventListener("click", async function () {
        const file = fileInput.files[0];

        if (!file) {
            setStatus("Please select a CSV file first.", true);
            return;
        }
        if (!targetSelect.value) {
            setStatus("Please choose the target column.", true);
            return;
        }
        if (!taskSelect.value) {
            setStatus("Please choose the task (classification or regression).", true);
            return;
        }

        const form = new FormData();
        form.append("file", file);
        form.append("target", targetSelect.value);
        form.append("task", taskSelect.value);

        analyzeButton.disabled = true;
        setStatus("Analyzing... this can take a little while for large files.");

        try {
            const response = await fetch(API_URL + "/recommend", {
                method: "POST",
                body: form
            });

            if (!response.ok) {
                throw new Error(await readError(response));
            }

            const result = await response.json();
            result.fileName = file.name;
            result.target = targetSelect.value;
            result.task = taskSelect.value;

            sessionStorage.setItem("preprocessaiResult", JSON.stringify(result));
            window.location.href = "results.html";
        } catch (error) {
            if (error instanceof TypeError) {
                setStatus("Cannot reach the API at " + API_URL + ". Is it running?", true);
            } else {
                setStatus(error.message, true);
            }
            analyzeButton.disabled = false;
        }
    });
})();
