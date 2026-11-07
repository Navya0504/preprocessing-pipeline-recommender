(function () {
    let result = null;
    try {
        result = JSON.parse(sessionStorage.getItem("preprocessaiResult"));
    } catch (ignored) {
        result = null;
    }
    if (!result || !result.recommendation_id) {
        return;
    }

    const codeButton = document.getElementById("downloadCode");
    const cleanButton = document.getElementById("downloadCleaned");
    const cleanFile = document.getElementById("cleanFile");
    const status = document.getElementById("downloadStatus");

    function setStatus(text, isError) {
        status.textContent = text;
        status.classList.toggle("error", Boolean(isError));
    }

    function saveBlob(blob, filename) {
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        link.remove();
        URL.revokeObjectURL(url);
    }

    async function readError(response) {
        if (response.status === 404) {
            return "This recommendation is no longer on the server (it may have restarted). Please upload the dataset again.";
        }
        try {
            const body = await response.json();
            if (typeof body.detail === "string") {
                return body.detail;
            }
        } catch (ignored) {
            // The body was not JSON.
        }
        return "The server returned an error (" + response.status + ").";
    }

    codeButton.addEventListener("click", async function () {
        codeButton.disabled = true;
        setStatus("Preparing the code...");
        try {
            const response = await fetch(API_URL + "/export/" + encodeURIComponent(result.recommendation_id));
            if (!response.ok) {
                throw new Error(await readError(response));
            }
            saveBlob(await response.blob(), "preprocessing_pipeline.py");
            setStatus("Downloaded preprocessing_pipeline.py");
        } catch (error) {
            setStatus(error instanceof TypeError ? "Cannot reach the API at " + API_URL + ". Is it running?" : error.message, true);
        }
        codeButton.disabled = false;
    });

    cleanButton.addEventListener("click", async function () {
        const file = cleanFile.files[0];
        if (!file) {
            setStatus("Please choose the same CSV file again first.", true);
            return;
        }

        const form = new FormData();
        form.append("recommendation_id", result.recommendation_id);
        form.append("file", file);

        cleanButton.disabled = true;
        setStatus("Cleaning the dataset...");
        try {
            const response = await fetch(API_URL + "/clean", { method: "POST", body: form });
            if (!response.ok) {
                throw new Error(await readError(response));
            }
            saveBlob(await response.blob(), "cleaned_dataset.csv");
            setStatus("Downloaded cleaned_dataset.csv");
        } catch (error) {
            setStatus(error instanceof TypeError ? "Cannot reach the API at " + API_URL + ". Is it running?" : error.message, true);
        }
        cleanButton.disabled = false;
    });
})();
