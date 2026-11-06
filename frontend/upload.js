if (localStorage.getItem("preprocessaiLoggedIn") !== "true") {
    window.location.href = "login.html";
}
if (localStorage.getItem("preprocessaiLoggedIn") !== "true") {
    window.location.href = "login.html";
}
const fileInput = document.getElementById("datasetFile");
const fileName = document.getElementById("fileName");
const analyzeButton = document.getElementById("analyzeButton");
const statusMessage = document.getElementById("statusMessage");

fileInput.addEventListener("change", function () {
    const file = fileInput.files[0];

    statusMessage.textContent = "";

    if (!file) {
        fileName.textContent = "No file selected";
        return;
    }

    if (!file.name.toLowerCase().endsWith(".csv")) {
        fileName.textContent = "No valid CSV selected";
        statusMessage.textContent = "Please select a CSV file.";
        fileInput.value = "";
        return;
    }

    fileName.textContent = "Selected: " + file.name;
});

analyzeButton.addEventListener("click", function () {
    const file = fileInput.files[0];

    if (!file) {
        statusMessage.textContent = "Please select a CSV file first.";
        return;
    }

    statusMessage.textContent =
        "File selected! Backend integration is the next step.";
});