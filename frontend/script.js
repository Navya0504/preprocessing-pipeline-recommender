
const startButton = document.getElementById("startButton");
const message = document.getElementById("message");

startButton.addEventListener("click", function () {
    message.textContent =
        "Dataset analysis will be available when the backend is connected.";
});