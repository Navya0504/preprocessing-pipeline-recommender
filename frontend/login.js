
const loginForm = document.getElementById("loginForm");
const password = document.getElementById("password");
const togglePassword = document.getElementById("togglePassword");
const message = document.getElementById("message");

// Show and hide password
togglePassword.addEventListener("click", function () {

    if (password.type === "password") {
        password.type = "text";
        togglePassword.textContent = "Hide";
    } else {
        password.type = "password";
        togglePassword.textContent = "Show";
    }

});

// Login form
loginForm.addEventListener("submit", function (event) {

    event.preventDefault();

    const email = document.getElementById("email").value.trim();
    const passwordValue = password.value;

    if (!email || !passwordValue) {
        message.textContent = "Please enter email and password.";
        return;
    }

    // Demo only: replace with backend authentication
    localStorage.setItem("preprocessaiLoggedIn", "true");

    message.style.color = "green";
    message.textContent = "Login successful!";

    setTimeout(function () {
        window.location.href = "upload.html";
    }, 800);

});

// Forgot password
document.getElementById("forgotPassword")
    .addEventListener("click", function (event) {
        event.preventDefault();
        message.textContent = "Password reset will be available soon.";
    });
