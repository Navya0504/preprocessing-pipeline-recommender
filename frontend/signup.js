
const signupForm = document.getElementById("signupForm");
const signupMessage = document.getElementById("signupMessage");

signupForm.addEventListener("submit", function(event) {
    event.preventDefault();

    const name = document.getElementById("name").value.trim();
    const email = document.getElementById("email").value.trim();
    const password = document.getElementById("password").value;
    const confirmPassword = document.getElementById("confirmPassword").value;

    if (password !== confirmPassword) {
        signupMessage.style.color = "red";
        signupMessage.textContent = "Passwords do not match!";
        return;
    }

    // Frontend demo only. Not secure authentication.
    localStorage.setItem("preprocessaiUser", JSON.stringify({
        name: name,
        email: email
    }));

    signupMessage.style.color = "green";
    signupMessage.textContent = "Account created successfully!";

    setTimeout(function() {
        window.location.href = "login.html";
    }, 1000);
});
