   // When the page is served by the API (one address), call the same address.
   // When it is opened from the separate test server on port 5500, call the API on port 8000.
   const API_URL = window.location.port === "5500" ? "http://127.0.0.1:8000" : "";