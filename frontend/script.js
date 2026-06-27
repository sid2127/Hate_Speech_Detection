async function predict() {

    const text = document.getElementById("textInput").value.trim();

    if (!text) {

        document.getElementById("output").innerHTML =
            "Please type a comment!";

        return;
    }

    try {

        const response = await fetch(
            "http://127.0.0.1:8000/predict",
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify({
                    text: text
                })
            }
        );

        const data = await response.json();

        let html = `
            <strong>Input:</strong> ${data.input_text}<br><br>

            <strong>Cleaned Text:</strong>
            ${data.cleaned_text}<br><br>

            <strong>Prediction:</strong>
            ${data.prediction}<br><br>

            <strong>Confidence:</strong>
            ${data.confidence}%<br><br>

            <strong>Hide Comment:</strong>
            ${data.hide_comment}
        `;

        document.getElementById("output").innerHTML = html;

    } catch (error) {

        document.getElementById("output").innerHTML =
            "Error connecting to backend!";
    }
}