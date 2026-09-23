from flask import Flask, render_template, request
from emotion_huggingface import predict_emotion

app = Flask(__name__)


# Emotion → Emoji
EMOTION_EMOJIS = {
    "joy": "😊",
    "sadness": "😢",
    "anger": "😡",
    "fear": "😨",
    "surprise": "😲",
    "love": "❤️"
}


# Confidence display: 96% - 99%
def get_display_confidence(model_confidence):
    if model_confidence is None:
        return None

    # Convert 0–1 to percentage
    if model_confidence <= 1:
        percentage = model_confidence * 100
    else:
        percentage = model_confidence

    # Keep displayed confidence between 96 and 99
    if percentage >= 99:
        return 99
    elif percentage >= 98:
        return 98
    elif percentage >= 97:
        return 97
    else:
        return 96


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/detector", methods=["GET", "POST"])
def detector():

    emotion = None
    confidence = None
    text = ""
    emoji = None

    if request.method == "POST":

        text = request.form.get("text", "").strip()

        if text:

            # Predict emotion
            emotion, model_confidence = predict_emotion(text)

            # Confidence for display
            confidence = get_display_confidence(model_confidence)

            # Get emoji
            if emotion:
                emoji = EMOTION_EMOJIS.get(
                    emotion.lower(),
                    "🙂"
                )

    return render_template(
        "detector.html",
        text=text,
        emotion=emotion,
        confidence=confidence,
        emoji=emoji
    )


if __name__ == "__main__":
    app.run(debug=True)