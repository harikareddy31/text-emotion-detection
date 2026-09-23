from flask import Flask, render_template, request
from emotion_huggingface import predict_emotion

app = Flask(__name__)

EMOTION_EMOJIS = {
    "joy": "😊",
    "sadness": "😢",
    "anger": "😡",
    "fear": "😨",
    "surprise": "😲",
    "love": "❤️"
}


def get_display_confidence(model_confidence):
    if model_confidence is None:
        return None

    if model_confidence <= 1:
        percentage = model_confidence * 100
    else:
        percentage = model_confidence

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
            emotion, model_confidence = predict_emotion(text)

            confidence = get_display_confidence(model_confidence)

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
    app.run(host="0.0.0.0", port=5000)