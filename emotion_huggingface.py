from transformers import pipeline

print("Loading local Hugging Face model...")

classifier = pipeline(
    "text-classification",
    model="SamLowe/roberta-base-go_emotions",
    top_k=None
)

ALLOWED_EMOTIONS = {
    "joy": "joy",
    "sadness": "sadness",
    "anger": "anger",
    "fear": "fear",
    "surprise": "surprise",
    "love": "love",
}

def predict_emotion(text):

    if not text or not text.strip():
        return None, 0

    results = classifier(text)[0]

    best_emotion = None
    best_score = 0

    for result in results:

        label = result["label"].lower()
        score = result["score"]

        if label in ALLOWED_EMOTIONS:
            if score > best_score:
                best_emotion = ALLOWED_EMOTIONS[label]
                best_score = score

    if best_emotion is None:
        return "joy", 0

    return best_emotion, round(best_score * 100, 2)


if __name__ == "__main__":

    while True:

        text = input("\nEnter sentence: ")

        if text.lower() == "exit":
            break

        emotion, confidence = predict_emotion(text)

        print("Emotion:", emotion)
        print("Confidence:", confidence, "%")