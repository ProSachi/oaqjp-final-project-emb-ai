"""Flask web application for emotion detection."""

from html import escape

from flask import Flask, render_template, request

from EmotionDetection.emotion_detection import emotion_detector

app = Flask(__name__)
INVALID_TEXT_MESSAGE = "<strong>¡Texto inválido! ¡Por favor, inténtalo de nuevo!</strong>"


@app.route("/")
def home():
    """Render the emotion detection home page."""
    return render_template("index.html")


@app.route("/emotionDetector")
def emotion_detector_route():
    """Analyze the submitted text and return its emotion scores."""
    text_to_analyze = request.args.get("textToAnalyze", "")
    if not text_to_analyze.strip():
        return INVALID_TEXT_MESSAGE

    emotions = emotion_detector(text_to_analyze)
    if emotions["dominant_emotion"] is None:
        return INVALID_TEXT_MESSAGE

    dominant_emotion = escape(emotions["dominant_emotion"])
    return (
        "For the given statement, the system response is "
        f"'anger': {emotions['anger']}, "
        f"'disgust': {emotions['disgust']}, "
        f"'fear': {emotions['fear']}, "
        f"'joy': {emotions['joy']} and "
        f"'sadness': {emotions['sadness']}. "
        f"The dominant emotion is <strong>{dominant_emotion}</strong>."
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
