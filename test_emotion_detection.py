import unittest
from unittest.mock import Mock, patch

from EmotionDetection.emotion_detection import emotion_detector


class EmotionDetectionTests(unittest.TestCase):
    def assert_dominant_emotion(self, text, expected_emotion):
        emotion_scores = {
            "anger": 0.025,
            "disgust": 0.025,
            "fear": 0.025,
            "joy": 0.025,
            "sadness": 0.025,
        }
        emotion_scores[expected_emotion] = 0.9
        response = Mock()
        response.json.return_value = {
            "emotionPredictions": [{"emotion": emotion_scores}]
        }

        with patch(
            "EmotionDetection.emotion_detection.requests.post",
            return_value=response,
        ) as post:
            result = emotion_detector(text)

        self.assertEqual(result["dominant_emotion"], expected_emotion)
        post.assert_called_once()
        self.assertEqual(
            post.call_args.kwargs["json"], {"raw_document": {"text": text}}
        )

    def test_joy(self):
        self.assert_dominant_emotion("Me alegra que esto haya sucedido", "joy")

    def test_anger(self):
        self.assert_dominant_emotion("Estoy realmente enojado por esto", "anger")

    def test_disgust(self):
        self.assert_dominant_emotion(
            "Me siento disgustado solo de oír hablar de esto", "disgust"
        )

    def test_sadness(self):
        self.assert_dominant_emotion("Estoy tan triste por esto", "sadness")

    def test_fear(self):
        self.assert_dominant_emotion("Tengo mucho miedo de que esto suceda", "fear")

    def test_empty_input_returns_none_for_all_emotions(self):
        response = Mock(status_code=400)
        expected = {
            "anger": None,
            "disgust": None,
            "fear": None,
            "joy": None,
            "sadness": None,
            "dominant_emotion": None,
        }

        with patch(
            "EmotionDetection.emotion_detection.requests.post",
            return_value=response,
        ) as post:
            result = emotion_detector("")

        self.assertEqual(result, expected)
        response.raise_for_status.assert_not_called()
        response.json.assert_not_called()
        self.assertEqual(post.call_args.kwargs["json"], {"raw_document": {"text": ""}})


if __name__ == "__main__":
    unittest.main()
