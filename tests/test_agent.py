"""Tests for TikTok Content Agent components."""

import json
import unittest
from unittest.mock import MagicMock, patch


class TestKnowledgeBase(unittest.TestCase):
    """Test RAG knowledge base functionality."""

    def test_seed_knowledge_base(self):
        from src.rag.knowledge_base import seed_knowledge_base, KNOWLEDGE_DOCUMENTS
        count = seed_knowledge_base()
        self.assertEqual(count, len(KNOWLEDGE_DOCUMENTS))

    def test_seed_idempotent(self):
        from src.rag.knowledge_base import seed_knowledge_base
        count1 = seed_knowledge_base()
        count2 = seed_knowledge_base()
        self.assertEqual(count1, count2)

    @patch("src.rag.knowledge_base._get_collection")
    def test_query_returns_results(self, mock_get_collection):
        mock_collection = MagicMock()
        mock_collection.count.return_value = 8
        mock_collection.query.return_value = {
            "documents": [["TikTok Hook Strategies: The first 1-3 seconds..."]],
            "metadatas": [[{"category": "hooks"}]],
        }
        mock_get_collection.return_value = mock_collection

        from src.rag.knowledge_base import query_knowledge_base
        result = query_knowledge_base("how to write a good hook")
        self.assertIn("Hook", result)

    @patch("src.rag.knowledge_base._get_collection")
    def test_query_empty_collection(self, mock_get_collection):
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_collection.query.return_value = {"documents": [[]], "metadatas": [[]]}
        mock_get_collection.return_value = mock_collection

        from src.rag.knowledge_base import query_knowledge_base
        result = query_knowledge_base("anything")
        self.assertIn("No relevant", result)


class TestVideoAnalyzer(unittest.TestCase):
    """Test video analyzer tool."""

    def test_missing_video_returns_error(self):
        from src.tools.video_analyzer import analyze_video
        result = analyze_video.invoke({"video_path": "/nonexistent/video.mp4"})
        self.assertIn("Error", result)

    @patch("src.tools.video_analyzer._analyze_frames_with_vision")
    @patch("src.tools.video_analyzer._extract_frames")
    @patch("os.path.exists", return_value=True)
    def test_analyze_video_success(self, mock_exists, mock_extract, mock_analyze):
        mock_extract.return_value = ["base64frame1", "base64frame2"]
        mock_analyze.return_value = "This video shows a smart home setup with LED lights."

        from src.tools.video_analyzer import analyze_video
        result = analyze_video.invoke({"video_path": "/fake/video.mp4"})
        self.assertIn("smart home", result)
        self.assertIn("2 frames", result)


class TestHashtagResearcher(unittest.TestCase):
    """Test hashtag researcher tool."""

    @patch("src.tools.hashtag_researcher.client")
    def test_research_hashtags(self, mock_client):
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="#smarthome #tech #iot #automation #homesetup"))
        ]
        mock_client.chat.completions.create.return_value = mock_response

        from src.tools.hashtag_researcher import research_hashtags
        result = research_hashtags.invoke({
            "video_analysis": "Smart home tour with LED lights",
            "niche": "tech",
        })
        self.assertIn("#smarthome", result)


class TestDescriptionGenerator(unittest.TestCase):
    """Test description generator tool."""

    @patch("src.tools.description_generator.query_personal_posts")
    @patch("src.tools.description_generator.query_knowledge_base")
    @patch("src.tools.description_generator.client")
    def test_generate_description(self, mock_client, mock_rag, mock_personal):
        mock_rag.return_value = "Use hooks in first line. Keep under 150 chars."
        mock_personal.return_value = ""
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(
                content=(
                    "---TIKTOK---\n"
                    "Wait until you see what my lights can do\n\n#smarthome #tech\n\n"
                    "---INSTAGRAM REELS---\n"
                    "My smart home setup is next level\n\n#smarthome #tech\n\n"
                    "---YOUTUBE SHORTS---\n"
                    "Title: Smart Home Setup That Will Blow Your Mind\n"
                    "Description: Check out my automated smart home. #Shorts #smarthome"
                )
            ))
        ]
        mock_client.chat.completions.create.return_value = mock_response

        from src.tools.description_generator import generate_description
        result = generate_description.invoke({
            "video_analysis": "Smart home tour",
            "hashtags": "#smarthome #tech",
            "style": "engaging",
        })
        self.assertIn("---TIKTOK---", result)
        self.assertIn("---INSTAGRAM REELS---", result)
        self.assertIn("---YOUTUBE SHORTS---", result)


class TestEvaluator(unittest.TestCase):
    """Test evaluation/self-reflection tool."""

    @patch("src.evaluation.evaluator.client")
    def test_evaluate_returns_scores(self, mock_client):
        eval_result = {
            "scores": {
                "hook_quality": {"score": 8, "reason": "Strong opening"},
                "relevance": {"score": 9, "reason": "Matches video"},
                "engagement_potential": {"score": 7, "reason": "Good CTA"},
                "hashtag_quality": {"score": 8, "reason": "Well mixed"},
                "cta_effectiveness": {"score": 7, "reason": "Clear action"},
                "readability": {"score": 9, "reason": "Clean format"},
                "originality": {"score": 7, "reason": "Fresh angle"},
            },
            "overall_score": 7.9,
            "verdict": "PASS",
            "improvements": [],
            "improved_description": None,
        }
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content=json.dumps(eval_result)))
        ]
        mock_client.chat.completions.create.return_value = mock_response

        from src.evaluation.evaluator import evaluate_description
        result = evaluate_description.invoke({
            "description": "Amazing smart home setup 🏠\n#smarthome #tech",
            "video_analysis": "Smart home tour with LED lights",
        })
        data = json.loads(result)
        self.assertEqual(data["verdict"], "PASS")
        self.assertGreaterEqual(data["overall_score"], 7)

    @patch("src.evaluation.evaluator.client")
    def test_evaluate_needs_improvement(self, mock_client):
        eval_result = {
            "scores": {
                "hook_quality": {"score": 4, "reason": "Weak"},
                "relevance": {"score": 5, "reason": "Ok"},
                "engagement_potential": {"score": 3, "reason": "Low"},
                "hashtag_quality": {"score": 5, "reason": "Ok"},
                "cta_effectiveness": {"score": 4, "reason": "Weak"},
                "readability": {"score": 6, "reason": "Ok"},
                "originality": {"score": 3, "reason": "Generic"},
            },
            "overall_score": 4.3,
            "verdict": "NEEDS_IMPROVEMENT",
            "improvements": ["Add a stronger hook", "Include a CTA"],
            "improved_description": "Better version here",
        }
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content=json.dumps(eval_result)))
        ]
        mock_client.chat.completions.create.return_value = mock_response

        from src.evaluation.evaluator import evaluate_description
        result = evaluate_description.invoke({
            "description": "Video.",
            "video_analysis": "Smart home tour",
        })
        data = json.loads(result)
        self.assertEqual(data["verdict"], "NEEDS_IMPROVEMENT")
        self.assertIsNotNone(data["improved_description"])


if __name__ == "__main__":
    unittest.main()
