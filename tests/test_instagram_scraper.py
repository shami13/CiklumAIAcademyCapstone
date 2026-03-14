"""Tests for Instagram profile scraper."""

import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch


def _make_mock_post(
    caption="Test caption",
    likes=100,
    comments=10,
    typename="GraphImage",
    hashtags=None,
    date=None,
):
    """Create a mock instaloader.Post object."""
    post = MagicMock()
    post.caption = caption
    post.likes = likes
    post.comments = comments
    post.typename = typename
    post.hashtags = hashtags or set()
    post.date_utc = date or datetime(2026, 3, 1, 12, 0, 0)
    return post


class TestExtractUsername(unittest.TestCase):
    """Test username extraction from various input formats."""

    def test_plain_username(self):
        from src.scrapers.instagram_scraper import extract_username
        self.assertEqual(extract_username("natgeo"), "natgeo")

    def test_username_with_at(self):
        from src.scrapers.instagram_scraper import extract_username
        self.assertEqual(extract_username("@natgeo"), "natgeo")

    def test_url_basic(self):
        from src.scrapers.instagram_scraper import extract_username
        self.assertEqual(
            extract_username("https://www.instagram.com/natgeo/"), "natgeo"
        )

    def test_url_without_trailing_slash(self):
        from src.scrapers.instagram_scraper import extract_username
        self.assertEqual(
            extract_username("https://www.instagram.com/natgeo"), "natgeo"
        )

    def test_url_with_query_params(self):
        from src.scrapers.instagram_scraper import extract_username
        self.assertEqual(
            extract_username("https://www.instagram.com/natgeo?hl=en"), "natgeo"
        )

    def test_url_http(self):
        from src.scrapers.instagram_scraper import extract_username
        self.assertEqual(
            extract_username("http://instagram.com/natgeo"), "natgeo"
        )

    def test_empty_input_raises(self):
        from src.scrapers.instagram_scraper import extract_username
        with self.assertRaises(ValueError):
            extract_username("")

    def test_whitespace_only_raises(self):
        from src.scrapers.instagram_scraper import extract_username
        with self.assertRaises(ValueError):
            extract_username("   ")

    def test_username_with_dots_and_underscores(self):
        from src.scrapers.instagram_scraper import extract_username
        self.assertEqual(extract_username("user.name_123"), "user.name_123")


class TestMapTypename(unittest.TestCase):
    """Test GraphQL typename to content type mapping."""

    def test_image(self):
        from src.scrapers.instagram_scraper import _map_typename_to_content_type
        self.assertEqual(_map_typename_to_content_type("GraphImage"), "image")

    def test_video(self):
        from src.scrapers.instagram_scraper import _map_typename_to_content_type
        self.assertEqual(_map_typename_to_content_type("GraphVideo"), "video")

    def test_sidecar(self):
        from src.scrapers.instagram_scraper import _map_typename_to_content_type
        self.assertEqual(_map_typename_to_content_type("GraphSidecar"), "carousel")

    def test_unknown(self):
        from src.scrapers.instagram_scraper import _map_typename_to_content_type
        self.assertEqual(_map_typename_to_content_type("SomethingElse"), "unknown")


class TestInstaloaderPostToDict(unittest.TestCase):
    """Test conversion of instaloader Post to standard dict format."""

    def test_basic_conversion(self):
        from src.scrapers.instagram_scraper import _instaloader_post_to_dict
        post = _make_mock_post(
            caption="Amazing photo! #travel #nature",
            likes=500,
            comments=25,
            typename="GraphImage",
            hashtags={"travel", "nature"},
            date=datetime(2026, 3, 1, 12, 0, 0),
        )
        result = _instaloader_post_to_dict(post)

        self.assertEqual(result["caption"], "Amazing photo! #travel #nature")
        self.assertEqual(result["likes"], 500)
        self.assertEqual(result["comments"], 25)
        self.assertEqual(result["shares"], 0)
        self.assertEqual(result["views"], 0)
        self.assertEqual(result["posted_at"], "2026-03-01T12:00:00")
        self.assertEqual(result["content_type"], "image")
        self.assertIn("#nature", result["hashtags"])
        self.assertIn("#travel", result["hashtags"])

    def test_none_caption(self):
        from src.scrapers.instagram_scraper import _instaloader_post_to_dict
        post = _make_mock_post(caption=None)
        result = _instaloader_post_to_dict(post)
        self.assertEqual(result["caption"], "")

    def test_empty_hashtags(self):
        from src.scrapers.instagram_scraper import _instaloader_post_to_dict
        post = _make_mock_post(hashtags=set())
        result = _instaloader_post_to_dict(post)
        self.assertEqual(result["hashtags"], [])


class TestFetchInstagramPosts(unittest.TestCase):
    """Test fetching posts from Instagram."""

    @patch("src.scrapers.instagram_scraper.instaloader")
    def test_success(self, mock_instaloader_module):
        mock_loader = MagicMock()
        mock_instaloader_module.Instaloader.return_value = mock_loader

        mock_profile = MagicMock()
        mock_profile.is_private = False
        mock_profile.get_posts.return_value = iter([
            _make_mock_post(caption="Post 1", likes=100),
            _make_mock_post(caption="Post 2", likes=200),
        ])
        mock_instaloader_module.Profile.from_username.return_value = mock_profile

        from src.scrapers.instagram_scraper import fetch_instagram_posts
        posts = fetch_instagram_posts("testuser", post_count=10)

        self.assertEqual(len(posts), 2)
        self.assertEqual(posts[0]["caption"], "Post 1")
        self.assertEqual(posts[1]["likes"], 200)

    @patch("src.scrapers.instagram_scraper.instaloader")
    def test_private_profile_raises(self, mock_instaloader_module):
        mock_loader = MagicMock()
        mock_instaloader_module.Instaloader.return_value = mock_loader

        mock_profile = MagicMock()
        mock_profile.is_private = True
        mock_instaloader_module.Profile.from_username.return_value = mock_profile

        from src.scrapers.instagram_scraper import fetch_instagram_posts
        with self.assertRaises(ValueError) as ctx:
            fetch_instagram_posts("privateuser")
        self.assertIn("private", str(ctx.exception))

    @patch("src.scrapers.instagram_scraper.instaloader")
    def test_profile_not_found_raises(self, mock_instaloader_module):
        mock_loader = MagicMock()
        mock_instaloader_module.Instaloader.return_value = mock_loader
        mock_instaloader_module.exceptions.ProfileNotExistsException = type(
            "ProfileNotExistsException", (Exception,), {}
        )
        mock_instaloader_module.Profile.from_username.side_effect = (
            mock_instaloader_module.exceptions.ProfileNotExistsException()
        )

        from src.scrapers.instagram_scraper import fetch_instagram_posts
        with self.assertRaises(ValueError) as ctx:
            fetch_instagram_posts("nonexistent")
        self.assertIn("not found", str(ctx.exception))

    @patch("src.scrapers.instagram_scraper.instaloader")
    def test_respects_post_count(self, mock_instaloader_module):
        mock_loader = MagicMock()
        mock_instaloader_module.Instaloader.return_value = mock_loader

        mock_profile = MagicMock()
        mock_profile.is_private = False
        mock_profile.get_posts.return_value = iter([
            _make_mock_post(caption=f"Post {i}") for i in range(10)
        ])
        mock_instaloader_module.Profile.from_username.return_value = mock_profile

        from src.scrapers.instagram_scraper import fetch_instagram_posts
        posts = fetch_instagram_posts("testuser", post_count=3)

        self.assertEqual(len(posts), 3)


class TestScrapeAndImport(unittest.TestCase):
    """Test the full scrape-and-import flow."""

    @patch("src.scrapers.instagram_scraper._get_posts_collection")
    @patch("src.scrapers.instagram_scraper.fetch_instagram_posts")
    def test_imports_posts(self, mock_fetch, mock_get_collection):
        mock_fetch.return_value = [
            {"caption": "Post 1", "hashtags": ["#a"], "likes": 100,
             "comments": 10, "shares": 0, "views": 0,
             "posted_at": "2026-03-01T12:00:00", "content_type": "image"},
            {"caption": "Post 2", "hashtags": ["#b"], "likes": 200,
             "comments": 20, "shares": 0, "views": 0,
             "posted_at": "2026-03-02T12:00:00", "content_type": "video"},
        ]
        mock_collection = MagicMock()
        mock_get_collection.return_value = mock_collection

        from src.scrapers.instagram_scraper import scrape_and_import_instagram
        count = scrape_and_import_instagram("testuser", post_count=5)

        self.assertEqual(count, 2)
        mock_collection.upsert.assert_called_once()
        call_args = mock_collection.upsert.call_args
        self.assertEqual(len(call_args.kwargs.get("ids", call_args[1].get("ids", []))), 2)

    @patch("src.scrapers.instagram_scraper.fetch_instagram_posts")
    def test_no_posts_returns_zero(self, mock_fetch):
        mock_fetch.return_value = []

        from src.scrapers.instagram_scraper import scrape_and_import_instagram
        count = scrape_and_import_instagram("emptyuser")

        self.assertEqual(count, 0)


if __name__ == "__main__":
    unittest.main()
