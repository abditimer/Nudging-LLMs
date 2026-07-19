import tempfile
import unittest
from pathlib import Path

from nudging.data_loader import load_data, preprocess_text


class TestDataLoader(unittest.TestCase):
    def test_preprocess_text_podcasts_removes_metadata(self):
        cleaned = preprocess_text(
            "podcasts",
            "0:05\nHOST NAME: Hello [MUSIC PLAYING] 01:12 world",
        )
        self.assertEqual(cleaned, "Hello world")

    def test_preprocess_text_songs_preserves_words(self):
        self.assertEqual(preprocess_text("songs", "one\n\n\ntwo"), "one\n\ntwo")

    def test_load_data_filters_and_uses_structured_ids(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            song_dir = root / "songs" / "artist"
            song_dir.mkdir(parents=True)
            (song_dir / "kept.txt").write_text("one two three four", encoding="utf-8")
            (song_dir / "short.txt").write_text("one two", encoding="utf-8")

            contents = load_data(root, min_words=3, categories=["songs"])

        self.assertEqual(contents, {"songs::artist::kept": "one two three four"})

    def test_load_data_filters_categories(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for category in ("songs", "podcasts"):
                directory = root / category / "owner"
                directory.mkdir(parents=True)
                (directory / "text.txt").write_text("one two three", encoding="utf-8")

            contents = load_data(root, min_words=3, categories=["songs"])

        self.assertEqual(list(contents), ["songs::owner::text"])
