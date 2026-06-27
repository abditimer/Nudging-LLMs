import unittest

from nudging.experiment import (
    _generate_response,
    _get_num_predict_for_target,
    _trim_to_n_words,
)


class FakeModelClient:
    words_to_token_multiplier = 1.5

    def __init__(self, response):
        self.response = response
        self.calls = []

    def generate(self, prompt, **options):
        self.calls.append({"prompt": prompt, "options": options})
        return self.response


class TestExperimentLengthControl(unittest.TestCase):
    def test_trim_to_n_words(self):
        self.assertEqual(_trim_to_n_words("one two three four", 2), "one two")

    def test_num_predict_uses_token_multiplier(self):
        self.assertEqual(_get_num_predict_for_target(10, 1.5), 15)
        self.assertEqual(_get_num_predict_for_target(11, 1.5), 17)

    def test_generate_response_caps_tokens_and_trims_to_target_words(self):
        client = FakeModelClient("five six seven eight nine")
        content = "one two three four"

        generated, context, target, metadata = _generate_response(
            content=content,
            percentage=50,
            model_client=client,
        )

        self.assertEqual(context, "one two")
        self.assertEqual(target, "three four")
        self.assertEqual(generated, "five six")
        self.assertEqual(client.calls[0]["options"]["num_predict"], 3)
        self.assertEqual(metadata["raw_generated_words"], 5)
        self.assertEqual(len(generated.split()), 2)
        self.assertTrue(metadata["trimmed_to_target_words"])
        self.assertTrue(metadata["length_controlled"])


if __name__ == "__main__":
    unittest.main()
