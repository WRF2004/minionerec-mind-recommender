import importlib.util
from pathlib import Path
import unittest


def load_script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ScriptTests(unittest.TestCase):
    def test_collision_suffixes_are_unique_and_deterministic(self):
        resolve = load_script("generate_sid.py").resolve_collisions
        result = resolve([["<a_1>", "<b_2>"], ["<a_1>", "<b_2>"], ["<a_3>", "<b_4>"]])
        self.assertEqual(result[0], ["<a_1>", "<b_2>", "<c_0>"])
        self.assertEqual(result[1], ["<a_1>", "<b_2>", "<c_1>"])
        self.assertEqual(len({tuple(code) for code in result}), 3)

    def test_upstream_patch_is_idempotent(self):
        patch = load_script("patch_upstream.py").patch_text
        original = 'category_dict = {"Books": "books"}\n    os.environ["CUDA_VISIBLE_DEVICES"] = "0"\n'
        once = patch("evaluate.py", original)
        self.assertIn('"MIND": "news articles"', once)
        self.assertNotIn('os.environ["CUDA_VISIBLE_DEVICES"]', once)
        self.assertEqual(once, patch("evaluate.py", once))


if __name__ == "__main__":
    unittest.main()
