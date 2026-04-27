from __future__ import annotations

import unittest

from invest_digital_human.cleaning import clean_content, repair_mojibake


class CleaningTest(unittest.TestCase):
    def test_repair_mojibake_keeps_clean_text_stable(self) -> None:
        original = "\u5f00\u59cb\u4e00\u4e2a\u65b0\u9636\u6bb5~"
        self.assertEqual(repair_mojibake(original), original)

    def test_clean_content_removes_known_noise_lines(self) -> None:
        raw = (
            "\u6b63\u6587\u7b2c\u4e00\u6bb5\n\n"
            "\u5fae\u4fe1\u626b\u4e00\u626b\u5173\u6ce8\u8be5\u516c\u4f17\u53f7\n\n"
            "\u6b63\u6587\u7b2c\u4e8c\u6bb5"
        )
        cleaned = clean_content(raw)
        self.assertEqual(cleaned, "\u6b63\u6587\u7b2c\u4e00\u6bb5\n\n\u6b63\u6587\u7b2c\u4e8c\u6bb5")


if __name__ == "__main__":
    unittest.main()
