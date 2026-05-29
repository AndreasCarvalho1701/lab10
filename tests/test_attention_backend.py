import unittest

from lab10_attention import can_use_flash_attention_2, choose_attention_implementation


class AttentionBackendTests(unittest.TestCase):
    def test_flash_requires_cuda_ampere_and_package(self):
        self.assertTrue(can_use_flash_attention_2(True, (8, 0), True))
        self.assertFalse(can_use_flash_attention_2(False, (8, 0), True))
        self.assertFalse(can_use_flash_attention_2(True, (7, 5), True))
        self.assertFalse(can_use_flash_attention_2(True, (8, 0), False))

    def test_t4_falls_back_to_sdpa(self):
        choice = choose_attention_implementation(
            prefer_flash=True,
            cuda_available=True,
            capability=(7, 5),
            flash_attn_installed=True,
        )
        self.assertEqual(choice.implementation, "sdpa")
        self.assertIn("Ampere", choice.reason)

    def test_ampere_with_flash_uses_flash_attention_2(self):
        choice = choose_attention_implementation(
            prefer_flash=True,
            cuda_available=True,
            capability=(8, 6),
            flash_attn_installed=True,
        )
        self.assertEqual(choice.implementation, "flash_attention_2")
        self.assertTrue(choice.flash_available)

    def test_disable_flash_uses_sdpa(self):
        choice = choose_attention_implementation(
            prefer_flash=False,
            cuda_available=True,
            capability=(8, 6),
            flash_attn_installed=True,
        )
        self.assertEqual(choice.implementation, "sdpa")


if __name__ == "__main__":
    unittest.main()
