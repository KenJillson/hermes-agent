"""The checkpoint redactor's fast pre-filter must never skip text the canonical
sanitizer would change or halt on (build_graph_sanitize._might_match docstring).

This contract was stated as "asserted by test" but no test existed before CD-102.
It runs against the REAL lib/model_call sanitizer (HERMES_CONFIG_ROOT), so a rule
added there without a filter token fails here.
"""
import unittest

from hermes_cli import build_graph_sanitize as bgs

CORPUS = [
    # rules 1-7 (sanitizer v2)
    'api_key = "abcdef123456"', "password: hunter2secret", "export TOKEN=abcdef123456",
    "Authorization: Bearer abcdefghijklmnop", "AKIAABCDEFGHIJKLMNOP",
    "postgres://u:pa55word@db/app", "see /home/data/x", "/home/overlord/genomics-platform/a",
    "-----BEGIN OPENSSH PRIVATE KEY-----", "a peer_representation here", "working_representation",
    "honcho_context",
    # rule 8 (F-S1), present only when the canonical sanitizer has it
    'GH = "ghp_' + "A" * 36 + '"', "github_pat_" + "a" * 30, "xoxb-1111-2222-abcdefghijkl",
    "sk-ant-api03-" + "Z" * 30, "glpat-" + "q" * 20, "AIza" + "S" * 35,
    "sk_live_" + "a" * 24, "rk_test_" + "b" * 24,
    "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U",
]


class FastFilterContract(unittest.TestCase):
    def test_filter_never_skips_what_sanitize_would_touch(self):
        mod = bgs._load_sanitizer()
        for text in CORPUS:
            with self.subTest(text=text[:30]):
                try:
                    touched = mod.sanitize(text) != text
                except mod.SanitizerHalt:
                    touched = True
                if touched:
                    self.assertTrue(bgs._might_match(text), "filter would skip: %r" % text[:40])

    def test_rule8_is_active_in_the_canonical_sanitizer(self):
        # Guards the test above against passing vacuously on a pre-F-S1 sanitizer.
        mod = bgs._load_sanitizer()
        self.assertTrue(hasattr(mod, "FAST_FILTER_TOKENS"), "F-S1 sanitizer not installed")
        self.assertNotEqual(mod.sanitize("xoxb-1111-2222-abcdefghijkl"), "xoxb-1111-2222-abcdefghijkl")


if __name__ == "__main__":
    unittest.main()
