import unittest

from kokoro.de_text import normalize_german


class TestGermanNormalization(unittest.TestCase):
    def test_euro_amount(self):
        self.assertIn("Euro", normalize_german("Das kostet 49,99 EUR heute."))

    def test_decimal_unit(self):
        out = normalize_german("Verbrauch: 2,5 kWh pro Tag.")
        self.assertIn("Komma", out)
        self.assertIn("Kilowattstunden", out)

    def test_abbreviation(self):
        out = normalize_german("zzgl. Versand, ca. 3 Tage.")
        self.assertIn("zuzüglich", out)

    def test_max_abbrev_does_not_corrupt_maximal(self):
        out = normalize_german("max. 10 min.")
        self.assertIn("maximal", out)
        self.assertNotIn("Milliampere", out)


if __name__ == "__main__":
    unittest.main()
