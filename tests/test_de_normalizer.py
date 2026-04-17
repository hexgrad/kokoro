"""
Tests for German text normalisation (de_normalizer.py).

No external deps — runs on any CI without espeak-ng.
Run with:  python -m pytest tests/test_de_normalizer.py -v
"""
import pytest
from kokoro.de_normalizer import normalize_text_de, int_to_de, ordinal_stem_de, year_de


# ─── int_to_de ────────────────────────────────────────────────────────────────

class TestIntToDE:
    def test_zero(self):           assert int_to_de(0)   == "null"
    def test_one(self):            assert int_to_de(1)   == "ein"
    def test_eleven(self):         assert int_to_de(11)  == "elf"
    def test_twelve(self):         assert int_to_de(12)  == "zwölf"
    def test_thirteen(self):       assert int_to_de(13)  == "dreizehn"
    def test_nineteen(self):       assert int_to_de(19)  == "neunzehn"
    def test_twenty(self):         assert int_to_de(20)  == "zwanzig"
    def test_thirty(self):         assert int_to_de(30)  == "dreißig"
    def test_twenty_one(self):     assert int_to_de(21)  == "einundzwanzig"
    def test_forty_two(self):      assert int_to_de(42)  == "zweiundvierzig"
    def test_ninety_nine(self):    assert int_to_de(99)  == "neunundneunzig"
    def test_hundred(self):        assert int_to_de(100) == "einhundert"
    def test_two_hundred(self):    assert int_to_de(200) == "zweihundert"
    def test_hundred_one(self):    assert int_to_de(101) == "einhundertein"
    def test_nine_ninety_nine(self): assert int_to_de(999) == "neunhundertneunundneunzig"
    def test_thousand(self):       assert int_to_de(1000) == "eintausend"
    def test_two_thousand(self):   assert int_to_de(2000) == "zweitausend"
    def test_million(self):        assert int_to_de(1_000_000) == "eine Million"
    def test_two_million(self):    assert int_to_de(2_000_000) == "zwei Millionen"
    def test_negative(self):       assert int_to_de(-5) == "minus fünf"
    def test_large(self):
        r = int_to_de(12_345)
        assert "zwölf" in r
        assert "tausend" in r


class TestOrdinalStemDE:
    def test_1(self):  assert ordinal_stem_de(1)  == "erst"
    def test_2(self):  assert ordinal_stem_de(2)  == "zweit"
    def test_3(self):  assert ordinal_stem_de(3)  == "dritt"
    def test_7(self):  assert ordinal_stem_de(7)  == "siebt"
    def test_8(self):  assert ordinal_stem_de(8)  == "acht"
    def test_4(self):  assert ordinal_stem_de(4)  == "viert"
    def test_20(self): assert ordinal_stem_de(20) == "zwanzigst"
    def test_21(self): assert "einundzwanzig" in ordinal_stem_de(21)


class TestYearDE:
    def test_1985(self): assert year_de(1985) == "neunzehnhundertfünfundachtzig"
    def test_1900(self): assert year_de(1900) == "neunzehnhundert"
    def test_1100(self): assert year_de(1100) == "elfhundert"
    def test_2000(self): assert "zweitausend" in year_de(2000)
    def test_2024(self): assert year_de(2024) == "zweitausendvierundzwanzig"
    def test_800(self):  assert "acht" in year_de(800)


# ─── normalize_text_de ────────────────────────────────────────────────────────

class TestQuotes:
    def test_german_low_high(self):
        r = normalize_text_de('Er sagte: „Guten Morgen."')
        assert '„' not in r and '\u201c' not in r

    def test_angle_quotes(self):
        r = normalize_text_de("Das ist «toll».")
        assert '«' not in r and '»' not in r

    def test_curly_single(self):
        r = normalize_text_de("It\u2019s fine")
        assert '\u2019' not in r


class TestAbbreviations:
    def test_dr(self):     assert "Doktor"    in normalize_text_de("Dr. Müller")
    def test_prof(self):   assert "Professor" in normalize_text_de("Prof. Schmidt hält")
    def test_str_suffix(self): assert "Straße" in normalize_text_de("In der Hauptstr. links")
    def test_str_standalone(self): assert "Straße" in normalize_text_de("Str. des Friedens")
    def test_nr(self):     assert "Nummer"    in normalize_text_de("Nr. 5 bitte")
    def test_zb(self):     assert "zum Beispiel"  in normalize_text_de("z.B. morgen")
    def test_dh(self):     assert "das heißt"     in normalize_text_de("d.h. später")
    def test_usw(self):    assert "und so weiter" in normalize_text_de("Äpfel usw.")
    def test_bzw(self):    assert "beziehungsweise" in normalize_text_de("der Hund bzw. die Katze")
    def test_etc(self):    assert "et cetera"      in normalize_text_de("und so etc.")
    def test_ca(self):     assert "circa"          in normalize_text_de("ca. 10 Minuten")
    def test_gmbh(self):   assert "Gesellschaft" in normalize_text_de("Muster GmbH")
    def test_jan(self):    assert "Januar"    in normalize_text_de("Jan. 2024 ")
    def test_dez(self):    assert "Dezember"  in normalize_text_de("Dez. 2024 ")
    def test_okt(self):    assert "Oktober"   in normalize_text_de("Okt. 2023 ")


class TestNumbers:
    def test_zero(self):   assert "null"            in normalize_text_de("Er hat 0 Punkte.")
    def test_single(self): assert "fünf"            in normalize_text_de("5 Katzen.")
    def test_double(self): assert "zweiundvierzig"  in normalize_text_de("42 Leute.")
    def test_no_digit(self): assert "42" not in normalize_text_de("42 Leute.")
    def test_triple(self): assert "einhundert"      in normalize_text_de("100 Punkte.")
    def test_thousands_dot(self): assert "tausend"  in normalize_text_de("1.000 Menschen.")
    def test_million_format(self):
        r = normalize_text_de("1.234.567 Einwohner.")
        assert "Million" in r or "tausend" in r
        assert "1.234.567" not in r
    def test_decimal_comma(self):
        r = normalize_text_de("36,9 Grad.")
        assert "Komma" in r
        assert "36,9" not in r


class TestCurrency:
    def test_euro_before(self):
        r = normalize_text_de("kostet €10")
        assert "Euro" in r and "€" not in r and "zehn" in r

    def test_euro_after(self):
        r = normalize_text_de("kostet 10€")
        assert "Euro" in r and "€" not in r

    def test_euro_cents(self):
        r = normalize_text_de("€9,99 bitte")
        assert "Euro" in r and "Cent" in r

    def test_dollar(self):
        r = normalize_text_de("$100 Rabatt")
        assert "Dollar" in r and "$" not in r

    def test_pound(self):
        r = normalize_text_de("£50 worth")
        assert "Pfund" in r


class TestTimes:
    def test_full_hour(self):
        r = normalize_text_de("Um 14:00 Uhr.")
        assert "vierzehn Uhr" in r and "14:00" not in r

    def test_with_minutes(self):
        r = normalize_text_de("Um 8:30 Uhr.")
        assert "acht Uhr dreißig" in r

    def test_midnight(self):
        r = normalize_text_de("Um 0:00 Uhr.")
        assert "null Uhr" in r

    def test_midday(self):
        r = normalize_text_de("Um 12:00 Uhr.")
        assert "zwölf Uhr" in r

    def test_no_trailing_zero(self):
        # 15:00 → "fünfzehn Uhr" (no trailing null)
        r = normalize_text_de("Um 15:00")
        assert "fünfzehn Uhr" in r
        assert "null" not in r


class TestDates:
    def test_christmas(self):
        r = normalize_text_de("Am 24.12.2024.")
        assert "Dezember" in r and "24.12.2024" not in r

    def test_first_jan(self):
        r = normalize_text_de("Am 1.1.2000.")
        assert "Januar" in r

    def test_ordinal_third(self):
        r = normalize_text_de("Am 3.10.1990.")
        assert "dritt" in r and "Oktober" in r

    def test_year_in_date(self):
        r = normalize_text_de("Am 9.11.1989.")
        assert "neunzehnhundert" in r


class TestYears:
    def test_1989(self):
        r = normalize_text_de("Im Jahr 1989.")
        assert "neunzehnhundert" in r and "1989" not in r

    def test_2024(self):
        r = normalize_text_de("Im Jahr 2024.")
        assert "zweitausend" in r

    def test_1900(self):
        r = normalize_text_de("Seit 1900 hat sich viel geändert.")
        assert "neunzehnhundert" in r


class TestWhitespace:
    def test_double_space(self):  assert "  " not in normalize_text_de("Hallo   Welt")
    def test_strips(self):        assert normalize_text_de("  Hallo  ") == normalize_text_de("  Hallo  ").strip()
    def test_nbsp(self):          assert "\u00a0" not in normalize_text_de("Hallo\u00a0Welt")


class TestUmlauts:
    def test_preserved(self):
        t = "Äpfel, Österreich, Überraschung, Größe"
        r = normalize_text_de(t)
        for w in ["Äpfel", "Österreich", "Überraschung", "Größe"]:
            assert w in r

    def test_sz(self):
        assert "Straße" in normalize_text_de("Musterstr. links")


class TestEdgeCases:
    def test_empty(self):     assert normalize_text_de("") == ""
    def test_plain(self):
        r = normalize_text_de("Guten Morgen, wie geht es Ihnen?")
        assert "Guten Morgen" in r and "Ihnen" in r

    def test_complex_sentence(self):
        t = "Dr. Müller hat am 3. Mai 2023 um 14:30 Uhr 3 Pakete für €29,99 gekauft."
        r = normalize_text_de(t)
        assert "Doktor" in r
        assert "Mai" in r
        assert "vierzehn Uhr dreißig" in r
        assert "Euro" in r
        assert "€" not in r
        assert "Dr." not in r


# ─── integration (requires espeak-ng) ────────────────────────────────────────

@pytest.mark.integration
class TestDEG2PIntegration:
    """Requires espeak-ng.  Run with: pytest -m integration"""

    @pytest.fixture(autouse=True)
    def _skip_no_espeak(self):
        try:
            from kokoro.de_g2p import DEG2P
            DEG2P()
        except Exception:
            pytest.skip("espeak-ng not available")

    def test_returns_string(self):
        from kokoro.de_g2p import DEG2P
        ps, _ = DEG2P()("Hallo Welt")
        assert isinstance(ps, str) and len(ps) > 0

    def test_ipa_chars(self):
        from kokoro.de_g2p import DEG2P
        ps, _ = DEG2P()("Schöner Tag")
        assert any(c in ps for c in "aeiouɐəɛɪʊøy")

    def test_normalisation_applied(self):
        """Number-expanded text must phonemise to same phonemes as written-out text."""
        from kokoro.de_g2p import DEG2P
        g2p = DEG2P()
        ps_num, _  = g2p("Es waren 42 Leute da.")
        ps_word, _ = g2p("Es waren zweiundvierzig Leute da.")
        assert len(ps_num) > 0 and len(ps_word) > 0

    def test_pipeline_lang_d(self):
        from kokoro.pipeline import KPipeline
        p = KPipeline(lang_code='d', model=False)
        assert p.lang_code == 'd'
        results = list(p("Guten Morgen.", voice=None))
        assert len(results) >= 1
        assert results[0].graphemes
        assert results[0].phonemes

    def test_pipeline_alias_de(self):
        from kokoro.pipeline import KPipeline
        p = KPipeline(lang_code='de', model=False)
        assert p.lang_code == 'd'
