import { describe, test, expect } from "vitest";
import { phonemize, normalize_text } from "../src/phonemize.js";

const A_TEST_CASES = new Map([
  ["‘Hello’", "həlˈoʊ"],
  ["‘Test’ and ‘Example’", "tˈɛst ænd ɛɡzˈæmpəl"],
  ["«Bonjour»", '"bɔːnʒˈʊɹ"'],
  ["«Test «nested» quotes»", '"tˈɛst "nˈɛstᵻd" kwˈoʊts"'],
  ["(Hello)", "«həlˈoʊ»"],
  ["(Nested (Parentheses))", "«nˈɛstᵻd «pɚɹˈɛnθəsˌiːz»»"],
  ["こんにちは、世界！", "dʒˈæpəniːzlˌɛɾɚ dʒˈæpəniːzlˌɛɾɚ dʒˈæpəniːzlˌɛɾɚ dʒˈæpəniːzlˌɛɾɚ dʒˈæpəniːzlˌɛɾɚ, tʃˈaɪniːzlˌɛɾɚ tʃˈaɪniːzlˌɛɾɚ!"],
  ["これはテストです：はい？", "dʒˈæpəniːzlˌɛɾɚ dʒˈæpəniːzlˌɛɾɚ dʒˈæpəniːzlˌɛɾɚ dʒˈæpəniːzlˌɛɾɚ dʒˈæpəniːzlˌɛɾɚ dʒˈæpəniːzlˌɛɾɚ dʒˈæpəniːzlˌɛɾɚ dʒˈæpəniːzlˌɛɾɚ: dʒˈæpəniːzlˌɛɾɚ dʒˈæpəniːzlˌɛɾɚ?"],
  ["Hello World", "həlˈoʊ wˈɜːld"],
  ["Hello   World", "həlˈoʊ wˈɜːld"],
  ["Hello\n   \nWorld", "həlˈoʊ wˈɜːld"],
  ["Dr. Smith", "dˈɑːktɚ smˈɪθ"],
  ["DR. Brown", "dˈɑːktɚ bɹˈaʊn"],
  ["Mr. Smith", "mˈɪstɚ smˈɪθ"],
  ["MR. Anderson", "mˈɪstɚɹ ˈændɚsən"],
  ["Ms. Taylor", "mˈɪs tˈeɪlɚ"],
  ["MS. Carter", "mˈɪs kˈɑːɹɾɚ"],
  ["Mrs. Johnson", "mˈɪsɪz dʒˈɑːnsən"],
  ["MRS. Wilson", "mˈɪsɪz wˈɪlsən"],
  ["Apples, oranges, etc.", "ˈæpəlz, ˈɔɹɪndʒᵻz, ɛtsˈɛtɹə"],
  ["Apples, etc. Pears.", "ˈæpəlz, ɛtsˈɛtɹə. pˈɛɹz."],
  ["Yeah", "jˈɛə"],
  ["yeah", "jˈɛə"],
  ["1990", "nˈaɪntiːn nˈaɪndi"],
  ["12:34", "twˈɛlv θˈɜːɾi fˈoːɹ"],
  ["2022s", "twˈɛnti twˈɛnti tˈuːz"],
  ["1,000", "wˈʌn θˈaʊzənd"],
  ["12,345,678", "twˈɛlv mˈɪliən θɹˈiː hˈʌndɹɪd fˈoːɹɾi fˈaɪv θˈaʊzənd sˈɪks hˈʌndɹɪd sˈɛvənti ˈeɪt"],
  ["$100", "wˈʌn hˈʌndɹɪd dˈɑːlɚz"],
  ["£1.50", "wˈʌn pˈaʊnd ænd fˈɪfti pˈɛns"],
  ["12.34", "twˈɛlv pˈɔɪnt θɹˈiː fˈoːɹ"],
  ["0.01", "zˈiəɹoʊ pˈɔɪnt zˈiəɹoʊ wˈʌn"],
  ["10-20", "tˈɛn tə twˈɛnti"],
  ["5-10", "fˈaɪv tə tˈɛn"],
  ["10S", "tˈɛn ˈɛs"],
  ["5S", "fˈaɪv ˈɛs"],
  ["Cat's tail", "kˈæts tˈeɪl"],
  ["X's mark", "ˈɛksᵻz mˈɑːɹk"],
  ["U.S.A.", "jˈuːˈɛsˈeɪ."],
  ["A.B.C", "ˈeɪbˈiːsˈiː"],
]);

const B_TEST_CASES = new Map([
  ["‘Hello’", "həlˈəʊ"],
  ["‘Test’ and ‘Example’", "tˈɛst and ɛɡzˈampəl"],
  ["«Bonjour»", '"bɔːnʒˈʊə"'],
  ["«Test «nested» quotes»", '"tˈɛst "nˈɛstɪd" kwˈəʊts"'],
  ["(Hello)", "«həlˈəʊ»"],
  ["(Nested (Parentheses))", "«nˈɛstɪd «pəɹˈɛnθəsˌiːz»»"],
  ["こんにちは、世界！", "dʒˈapəniːzlˌɛtə dʒˈapəniːzlˌɛtə dʒˈapəniːzlˌɛtə dʒˈapəniːzlˌɛtə dʒˈapəniːzlˌɛtə, tʃˈaɪniːzlˌɛtə tʃˈaɪniːzlˌɛtə!"],
  ["これはテストです：はい？", "dʒˈapəniːzlˌɛtə dʒˈapəniːzlˌɛtə dʒˈapəniːzlˌɛtə dʒˈapəniːzlˌɛtə dʒˈapəniːzlˌɛtə dʒˈapəniːzlˌɛtə dʒˈapəniːzlˌɛtə dʒˈapəniːzlˌɛtə: dʒˈapəniːzlˌɛtə dʒˈapəniːzlˌɛtə?"],
  ["Hello World", "həlˈəʊ wˈɜːld"],
  ["Hello   World", "həlˈəʊ wˈɜːld"],
  ["Hello\n   \nWorld", "həlˈəʊ wˈɜːld"],
  ["Dr. Smith", "dˈɒktə smˈɪθ"],
  ["DR. Brown", "dˈɒktə bɹˈaʊn"],
  ["Mr. Smith", "mˈɪstə smˈɪθ"],
  ["MR. Anderson", "mˈɪstəɹ ˈandəsən"],
  ["Ms. Taylor", "mˈɪs tˈeɪlə"],
  ["MS. Carter", "mˈɪs kˈɑːtə"],
  ["Mrs. Johnson", "mˈɪsɪz dʒˈɒnsən"],
  ["Apples, oranges, etc.", "ˈapəlz, ˈɒɹɪndʒɪz, ɛtsˈɛtɹə"],
  ["Apples, etc. Pears.", "ˈapəlz, ɛtsˈɛtɹə. pˈeəz."],
  ["1990", "nˈaɪntiːn nˈaɪnti"],
  ["12:34", "twˈɛlv θˈɜːti fˈɔː"],
  ["1,000", "wˈɒn θˈaʊzənd"],
  ["12,345,678", "twˈɛlv mˈɪliən θɹˈiː hˈʌndɹɪdən fˈɔːti fˈaɪv θˈaʊzənd sˈɪks hˈʌndɹɪdən sˈɛvənti ˈeɪt"],
  ["$100", "wˈɒn hˈʌndɹɪd dˈɒləz"],
  ["£1.50", "wˈɒn pˈaʊnd and fˈɪfti pˈɛns"],
  ["12.34", "twˈɛlv pˈɔɪnt θɹˈiː fˈɔː"],
  ["0.01", "zˈiəɹəʊ pˈɔɪnt zˈiəɹəʊ wˈɒn"],
  ["Cat's tail", "kˈats tˈeɪl"],
  ["X's mark", "ˈɛksɪz mˈɑːk"],
]);

describe("normalize_text", () => {
  // Quotes and brackets
  test("smart single quotes → straight", () => expect(normalize_text("\u2018Hello\u2019")).toBe("'Hello'"));
  test("double curly quotes → straight", () => expect(normalize_text("\u201cHello\u201d")).toBe('"Hello"'));
  test("parentheses → angle-quote brackets", () => expect(normalize_text("(Hello)")).toBe("«Hello»"));

  // Whitespace
  test("multiple spaces collapsed", () => expect(normalize_text("Hello   World")).toBe("Hello World"));
  test("tabs → single space", () => expect(normalize_text("Hello\tWorld")).toBe("Hello World"));

  // Abbreviations
  test("Dr. before capital → Doctor", () => expect(normalize_text("Dr. Smith")).toBe("Doctor Smith"));
  test("Mr. → Mister", () => expect(normalize_text("Mr. Smith")).toBe("Mister Smith"));
  test("Ms. → Miss", () => expect(normalize_text("Ms. Taylor")).toBe("Miss Taylor"));
  test("Mrs. → Mrs", () => expect(normalize_text("Mrs. Johnson")).toBe("Mrs Johnson"));
  test("etc. mid-sentence → etc", () => expect(normalize_text("apples, etc. Pears")).toBe("apples, etc. Pears"));
  test("etc. end → etc", () => expect(normalize_text("apples, etc.")).toBe("apples, etc"));

  // Numbers
  test("year 1990 → split", () => expect(normalize_text("1990")).toBe("19 90"));
  test("time 12:34 → split", () => expect(normalize_text("12:34")).toBe("12 34"));
  test("thousands separator removed", () => expect(normalize_text("1,000")).toBe("1000"));
  test("decimal → point form", () => expect(normalize_text("12.34")).toBe("12 point 3 4"));
  test("number range → 'to'", () => expect(normalize_text("10-20")).toBe("10 to 20"));

  // Currency
  test("$100 → dollar form", () => expect(normalize_text("$100")).toBe("100 dollars"));
  test("£1.50 → pound form", () => expect(normalize_text("£1.50")).toBe("1 pound and 50 pence"));

  // Possessives — the rule uppercases 's' only when preceded by an uppercase consonant
  test("uppercase consonant possessive → uppercase S", () => expect(normalize_text("CAT's tail")).toBe("CAT'S tail"));
  test("lowercase consonant possessive unchanged", () => expect(normalize_text("Cat's tail")).toBe("Cat's tail"));

  // Hyphenated initials
  test("A.B.C → A-B-C", () => expect(normalize_text("A.B.C")).toBe("A-B-C"));
  test("U.S.A. keeps trailing dot", () => expect(normalize_text("U.S.A.")).toBe("U-S-A."));
});

describe("phonemize", () => {
  describe("en-us", () => {
    for (const [input, expected] of A_TEST_CASES) {
      test(`phonemize("${input}")`, async () => {
        expect(await phonemize(input)).toEqual(expected);
      });
    }
  });
  describe("en-gb", () => {
    for (const [input, expected] of B_TEST_CASES) {
      test(`phonemize("${input}")`, async () => {
        expect(await phonemize(input, "b")).toEqual(expected);
      });
    }
  });
});
