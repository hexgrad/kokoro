import { describe, test, expect } from "vitest";
import { phonemize } from "../src/phonemize.js";

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
  ["12:34", "twˈɛlv θˈɜːɾi fˈɔːɹ"],
  ["2022s", "twˈɛnti twˈɛnti tˈuːz"],
  ["1,000", "wˈʌn θˈaʊzənd"],
  ["12,345,678", "twˈɛlv mˈɪliən θɹˈiː hˈʌndɹɪd fˈɔːɹɾi fˈaɪv θˈaʊzənd sˈɪks hˈʌndɹɪd sˈɛvənti ˈeɪt"],
  ["$100", "wˈʌn hˈʌndɹɪd dˈɑːlɚz"],
  ["£1.50", "wˈʌn pˈaʊnd ænd fˈɪfti pˈɛns"],
  ["12.34", "twˈɛlv pˈɔɪnt θɹˈiː fˈɔːɹ"],
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

// French test cases
const F_TEST_CASES = new Map([
  ["Bonjour le monde", expect.any(String)],
]);

// Spanish test cases
const E_TEST_CASES = new Map([
  ["Hola mundo", expect.any(String)],
]);

// Japanese test cases
const J_TEST_CASES = new Map([
  ["こんにちは", expect.any(String)],
]);

// Chinese (Mandarin) test cases
const Z_TEST_CASES = new Map([
  ["你好世界", expect.any(String)],
]);

// Hindi test cases
const H_TEST_CASES = new Map([
  ["नमस्ते दुनिया", expect.any(String)],
]);

// Italian test cases
const I_TEST_CASES = new Map([
  ["Ma la volpe col suo balzo ha raggiunto il quieto Fido", expect.any(String)],
]);

// Portuguese test cases
const P_TEST_CASES = new Map([
  ["Olá mundo", expect.any(String)],
]);

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
  describe("fr", () => {
    for (const [input] of F_TEST_CASES) {
      test(`phonemize("${input}", "f")`, async () => {
        const result = await phonemize(input, "f");
        expect(result).toBeTruthy();
        expect(result.length).toBeGreaterThan(0);
      });
    }
  });
  describe("es", () => {
    for (const [input] of E_TEST_CASES) {
      test(`phonemize("${input}", "e")`, async () => {
        const result = await phonemize(input, "e");
        expect(result).toBeTruthy();
        expect(result.length).toBeGreaterThan(0);
      });
    }
  });
  describe("ja", () => {
    for (const [input] of J_TEST_CASES) {
      test(`phonemize("${input}", "j")`, async () => {
        const result = await phonemize(input, "j");
        expect(result).toBeTruthy();
        expect(result.length).toBeGreaterThan(0);
      });
    }
  });
  describe("zh", () => {
    for (const [input] of Z_TEST_CASES) {
      test(`phonemize("${input}", "z")`, async () => {
        const result = await phonemize(input, "z");
        expect(result).toBeTruthy();
        expect(result.length).toBeGreaterThan(0);
      });
    }
  });
  describe("hi", () => {
    for (const [input] of H_TEST_CASES) {
      test(`phonemize("${input}", "h")`, async () => {
        const result = await phonemize(input, "h");
        expect(result).toBeTruthy();
        expect(result.length).toBeGreaterThan(0);
      });
    }
  });
  describe("it", () => {
    for (const [input] of I_TEST_CASES) {
      test(`phonemize("${input}", "i")`, async () => {
        const result = await phonemize(input, "i");
        expect(result).toBeTruthy();
        expect(result.length).toBeGreaterThan(0);
      });
    }
  });
  describe("pt-br", () => {
    for (const [input] of P_TEST_CASES) {
      test(`phonemize("${input}", "p")`, async () => {
        const result = await phonemize(input, "p");
        expect(result).toBeTruthy();
        expect(result.length).toBeGreaterThan(0);
      });
    }
  });
});
