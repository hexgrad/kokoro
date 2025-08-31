import { describe, test, expect } from "vitest";
import { TextSplitterStream, split } from "../src/splitter.js";

const TESTS = [
  {
    name: "Basic sentence splitting",
    input: "This is a test. This is another test.",
    target: ["This is a test.", "This is another test."],
  },
  {
    name: "Sentence with dash (em dash)",
    input: "This is a test — yes, it is.",
    target: ["This is a test — yes, it is."],
  },
  {
    name: "Sentences with quoted speech",
    input: 'She said, "Hello there. How are you?". I replied, "I\'m fine."',
    target: ['She said, "Hello there. How are you?".', 'I replied, "I\'m fine."'],
  },
  {
    name: "Sentences with abbreviations",
    input: "Dr. Smith is here. At 10 a.m. I saw him.",
    target: ["Dr. Smith is here.", "At 10 a.m. I saw him."],
  },
  {
    name: "Advanced sentences with abbreviations",
    input: "I went to Dr. Smith this morning at 10 a.m. and said hi.",
    target: ["I went to Dr. Smith this morning at 10 a.m. and said hi."],
  },
  {
    name: "Abbreviations with possessive",
    input: "The Dr.'s office.",
    target: ["The Dr.'s office."],
  },
  {
    name: "Ellipses in sentences",
    input: "Wait... what just happened? I don't understand...",
    target: ["Wait... what just happened?", "I don't understand..."],
  },
  {
    name: "Sentences with numbers and decimals",
    input: "The price is $4.99. Do you want to buy it?",
    target: ["The price is $4.99.", "Do you want to buy it?"],
  },
  {
    name: "Sentences starting and ending with numbers",
    input: "10 people died in 2025. 20 people died in 2026.",
    target: ["10 people died in 2025.", "20 people died in 2026."],
  },
  {
    name: "Sentences with scientific notation",
    input: "The star is 3.2×10^4 light-years away.",
    target: ["The star is 3.2×10^4 light-years away."],
  },
  {
    name: "Sentences with multiple punctuation marks",
    input: "What?! Are you serious?! This is crazy...",
    target: ["What?!", "Are you serious?!", "This is crazy..."],
  },
  {
    name: "Sentences with parentheses",
    input: "This is an example (which is quite useful). Do you agree?",
    target: ["This is an example (which is quite useful).", "Do you agree?"],
  },
  {
    name: "Nested sentences with parentheses",
    input: "This is an example (This is pretty cool. Another sentence). Do you agree?",
    target: ["This is an example (This is pretty cool. Another sentence).", "Do you agree?"],
  },
  {
    name: "Sentences with newlines",
    input: "First sentence.\nSecond sentence.\nThird sentence.",
    target: ["First sentence.", "Second sentence.", "Third sentence."],
  },
  {
    name: "Sentences with emojis",
    input: "I love pizza! 🍕 Do you? 😊",
    target: ["I love pizza!", "🍕 Do you?", "😊"],
  },
  {
    name: "Sentences with unicode and non-Latin characters",
    input: "これはテストです。 次の文です。",
    target: ["これはテストです。", "次の文です。"],
  },
  {
    name: "Sentences with bullet points",
    input: "- First point.\n- Second point.\n- Third point.",
    target: ["- First point.", "- Second point.", "- Third point."],
  },
  {
    name: "Sentences with email addresses",
    input: "My email is test@example.com. Contact me!",
    target: ["My email is test@example.com.", "Contact me!"],
  },
  {
    name: "Sentences with URLs",
    input: "Visit https://example.com. It's a great site!",
    target: ["Visit https://example.com.", "It's a great site!"],
  },
  {
    name: "Sentences with URLs (subdomains)",
    input: "Visit https://test.example.com. It's a great site!",
    target: ["Visit https://test.example.com.", "It's a great site!"],
  },
  {
    name: "Sentences with trailing spaces",
    input: "  This is a sentence.   Another one.  ",
    target: ["This is a sentence.", "Another one."],
  },
  {
    name: "Sentences with contractions",
    input: "You can't be serious. It's too late.",
    target: ["You can't be serious.", "It's too late."],
  },
  {
    name: "Sentences with title case and proper nouns",
    input: "Mr. Johnson went to New York. He loves it there.",
    target: ["Mr. Johnson went to New York.", "He loves it there."],
  },
  {
    name: "Sentences with mixed cases",
    input: "i am happy. Are you?",
    target: ["i am happy.", "Are you?"],
  },
  {
    name: "Sentences with missing punctuation",
    input: "This is a test without punctuation What should happen",
    target: ["This is a test without punctuation What should happen"],
  },
  {
    name: "Sentences with mixed symbols",
    input: "Hello @John! How's it going? #excited",
    target: ["Hello @John!", "How's it going?", "#excited"],
  },
  {
    name: "Sentences with math expressions",
    input: "The result is 3.14. It's an approximation of pi.",
    target: ["The result is 3.14.", "It's an approximation of pi."],
  },
  {
    name: "Excessive punctuation",
    input: "Wait!!!! Are you sure??? This is insane!!! Right???",
    target: ["Wait!!!!", "Are you sure???", "This is insane!!!", "Right???"],
  },
  {
    name: "Mixed languages in one line",
    input: "English sentence. 这是一句中文？ Another English sentence!",
    target: ["English sentence.", "这是一句中文？", "Another English sentence!"],
  },
  {
    name: "Sequence of punctuation plus emoji",
    input: "What??! 🤯Wait??  Hello!",
    target: ["What??!", "🤯Wait??", "Hello!"],
  },
  {
    name: "Nested parentheses and quotes",
    input: '(This is "very (strange)" text). Right?',
    target: ['(This is "very (strange)" text).', "Right?"],
  },
  {
    name: "Sentence with ellipsis following a question mark",
    input: "Are you coming? ... I don't know.",
    target: ["Are you coming?", "... I don't know."],
  },
  {
    name: "Sentence with mixed punctuation marks (colon, comma, question mark)",
    input: "What do you think: Is this the answer, or not?",
    target: ["What do you think: Is this the answer, or not?"],
  },
  {
    name: "Sentence with parentheses and question mark",
    input: "Did you understand (after all)?",
    target: ["Did you understand (after all)?"],
  },
  {
    name: "Sentence with repeated punctuation marks (exclamation)",
    input: "What a great day!!! This is amazing!!!",
    target: ["What a great day!!!", "This is amazing!!!"],
  },
  {
    name: "Sentence with multiple short sentences and abbreviations",
    input: "Dr. Lee is busy. Mr. Brown is in a meeting.",
    target: ["Dr. Lee is busy.", "Mr. Brown is in a meeting."],
  },
  {
    name: "Sentence with only emojis",
    input: "🍕🍔🍟🍦",
    target: ["🍕🍔🍟🍦"],
  },
  {
    name: "Sentence with single quotes around a word",
    input: "The word 'apple' is red.",
    target: ["The word 'apple' is red."],
  },
  {
    name: "Sentence with an email and a period",
    input: "My email is example@domain.com. Please contact me.",
    target: ["My email is example@domain.com.", "Please contact me."],
  },
  {
    name: "Sentence with non-standard punctuation (pipe)",
    input: "This | is | a | test.",
    target: ["This | is | a | test."],
  },
  {
    name: "Sentence with a URL and a period after it",
    input: "You can find more info at https://www.website.com. It’s reliable.",
    target: ["You can find more info at https://www.website.com.", "It’s reliable."],
  },
  {
    name: "Sentence with multiple hashtags",
    input: "I love coding! #developer #javascript #testing",
    target: ["I love coding!", "#developer #javascript #testing"],
  },
  {
    name: "Sentence with numbers and currency",
    input: "I have $99.99 in my wallet. It's not enough.",
    target: ["I have $99.99 in my wallet.", "It's not enough."],
  },
  {
    name: "Sentence with mixed punctuation marks and parentheses",
    input: "Are you sure (really)? I don't think so!",
    target: ["Are you sure (really)?", "I don't think so!"],
  },
  {
    name: "Sentence with parentheses and ellipses",
    input: "This is a test (and it’s great)... seriously.",
    target: ["This is a test (and it’s great)... seriously."],
  },
  {
    name: "Sentence with an uncommon abbreviation",
    input: "The event is scheduled for noon PST. I’ll be there.",
    target: ["The event is scheduled for noon PST.", "I’ll be there."],
  },
  {
    name: "Sentence with a phone number",
    input: "Call me at 555-1234. Or email me at example@domain.com.",
    target: ["Call me at 555-1234.", "Or email me at example@domain.com."],
  },
  {
    name: "Sentence with nested punctuation (quotes inside quotes)",
    input: 'He said, "It\'s a test," and left.',
    target: ['He said, "It\'s a test," and left.'],
  },
  {
    name: "Sentences only containing a quotation",
    input: `"It's not like I'm using," Case heard someone say, as he shouldered his way through the crowd around the door of the Chat. "It's like my body's developed this massive drug deficiency."\nThis is a test.`,
    target: [`"It's not like I'm using," Case heard someone say, as he shouldered his way through the crowd around the door of the Chat.`, `"It's like my body's developed this massive drug deficiency."`, "This is a test."],
  },
  {
    name: "Sentence with a URL containing a question mark",
    input: "Visit https://www.example.com?query=test. It’s useful.",
    target: ["Visit https://www.example.com?query=test.", "It’s useful."],
  },
  {
    name: "Sentence with mixed punctuation and commas",
    input: "Hello, how are you? I'm fine, thanks.",
    target: ["Hello, how are you?", "I'm fine, thanks."],
  },
  {
    name: "Sentence with a comma before 'and'",
    input: "I like ice cream, and I like cake.",
    target: ["I like ice cream, and I like cake."],
  },
  {
    name: "Sentence with capital letters inside parentheses",
    input: "I went to the store (THE BIG ONE).",
    target: ["I went to the store (THE BIG ONE)."],
  },
  {
    name: "Sentence with dates and periods",
    input: "The event is on January 1st. It's a new year.",
    target: ["The event is on January 1st.", "It's a new year."],
  },
  {
    name: "Sentence with suffixes and periods",
    input: "Kokoro.js is powered by Transformers.js, a JavaScript library by Hugging Face.",
    target: ["Kokoro.js is powered by Transformers.js, a JavaScript library by Hugging Face."],
  },
  {
    name: "Non-splitting after a period",
    input: "Pi is 3.14 i.e., a mathematical constant. J.R.R. Tolkien wrote The Lord of the Rings. Wait... what? The files are /path/to/file.txt, VIDEO.MP4 and image.jpg.",
    target: ["Pi is 3.14 i.e., a mathematical constant.", "J.R.R. Tolkien wrote The Lord of the Rings.", "Wait... what?", "The files are /path/to/file.txt, VIDEO.MP4 and image.jpg."],
  },
  {
    name: "Long text with multiple sentences",
    input: `The sky above the port was the color of television, tuned to a dead channel.\n"It's not like I'm using," Case heard someone say, as he shouldered his way through the crowd around the door of the Chat. "It's like my body's developed this massive drug deficiency."\nIt was a Sprawl voice and a Sprawl joke. The Chatsubo was a bar for professional expatriates; you could drink there for a week and never hear two words in Japanese.\nThese were to have an enormous impact, not only because they were associated with Constantine, but also because, as in so many other areas, the decisions taken by Constantine (or in his name) were to have great significance for centuries to come. One of the main issues was the shape that Christian churches were to take, since there was not, apparently, a tradition of monumental church buildings when Constantine decided to help the Christian church build a series of truly spectacular structures. The main form that these churches took was that of the basilica, a multipurpose rectangular structure, based ultimately on the earlier Greek stoa, which could be found in most of the great cities of the empire. Christianity, unlike classical polytheism, needed a large interior space for the celebration of its religious services, and the basilica aptly filled that need. We naturally do not know the degree to which the emperor was involved in the design of new churches, but it is tempting to connect this with the secular basilica that Constantine completed in the Roman forum (the so-called Basilica of Maxentius) and the one he probably built in Trier, in connection with his residence in the city at a time when he was still caesar.`,
    target: [
      "The sky above the port was the color of television, tuned to a dead channel.",
      "\"It's not like I'm using,\" Case heard someone say, as he shouldered his way through the crowd around the door of the Chat.",
      "\"It's like my body's developed this massive drug deficiency.\"",
      "It was a Sprawl voice and a Sprawl joke.",
      "The Chatsubo was a bar for professional expatriates; you could drink there for a week and never hear two words in Japanese.",
      "These were to have an enormous impact, not only because they were associated with Constantine, but also because, as in so many other areas, the decisions taken by Constantine (or in his name) were to have great significance for centuries to come.",
      "One of the main issues was the shape that Christian churches were to take, since there was not, apparently, a tradition of monumental church buildings when Constantine decided to help the Christian church build a series of truly spectacular structures.",
      "The main form that these churches took was that of the basilica, a multipurpose rectangular structure, based ultimately on the earlier Greek stoa, which could be found in most of the great cities of the empire.",
      "Christianity, unlike classical polytheism, needed a large interior space for the celebration of its religious services, and the basilica aptly filled that need.",
      "We naturally do not know the degree to which the emperor was involved in the design of new churches, but it is tempting to connect this with the secular basilica that Constantine completed in the Roman forum (the so-called Basilica of Maxentius) and the one he probably built in Trier, in connection with his residence in the city at a time when he was still caesar.",
    ],
  },
];

// Tests adapted from https://github.com/textlint-rule/sentence-splitter/blob/master/test/sentence-splitter-test.ts
TESTS.push(
  {
    name: "Basic sentence splitting",
    input: "text",
    target: ["text"],
  },
  {
    name: "Should not split number",
    input: "Temperature is 40.2 degrees.",
    target: ["Temperature is 40.2 degrees."],
  },
  {
    name: "Should not split in pair string with same mark",
    input: 'I hear "I\'m back to home." from radio.',
    target: ['I hear "I\'m back to home." from radio.'],
  },
  {
    name: "Should not split in pair string",
    input: "彼は「ココにある。」と言った。",
    target: ["彼は「ココにある。」と言った。"],
  },
  // {
  //   name: "Should not split in pair string and correct after sentence",
  //   input: "彼は「ココにある。」と言った。だけではそれは違った。",
  //   target: ["彼は「ココにある。」と言った。", "だけではそれは違った。"],
  // },
  {
    name: "Should split by first line break",
    input: "text",
    target: ["text"],
  },
  {
    name: "Should split by last line break",
    input: "text\n",
    target: ["text"],
  },
  {
    name: "Should split by double line break",
    input: "text\n\ntext",
    target: ["text", "text"],
  },
  {
    name: "Should split by 。",
    input: "text。。text",
    target: ["text。。", "text"],
  },
  {
    name: "Should split by 。 and linebreak",
    input: "text。\ntext",
    target: ["text。", "text"],
  },
  {
    name: "Should split by . and whitespace",
    input: "1st text. 2nd text",
    target: ["1st text.", "2nd text"],
  },
  {
    name: "Should split by multiple whitespaces",
    input: "1st text.   2nd text",
    target: ["1st text.", "2nd text"],
  },
  {
    name: "Should support start and end whitespace",
    input: " text. ",
    target: ["text."],
  },
  {
    name: "Should split by text, whitespaces, and newline",
    input: "1st text. \n 2nd text",
    target: ["1st text.", "2nd text"],
  },
  {
    name: "Should split by !?",
    input: "text!?text",
    target: ["text!?", "text"],
  },
  {
    name: "Should split by last 。",
    input: "text。",
    target: ["text。"],
  },
  {
    name: "Should not split numbered list",
    input: "1. 1st text.\n2. 2nd text.\n10. 10th text.",
    target: ["1. 1st text.", "2. 2nd text.", "10. 10th text."],
  },
);

// Tests adapted from https://github.com/wikimedia/sentencex-js/blob/main/test/en.test.js
TESTS.push(
  {
    name: "Dr. title should not split",
    input: "This is Dr. Watson",
    target: ["This is Dr. Watson"],
  },
  {
    name: "Basic sentence split",
    input: "Roses Are Red. Violets Are Blue",
    target: ["Roses Are Red.", "Violets Are Blue"],
  },
  {
    name: "Exclamation and question split",
    input: "Hello! How are you?",
    target: ["Hello!", "How are you?"],
  },
  {
    name: "Simple period split",
    input: "This is a test.",
    target: ["This is a test."],
  },
  {
    name: "Mr. title should not split",
    input: "Mr. Smith went to Washington.",
    target: ["Mr. Smith went to Washington."],
  },
  {
    name: "Words ending in title-like suffixes should split",
    input: "He hit the drums. Then he hit the cymbals.",
    target: ["He hit the drums.", "Then he hit the cymbals."],
  },
  {
    name: "Surprise sentence should not split",
    input: "What a surprise?!",
    target: ["What a surprise?!"],
  },
  {
    name: "Ellipsis should not split",
    input: "That's all folks...",
    target: ["That's all folks..."],
  },
  {
    name: "Single line break should split",
    input: "First line\nSecond line",
    target: ["First line", "Second line"],
  },
  {
    name: "Double line break should split",
    input: "First line\nSecond line\n\nThird line",
    target: ["First line", "Second line", "Third line"],
  },
  {
    name: "Abbreviations should not split",
    input: "This is UK. Not US",
    target: ["This is UK.", "Not US"],
  },
  {
    name: "Dollar amount should not split",
    input: "This balloon costs $1.20",
    target: ["This balloon costs $1.20"],
  },
  {
    name: "Basic multiple sentence split",
    input: "Hello World. My name is Jonas.",
    target: ["Hello World.", "My name is Jonas."],
  },
  {
    name: "Basic question and sentence split",
    input: "What is your name? My name is Jonas.",
    target: ["What is your name?", "My name is Jonas."],
  },
  {
    name: "Exclamation and period split",
    input: "There it is! I found it.",
    target: ["There it is!", "I found it."],
  },
  {
    name: "Middle initial should not split",
    input: "My name is Jonas E. Smith.",
    target: ["My name is Jonas E. Smith."],
  },
  {
    name: "Page reference should not split",
    input: "Please turn to p. 55.",
    target: ["Please turn to p. 55."],
  },
  {
    name: "Co. abbreviation should not split",
    input: "Were Jane and co. at the party?",
    target: ["Were Jane and co. at the party?"],
  },
  {
    name: "Business name should not split",
    input: "They closed the deal with Pitt, Briggs & Co. at noon.",
    target: ["They closed the deal with Pitt, Briggs & Co. at noon."],
  },
  {
    name: "Mount abbreviation should not split",
    input: "I can see Mt. Fuji from here.",
    target: ["I can see Mt. Fuji from here."],
  },
  {
    name: "Saint abbreviation should not split",
    input: "St. Michael's Church is on 5th st. near the light.",
    target: ["St. Michael's Church is on 5th st. near the light."],
  },
  {
    name: "JFK Jr. should not split",
    input: "That is JFK Jr.'s book.",
    target: ["That is JFK Jr.'s book."],
  },
  {
    name: "Country abbreviation should not split",
    input: "I visited the U.S.A. last year.",
    target: ["I visited the U.S.A. last year."],
  },
  {
    name: "Dollar amount with period split",
    input: "She has $100.00. It is in her bag.",
    target: ["She has $100.00.", "It is in her bag."],
  },
  {
    name: "Email should not split",
    input: "Her email is Jane.Doe@example.com. I sent her an email.",
    target: ["Her email is Jane.Doe@example.com.", "I sent her an email."],
  },
  {
    name: "URL should not split",
    input: "The site is, https://www.example.50.com/new-site/awesome_content.html. Please check it out.",
    target: ["The site is, https://www.example.50.com/new-site/awesome_content.html.", "Please check it out."],
  },
  // {
  //   name: "Yahoo! should not split",
  //   input: "She works at Yahoo! in the accounting department.",
  //   target: ["She works at Yahoo! in the accounting department."],
  // },
  {
    name: "Multiple exclamations should split",
    input: "Hello!! Long time no see.",
    target: ["Hello!!", "Long time no see."],
  },
  {
    name: "Mixed punctuation should split",
    input: "Hello?! Is that you?",
    target: ["Hello?!", "Is that you?"],
  },
  // {
  //   name: "Numbered reference should not split",
  //   input: "Saint Maximus (died 250) is a Christian saint and martyr.[1] The emperor Decius published a decree ordering the veneration of busts of the deified emperors.",
  //   target: ["Saint Maximus (died 250) is a Christian saint and martyr.[1]", "The emperor Decius published a decree ordering the veneration of busts of the deified emperors."],
  // },
);

const STREAMED_TESTS = [
  {
    name: "Basic sentence splitting",
    input: ["I went", " to the", " store. I", " bought an apple for $1.", "99. It was", " a good deal."],
    target: ["I went to the store.", "I bought an apple for $1.99.", "It was a good deal."],
  },
  {
    name: "URL with query parameters",
    input: ["Visit https://www", ".example.", "com", "?query=test."],
    target: ["Visit https://www.example.com?query=test."],
  },
];

describe("Sentence splitting", () => {
  describe("synchronous", () => {
    for (const { name, input, target } of TESTS) {
      test(name, () => {
        expect(split(input)).toEqual(target);
      });
    }
  });

  describe("for loop", () => {
    test("synchronous for ... of", () => {
      const streamer = new TextSplitterStream();
      // Initial text
      streamer.push("Hello, how are you? I'm fine, thanks.");

      // 1. Consume the current stream
      const sentences = [];
      for (const sentence of streamer) {
        sentences.push(sentence);
      }
      expect(sentences).toEqual(["Hello, how are you?", "I'm fine, thanks."]);

      // 2. Consume the stream again
      streamer.push("This is a test. This is unfinish-");
      const sentences2 = [];
      for (const sentence of streamer) {
        sentences2.push(sentence);
      }
      expect(sentences2).toEqual(["This is a test.", "This is unfinish-"]);

      // 3. Consume the stream again
      streamer.push("ed.");
      const sentences3 = [];
      for (const sentence of streamer) {
        sentences3.push(sentence);
      }
      expect(sentences3).toEqual(["ed."]);
    });

    test("asynchronous for ... of", async () => {
      const streamer = new TextSplitterStream();
      // Initial text
      streamer.push("Hello, how are");

      // Consumes the stream asynchronously
      const sentences = [];
      const consumeStream = (async () => {
        for await (const sentence of streamer) {
          sentences.push(sentence);
        }
      })();

      setTimeout(() => {
        streamer.push(" you? I'm fine, thanks.");
      }, 10);
      setTimeout(() => {
        streamer.push(" This is a test. This is unfinish-");
      }, 20);
      setTimeout(() => {
        streamer.push("ed.");
      }, 30);
      setTimeout(() => {
        streamer.close();
      }, 40);

      await consumeStream;
      expect(sentences).toEqual(["Hello, how are you?", "I'm fine, thanks.", "This is a test.", "This is unfinish-ed."]);
    });
  });

  describe("streaming", () => {
    const tests = [
      // Pre-defined test cases
      ...STREAMED_TESTS,
      // Test that adding character by character (the most extreme case) also works correctly
      ...TESTS.map(({ name, input, target }) => ({ name, input: Array.from(input), target })),
    ];
    for (const { name, input, target } of tests) {
      test(name, async () => {
        const streamer = new TextSplitterStream();

        const sentences = [];
        const consumeStream = (async () => {
          for await (const sentence of streamer) {
            sentences.push(sentence);
          }
        })();
        streamer.push(...input);
        streamer.close();

        await consumeStream;
        expect(sentences).toEqual(target);
      });
    }
  });
});
