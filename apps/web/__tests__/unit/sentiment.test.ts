import { describe, it, expect } from "vitest";
import { extractSentiment } from "@/lib/chatbot/sentiment";

const sentimentDict: Record<string, string> = {
  love: "pos",
  like: "pos",
  enjoy: "pos",
  hate: "neg",
  dislike: "neg",
  terrible: "neg",
  great: "pos",
};

describe("extractSentiment", () => {
  it("should detect positive sentiment", () => {
    expect(extractSentiment('I liked "The Notebook"', sentimentDict)).toBe(1);
  });

  it("should detect negative sentiment", () => {
    expect(extractSentiment('I hated "The Notebook"', sentimentDict)).toBe(-1);
  });

  it("should detect negation flipping sentiment", () => {
    expect(
      extractSentiment('I did not like "The Notebook"', sentimentDict),
    ).toBe(-1);
  });

  it("should return 0 for neutral input", () => {
    expect(extractSentiment('I saw "The Notebook"', sentimentDict)).toBe(0);
  });

  it("should handle past tense", () => {
    expect(extractSentiment('I loved "Titanic"', sentimentDict)).toBe(1);
    expect(extractSentiment('I enjoyed "Titanic"', sentimentDict)).toBe(1);
  });
});
