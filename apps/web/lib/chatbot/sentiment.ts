const NEGATION_LIST = ["not", "never", "no", "didn't", "didn't really", "not that"];
const PAST_TENSE = ["loved", "enjoyed", "liked", "hated", "disliked"];
const PRESENT_TENSE = ["love", "enjoy", "like", "hate", "dislike"];

/**
 * Extract sentiment from preprocessed input.
 * Returns 1 (positive), -1 (negative), or 0 (neutral).
 */
export function extractSentiment(
  preprocessedInput: string,
  sentimentDict: Record<string, string>,
): number {
  let numPos = 0;
  let numNeg = 0;

  // Remove quoted titles from the text
  let sentence = "";
  const left = preprocessedInput.indexOf('"');
  if (left !== -1) {
    sentence += preprocessedInput.slice(0, left);
    const remaining = preprocessedInput.slice(left + 1);
    const right = remaining.indexOf('"');
    if (right !== -1) {
      sentence += remaining.slice(right + 1);
    }
  } else {
    sentence = preprocessedInput;
  }

  const words = sentence.split(" ");

  // Check for negation
  let opposite = false;
  for (const neg of NEGATION_LIST) {
    if (sentence.includes(neg)) {
      opposite = true;
    }
  }

  // Check sentiment of each word
  for (let i = 0; i < words.length; i++) {
    let word = words[i];

    // Convert past tense to present
    const pastIdx = PAST_TENSE.indexOf(word);
    if (pastIdx !== -1) {
      word = PRESENT_TENSE[pastIdx];
    }

    if (word in sentimentDict) {
      if (sentimentDict[word] === "pos") {
        numPos += opposite ? -1 : 1;
      } else if (sentimentDict[word] === "neg") {
        numNeg += opposite ? -1 : 1;
      }
    }
  }

  if (numPos > numNeg) return 1;
  if (numPos < numNeg) return -1;
  return 0;
}
