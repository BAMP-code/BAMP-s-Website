import type { Movie } from "./types";

/**
 * Extract quoted movie titles from user input.
 */
export function extractTitles(input: string): string[] {
  const pattern = /"([^"]*)"/g;
  const matches: string[] = [];
  let match;
  while ((match = pattern.exec(input)) !== null) {
    matches.push(match[1]);
  }
  return matches;
}

/**
 * Find movie indices matching a title.
 * Handles article movement (e.g., "The Matrix" → "Matrix, The").
 */
export function findMoviesByTitle(
  title: string,
  titles: Movie[],
): number[] {
  const articles = ["a", "an", "the"];
  const res: number[] = [];

  const oldTitle = title.toLowerCase().split(" ");
  const titleWords = [...oldTitle];

  // Move articles to end (matching the database format)
  if (
    titleWords.length > 1 &&
    articles.includes(titleWords[0].toLowerCase())
  ) {
    const lastWord = titleWords[titleWords.length - 1];
    // Check if last word looks like a year in parens e.g. (1995)
    const isYear = /^\(\d+\)$/.test(lastWord);

    if (isYear) {
      // Move article before the year: "The Matrix (1999)" → "Matrix, The (1999)"
      const article = titleWords.shift()!;
      titleWords[titleWords.length - 2] = titleWords[titleWords.length - 2] + ",";
      titleWords.splice(titleWords.length - 1, 0, article);
    } else {
      // Move article to end: "The Matrix" → "Matrix, The"
      const article = titleWords.shift()!;
      titleWords[titleWords.length - 1] = titleWords[titleWords.length - 1] + ",";
      titleWords.push(article);
    }
  }

  for (let i = 0; i < titles.length; i++) {
    const data = titles[i].title.toLowerCase().split(" ");
    // Match with or without year suffix
    if (
      arraysEqual(titleWords, data) ||
      arraysEqual(titleWords, data.slice(0, -1)) ||
      arraysEqual(oldTitle, data) ||
      arraysEqual(oldTitle, data.slice(0, -1))
    ) {
      res.push(i);
    }
  }

  return res;
}

function arraysEqual(a: string[], b: string[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

/**
 * Combine multiple titles into a human-readable string.
 */
export function combineTitles(titles: string[]): string {
  if (titles.length === 1) return titles[0];
  return titles.slice(0, -1).join(", ") + " and " + titles[titles.length - 1];
}
