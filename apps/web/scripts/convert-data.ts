/**
 * One-time script to convert data/*.txt files into JSON for the Next.js chatbot.
 *
 * Usage: npx tsx apps/web/scripts/convert-data.ts
 */

import * as fs from "fs";
import * as path from "path";

const DATA_DIR = path.resolve(__dirname, "../../../data");
const OUT_DIR = path.resolve(__dirname, "../lib/data");

function loadMovies(): { title: string; genres: string }[] {
  const content = fs.readFileSync(path.join(DATA_DIR, "movies.txt"), "utf-8");
  return content
    .split("\n")
    .filter((l) => l.trim())
    .map((line) => {
      const parts = line.split("%");
      let title = parts[1];
      if (title.startsWith('"') && title.endsWith('"')) {
        title = title.slice(1, -1);
      }
      return { title, genres: parts[2] };
    });
}

function loadRatings(numMovies: number): number[][] {
  const content = fs.readFileSync(path.join(DATA_DIR, "ratings.txt"), "utf-8");
  const lines = content.split("\n").filter((l) => l.trim());

  // Find number of users
  const userIds = new Set<number>();
  for (const line of lines) {
    const parts = line.split("%");
    userIds.add(parseInt(parts[0]));
  }
  const numUsers = userIds.size;

  // Build matrix
  const mat: number[][] = Array.from({ length: numMovies }, () =>
    new Array(numUsers).fill(0),
  );

  for (const line of lines) {
    const parts = line.split("%");
    const userId = parseInt(parts[0]);
    const movieId = parseInt(parts[1]);
    const rating = parseFloat(parts[2]);
    mat[movieId][userId] = rating;
  }

  return mat;
}

function loadSentiment(): Record<string, string> {
  const content = fs.readFileSync(
    path.join(DATA_DIR, "sentiment.txt"),
    "utf-8",
  );
  const dict: Record<string, string> = {};
  for (const line of content.split("\n").filter((l) => l.trim())) {
    const [word, val] = line.split(",");
    dict[word] = val;
  }
  return dict;
}

// Run conversion
if (!fs.existsSync(OUT_DIR)) {
  fs.mkdirSync(OUT_DIR, { recursive: true });
}

console.log("Converting movies...");
const movies = loadMovies();
fs.writeFileSync(
  path.join(OUT_DIR, "movies.json"),
  JSON.stringify(movies),
);
console.log(`  ${movies.length} movies`);

console.log("Converting ratings...");
const ratings = loadRatings(movies.length);
fs.writeFileSync(
  path.join(OUT_DIR, "ratings.json"),
  JSON.stringify(ratings),
);
console.log(`  ${ratings.length} x ${ratings[0]?.length ?? 0} matrix`);

console.log("Converting sentiment...");
const sentiment = loadSentiment();
fs.writeFileSync(
  path.join(OUT_DIR, "sentiment.json"),
  JSON.stringify(sentiment),
);
console.log(`  ${Object.keys(sentiment).length} words`);

console.log("Done!");
