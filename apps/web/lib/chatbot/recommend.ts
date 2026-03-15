/**
 * Binarize a ratings matrix.
 * Values > threshold → 1, values <= threshold and != 0 → -1, 0 stays 0.
 */
export function binarize(
  ratings: number[][],
  threshold = 2.5,
): Int8Array[] {
  return ratings.map((row) => {
    const bin = new Int8Array(row.length);
    for (let j = 0; j < row.length; j++) {
      if (row[j] === 0) bin[j] = 0;
      else bin[j] = row[j] > threshold ? 1 : -1;
    }
    return bin;
  });
}

/**
 * Cosine similarity between two vectors.
 */
export function similarity(u: Int8Array | number[], v: Int8Array | number[]): number {
  let dot = 0;
  let normU = 0;
  let normV = 0;

  for (let i = 0; i < u.length; i++) {
    dot += u[i] * v[i];
    normU += u[i] * u[i];
    normV += v[i] * v[i];
  }

  const nU = Math.sqrt(normU);
  const nV = Math.sqrt(normV);

  if (nU === 0 || nV === 0) return 0;
  return dot / (nU * nV);
}

/**
 * Generate k movie recommendations using collaborative filtering.
 * userRatings: binarized 1D array (length = num_movies)
 * ratingsMatrix: binarized 2D array (num_movies x num_users)
 */
export function recommend(
  userRatings: number[],
  ratingsMatrix: Int8Array[],
  k = 10,
): number[] {
  const scores: [number, number][] = [];

  const ratedMovies: number[] = [];
  const unratedMovies: number[] = [];

  for (let i = 0; i < userRatings.length; i++) {
    if (userRatings[i] !== 0) ratedMovies.push(i);
    else unratedMovies.push(i);
  }

  for (const movie of unratedMovies) {
    let sim = 0;
    for (const ratedMovie of ratedMovies) {
      const s = similarity(ratingsMatrix[movie], ratingsMatrix[ratedMovie]);
      sim += userRatings[ratedMovie] * s;
    }
    scores.push([sim, movie]);
  }

  scores.sort((a, b) => b[0] - a[0]);
  return scores.slice(0, k).map(([, idx]) => idx);
}
