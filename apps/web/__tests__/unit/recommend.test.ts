import { describe, it, expect } from "vitest";
import { binarize, similarity, recommend } from "@/lib/chatbot/recommend";

describe("binarize", () => {
  it("should binarize ratings above threshold to 1", () => {
    const ratings = [[3.0, 4.0, 5.0]];
    const result = binarize(ratings);
    expect(Array.from(result[0])).toEqual([1, 1, 1]);
  });

  it("should binarize ratings at or below threshold to -1", () => {
    const ratings = [[1.0, 2.0, 2.5]];
    const result = binarize(ratings);
    expect(Array.from(result[0])).toEqual([-1, -1, -1]);
  });

  it("should keep 0 values as 0", () => {
    const ratings = [[0, 3.0, 0, 1.0]];
    const result = binarize(ratings);
    expect(Array.from(result[0])).toEqual([0, 1, 0, -1]);
  });
});

describe("similarity", () => {
  it("should return 1 for identical vectors", () => {
    const v = [1, 1, 1];
    expect(similarity(v, v)).toBeCloseTo(1.0);
  });

  it("should return -1 for opposite vectors", () => {
    const u = [1, 1, 1];
    const v = [-1, -1, -1];
    expect(similarity(u, v)).toBeCloseTo(-1.0);
  });

  it("should return 0 for orthogonal vectors", () => {
    const u = [1, 0];
    const v = [0, 1];
    expect(similarity(u, v)).toBeCloseTo(0.0);
  });

  it("should return 0 if either vector is zero", () => {
    expect(similarity([0, 0], [1, 1])).toBe(0);
  });
});

describe("recommend", () => {
  it("should return indices of unrated movies sorted by predicted preference", () => {
    // 3 movies, 2 users
    const rawRatings = [
      [3.0, 4.0],  // movie 0
      [1.0, 5.0],  // movie 1
      [0, 0],      // movie 2 - no ratings
    ];
    const binRatings = binarize(rawRatings);

    // User liked movie 0, disliked movie 1
    const userRatings = [1, -1, 0];
    const recs = recommend(userRatings, binRatings, 1);
    expect(recs).toHaveLength(1);
    expect(recs[0]).toBe(2); // only unrated movie
  });

  it("should exclude already-rated movies", () => {
    const binRatings = binarize([
      [3.0, 1.0],
      [4.0, 2.0],
    ]);
    const userRatings = [1, 1]; // both rated
    const recs = recommend(userRatings, binRatings, 5);
    expect(recs).toHaveLength(0);
  });
});
