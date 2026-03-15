import { describe, it, expect } from "vitest";
import { extractTitles, findMoviesByTitle, combineTitles } from "@/lib/chatbot/titles";
import type { Movie } from "@/lib/chatbot/types";

describe("extractTitles", () => {
  it("should extract a single quoted title", () => {
    expect(extractTitles('I liked "The Notebook" a lot.')).toEqual([
      "The Notebook",
    ]);
  });

  it("should extract multiple quoted titles", () => {
    expect(
      extractTitles('I liked "Titanic" and "The Matrix"'),
    ).toEqual(["Titanic", "The Matrix"]);
  });

  it("should return empty array when no titles found", () => {
    expect(extractTitles("I like movies")).toEqual([]);
  });
});

describe("findMoviesByTitle", () => {
  const titles: Movie[] = [
    { title: "Toy Story (1995)", genres: "Animation" },
    { title: "Titanic (1997)", genres: "Drama" },
    { title: "Matrix, The (1999)", genres: "Sci-Fi" },
  ];

  it("should find exact match without year", () => {
    expect(findMoviesByTitle("Toy Story", titles)).toEqual([0]);
  });

  it("should find match with article handling", () => {
    expect(findMoviesByTitle("The Matrix", titles)).toEqual([2]);
  });

  it("should return empty for non-existent movie", () => {
    expect(findMoviesByTitle("Nonexistent Movie", titles)).toEqual([]);
  });
});

describe("combineTitles", () => {
  it("should return single title as-is", () => {
    expect(combineTitles(["Titanic"])).toBe("Titanic");
  });

  it("should join two with 'and'", () => {
    expect(combineTitles(["Titanic", "The Matrix"])).toBe(
      "Titanic and The Matrix",
    );
  });

  it("should join three with commas and 'and'", () => {
    expect(combineTitles(["A", "B", "C"])).toBe("A, B and C");
  });
});
