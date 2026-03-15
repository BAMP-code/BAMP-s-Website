export type Movie = {
  title: string;
  genres: string;
};

export type ChatSession = {
  recommendations: number[];
  currentRecIndex: number;
  allUserGivenTitles: [string, number][];
};
