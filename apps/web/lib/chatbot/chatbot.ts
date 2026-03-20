import type { Movie, ChatSession } from "./types";
import { extractTitles, findMoviesByTitle, combineTitles } from "./titles";
import { extractSentiment } from "./sentiment";
import { recommend, binarize } from "./recommend";
import {
  titleInputErrorMessage,
  missingMovieErrorMessage,
  sentimentErrorMessage,
  posSentimentMessage,
  negSentimentMessage,
  tellMeMoreMessage,
  recommendationMessage,
} from "./messages";
import { simpleLlmCall } from "./llm";
import { MISSING_TITLE_PROMPT, PERSONALITY_PROMPT } from "./prompts";

export class Chatbot {
  private titles: Movie[];
  private ratings: Int8Array[];
  private sentiment: Record<string, string>;
  private llmEnabled: boolean;

  constructor(
    titles: Movie[],
    rawRatings: number[][],
    sentiment: Record<string, string>,
    llmEnabled = false,
  ) {
    this.titles = titles;
    this.ratings = binarize(rawRatings);
    this.sentiment = sentiment;
    this.llmEnabled = llmEnabled;
  }

  greeting(): string {
    return "Hello! I'm Bongtail, your movie recommender chatbot. How can I help you today?";
  }

  async process(line: string, session: ChatSession): Promise<string> {
    // Handle ongoing recommendation flow
    if (session.recommendations.length > 0 && session.currentRecIndex < session.recommendations.length) {
      return this.continuousRecResponse(line, session);
    }

    // Extract movie titles
    const extractedTitles = extractTitles(line);

    if (extractedTitles.length === 0) {
      if (this.llmEnabled) {
        return this.llmModeMissingTitleMessage(line);
      }
      return titleInputErrorMessage();
    }

    // Get overall sentiment
    const sentiment = extractSentiment(line, this.sentiment);
    if (sentiment === 0) {
      if (this.llmEnabled) {
        const personality = await this.llmPersonalityResponse(line);
        return personality + " " + sentimentErrorMessage(extractedTitles[0]);
      }
      return sentimentErrorMessage(extractedTitles[0]);
    }

    // Check all titles exist
    for (const title of extractedTitles) {
      const movies = findMoviesByTitle(title, this.titles);
      if (movies.length === 0) {
        if (this.llmEnabled) {
          const personality = await this.llmPersonalityResponse(line);
          return personality + " " + missingMovieErrorMessage(title);
        }
        return missingMovieErrorMessage(title);
      }
      session.allUserGivenTitles.push([title, sentiment]);
    }

    const allTitles = combineTitles(extractedTitles);

    if (session.allUserGivenTitles.length < 5) {
      const sentMsg = sentiment === 1
        ? posSentimentMessage(allTitles)
        : negSentimentMessage(allTitles);
      const more = tellMeMoreMessage();
      if (this.llmEnabled) {
        const personality = await this.llmPersonalityResponse(line);
        return personality + " " + sentMsg + " " + more;
      }
      return sentMsg + " " + more;
    }

    if (session.allUserGivenTitles.length >= 5) {
      const sentMsg = sentiment === 1
        ? posSentimentMessage(allTitles)
        : negSentimentMessage(allTitles);
      const recFlow = this.recommendationFlow(session);
      session.allUserGivenTitles = [];
      if (this.llmEnabled) {
        const personality = await this.llmPersonalityResponse(line);
        return personality + " " + sentMsg + " " + recFlow;
      }
      return sentMsg + " " + recFlow;
    }

    return "I processed your input!";
  }

  private recommendationFlow(session: ChatSession): string {
    const userRatings = new Array(this.titles.length).fill(0);
    for (const [title, sentiment] of session.allUserGivenTitles) {
      const indices = findMoviesByTitle(title, this.titles);
      for (const idx of indices) {
        userRatings[idx] = sentiment;
      }
    }

    const recommended = recommend(userRatings, this.ratings);
    session.recommendations = recommended;
    session.currentRecIndex = 0;

    if (recommended.length > 0) {
      const recTitle = this.titles[recommended[0]].title;
      return recommendationMessage(recTitle);
    }
    return "I'm sorry, I couldn't find any recommendations for you.";
  }

  private continuousRecResponse(userResponse: string, session: ChatSession): string {
    const yesPattern = /^(yes|y|yeah!?|sure!?|ok(ay)?!?)(\s|$)/;
    const noPattern = /^(no|n|nah!?|nope!?)(\s|$)/;
    const cleaned = userResponse.trim().toLowerCase();

    if (yesPattern.test(cleaned)) {
      if (session.currentRecIndex < session.recommendations.length - 1) {
        session.currentRecIndex++;
        const recIndex = session.recommendations[session.currentRecIndex];
        const recTitle = this.titles[recIndex].title;
        return recommendationMessage(recTitle);
      }
      session.recommendations = [];
      session.currentRecIndex = 0;
      if (this.llmEnabled) {
        return "The reel has run dry, my friend. Like the final frame of a film, some things must end. But cinema is infinite—perhaps it's time to revisit an old favorite?";
      }
      return "That's all the recommendations I have for now! Tell me about another movie.";
    }

    if (noPattern.test(cleaned)) {
      session.recommendations = [];
      session.currentRecIndex = 0;
      return "Alright! Let me know if you want to talk about more movies.";
    }

    return "I didn't quite understand that. Would you like another recommendation? (yes/no)";
  }

  private async llmModeMissingTitleMessage(input: string): Promise<string> {
    return simpleLlmCall(MISSING_TITLE_PROMPT, input, { stop: ["\n"] });
  }

  private async llmPersonalityResponse(input: string): Promise<string> {
    return simpleLlmCall(PERSONALITY_PROMPT, input, { stop: ["\n"] });
  }
}
