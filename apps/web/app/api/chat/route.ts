import { NextRequest, NextResponse } from "next/server";
import { Chatbot } from "@/lib/chatbot/chatbot";
import type { Movie, ChatSession } from "@/lib/chatbot/types";
import { cookies } from "next/headers";
import { randomUUID } from "crypto";

export const runtime = "nodejs";

// Lazy-loaded data and chatbot instance
let chatbot: Chatbot | null = null;

function getChatbot(): Chatbot {
  if (!chatbot) {
    // These are loaded once and cached in memory
    const movies: Movie[] = require("@/lib/data/movies.json");
    const ratings: number[][] = require("@/lib/data/ratings.json");
    const sentiment: Record<string, string> = require("@/lib/data/sentiment.json");

    const llmEnabled = !!process.env.TOGETHER_API_KEY;
    chatbot = new Chatbot(movies, ratings, sentiment, llmEnabled);
  }
  return chatbot;
}

// In-memory session store (sessions lost on cold start — acceptable for a movie rec bot)
const sessions = new Map<string, ChatSession>();

function getOrCreateSession(sessionId: string): ChatSession {
  let session = sessions.get(sessionId);
  if (!session) {
    session = {
      recommendations: [],
      currentRecIndex: 0,
      allUserGivenTitles: [],
    };
    sessions.set(sessionId, session);
  }
  return session;
}

export async function POST(request: NextRequest) {
  try {
    const { message } = await request.json();

    if (!message || typeof message !== "string") {
      return NextResponse.json(
        { error: "Message is required" },
        { status: 400 },
      );
    }

    // Session management via cookies
    const cookieStore = await cookies();
    let sessionId = cookieStore.get("chatbot_session")?.value;
    if (!sessionId) {
      sessionId = randomUUID();
    }

    const bot = getChatbot();
    const session = getOrCreateSession(sessionId);
    const response = await bot.process(message, session);

    const res = NextResponse.json({ response });
    res.cookies.set("chatbot_session", sessionId, {
      httpOnly: true,
      sameSite: "strict",
      maxAge: 60 * 60, // 1 hour
    });

    return res;
  } catch (error) {
    console.error("Chatbot error:", error);
    return NextResponse.json(
      { response: "Sorry, something went wrong. Please try again." },
      { status: 500 },
    );
  }
}
