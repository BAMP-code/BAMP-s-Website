import OpenAI from "openai";

let client: OpenAI | null = null;

function getClient(): OpenAI {
  if (!client) {
    const apiKey = process.env.TOGETHER_API_KEY;
    if (!apiKey) {
      throw new Error("TOGETHER_API_KEY not set");
    }
    client = new OpenAI({
      apiKey,
      baseURL: "https://api.together.xyz",
    });
  }
  return client;
}

export async function simpleLlmCall(
  systemPrompt: string,
  message: string,
  options?: { model?: string; maxTokens?: number; stop?: string[] },
): Promise<string> {
  const c = getClient();
  const completion = await c.chat.completions.create({
    messages: [
      { role: "system", content: systemPrompt },
      { role: "user", content: message },
    ],
    model: options?.model ?? "mistralai/Mixtral-8x7B-Instruct-v0.1",
    max_tokens: options?.maxTokens ?? 256,
    stop: options?.stop,
  });

  return completion.choices[0].message.content ?? "";
}
