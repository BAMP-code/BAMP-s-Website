function randomChoice<T>(arr: T[]): T {
  return arr[Math.floor(Math.random() * arr.length)];
}

export function titleInputErrorMessage(): string {
  return randomChoice([
    "Hmm, I don't recognize a movie title in what you just said. Would you please tell me about a movie you've seen recently?",
    "I didn't catch any movie title in your input. Could you tell me the name of a movie you watched?",
    "It seems like there's no movie title mentioned. Please share a movie title so I can help you.",
    "I'm not sure which movie you're referring to. Could you provide the title of a movie you've seen?",
    "I couldn't find any movie title in your message. Can you mention a movie title?",
    "Oops, I don't see a movie title in your response. Could you let me know which movie you're talking about?",
    "It looks like you forgot to include a movie title. Could you please tell me one?",
    "I'm having trouble identifying a movie title from your message. Can you mention one?",
    "Hmm, I need a movie title to continue. Could you share one?",
    "I can't tell which movie you're referring to. Could you specify the title?",
  ]);
}

export function missingMovieErrorMessage(title: string): string {
  return randomChoice([
    `I've never heard of '${title}', sorry... Tell me about another movie you liked.`,
    `Hmm, '${title}' doesn't seem to be in my database. Could you mention another movie?`,
    `Sorry, I couldn't find '${title}'. Please tell me about a different movie you've seen.`,
    `I don't recognize '${title}'. Maybe try another movie title?`,
    `'${title}' doesn't ring a bell. Can you provide another movie you enjoyed?`,
    `Unfortunately, I couldn't locate '${title}'. Do you have another movie in mind?`,
    `Hmm, '${title}' isn't in my system. Want to tell me about a different movie?`,
    `Sorry, but I don't have any information on '${title}'. Could you name another movie?`,
    `'${title}' seems to be off my radar. Can you share another title?`,
    `I'm not familiar with '${title}'. Do you have another movie to talk about?`,
  ]);
}

export function sentimentErrorMessage(title: string): string {
  return randomChoice([
    `I'm sorry, I'm not sure if you liked or disliked '${title}'. Tell me more about it. Please include the movie title in your input again.`,
    `I couldn't tell how you felt towards '${title}'. Could you clarify how you felt about it? Make sure to mention the movie title in your input again.`,
    `It seems unclear whether you liked or disliked '${title}'. Please tell me more and incorporate the movie title in your input again.`,
    `I'm not certain about your feelings towards '${title}'. Can you tell me if you liked it or not? Include the title in your input as well.`,
    `I need a bit more information on your feeling about '${title}'. Did you like or dislike it? Please include the movie title again as well`,
    `Your opinion on '${title}' isn't clear to me. Can you share more details? I will also need you to include the movie title in your input again.`,
    `I'm unsure if you enjoyed '${title}' or not. Could you tell me more? You must include the movie title in your input again.`,
    `Did '${title}' impress you, or was it disappointing? Let me know! Please specify the movie title again as well.`,
    `I couldn't quite catch whether you liked '${title}' or not. Can you clarify? Please also mention the movie title again.`,
    `Tell me more—was '${title}' a hit or a miss for you? Mention the movie title in the new input as well.`,
  ]);
}

export function posSentimentMessage(title: string): string {
  return randomChoice([
    `Ok, you liked '${title}'!`,
    `Great, it seems you enjoyed '${title}'.`,
    `Awesome, '${title}' really resonated with you!`,
    `Fantastic! '${title}' appears to be one of your favorites.`,
    `Nice! It sounds like you had a good time watching '${title}'.`,
    `Glad to hear you enjoyed '${title}'!`,
    `Sounds like '${title}' was a winner for you!`,
    `You really liked '${title}', didn't you? That's great!`,
    `Cool! '${title}' must have been a fun watch for you.`,
    `Happy to know you had a positive experience with '${title}'!`,
  ]);
}

export function negSentimentMessage(title: string): string {
  return randomChoice([
    `Ok, you didn't like '${title}'!!`,
    `It seems '${title}' wasn't your cup of tea.`,
    `Understood, '${title}' didn't impress you.`,
    `Alright, you weren't a fan of '${title}'.`,
    `Got it, '${title}' didn't quite meet your expectations.`,
    `Looks like '${title}' wasn't a hit for you.`,
    `Sorry that '${title}' wasn't what you hoped for.`,
    `Noted! '${title}' didn't live up to your expectations.`,
    `I see, '${title}' didn't do it for you.`,
    `Sounds like '${title}' was a letdown for you.`,
  ]);
}

export function tellMeMoreMessage(): string {
  return randomChoice([
    "Tell me what you thought of another movie.",
    "Could you share your thoughts on another film?",
    "Please tell me about another movie you've seen.",
    "Tell me more about another movie that caught your attention.",
    "Could you share another movie that captured your interest?",
    "I'd love to hear about another movie you watched!",
    "Tell me about another film you recently enjoyed (or disliked).",
    "Do you have another movie you'd like to discuss?",
    "Have another movie in mind? Tell me about it!",
    "What's another movie you've seen?",
  ]);
}

export function recommendationMessage(recTitle: string): string {
  return randomChoice([
    `Given what you told me, I think you would like '${recTitle}'. Would you like more recommendations? (yes/no)`,
    `Based on your taste, you might enjoy '${recTitle}'. Do you want another recommendation? (yes/no)`,
    `Based on your likes, how about '${recTitle}'? Would you like to hear another? (yes/no)`,
    `I recommend '${recTitle}'. Should I suggest another movie? (yes/no)`,
    `You might also like '${recTitle}'. Interested in another recommendation? (yes/no)`,
    `I have a feeling you'll enjoy '${recTitle}'. Would you like me to suggest more? (yes/no)`,
    `'${recTitle}' seems like a great fit for you! Want another suggestion? (yes/no)`,
    `How about '${recTitle}'? Let me know if you'd like more recommendations! (yes/no)`,
    `You might find '${recTitle}' interesting. Would you like to hear another suggestion? (yes/no)`,
    `I think '${recTitle}' could be right up your alley! Need more recommendations? (yes/no)`,
  ]);
}
