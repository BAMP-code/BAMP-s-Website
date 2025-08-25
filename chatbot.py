import util
import re
import random
from pydantic import BaseModel, Field
import requests
import numpy as np


# noinspection PyMethodMayBeStatic
class Chatbot:

    def __init__(self, llm_enabled=False):
       
        self.name = 'Bongtail'

        self.llm_enabled = llm_enabled

        # This matrix has the following shape: num_movies x num_users
        # The values stored in each row i and column j is the rating for
        # movie i by user j
        self.titles, ratings = util.load_ratings('data/ratings.txt')
        self.sentiment = util.load_sentiment_dictionary('data/sentiment.txt')

        self.recommendations = []
        self.current_rec_index = 0
        self.all_user_given_titles = []


        # Binarize the movie ratings before storing the binarized matrix.
        binarized_ratings = self.binarize(ratings)
        self.ratings = binarized_ratings
      

    def greeting(self):
        """Return a message that the chatbot uses to greet the user."""
       
        greeting_message = "Hello! I'm Bongtail, your movie recommender chatbot. How can I help you today?"
        
        return greeting_message

    def goodbye(self):
        """
        Return a message that the chatbot uses to bid farewell to the user.
        """

        goodbye_message = "Have a nice day!"

        return goodbye_message
    

    def process(self, line):
        """Process a line of input from the REPL and generate a response.

        This is the method that is called by the REPL loop directly with user
        input.

        You should delegate most of the work of processing the user's input to
        the helper functions you write later in this class.

        Takes the input string from the REPL and call delegated functions that
          1) extract the relevant information, and
          2) transform the information into a response to the user.

        Example:
          resp = chatbot.process('I loved "The Notebook" so much!!')
          print(resp) // prints 'So you loved "The Notebook", huh?'

        :param line: a user-supplied line of text
        :returns: a string containing the chatbot's response to the user input
        """
        
        #Preprocess the input.
        preprocessed = self.preprocess(line)

        if self.recommendations and self.current_rec_index < len(self.recommendations):
            return self.continuous_rec_response(line)

        # Extract movie titles
        extracted_titles = self.extract_titles(preprocessed)

        if not extracted_titles:
            if self.llm_enabled:
                return self.llm_mode_missing_title_message(preprocessed)
            else:
                return self.title_input_error_message()
                
        #LLM or not
        if self.llm_enabled:
            response = "I processed {} in LLM Programming mode!!".format(line)
            extracted_titles = [self.translate_titles(title) for title in extracted_titles]
        else:
            response = "I processed {} in Starter (GUS) mode!!".format(line)
        
        #Get overall sentiment
        sentiment = self.extract_sentiment(preprocessed)
        if sentiment == 0:
            if self.llm_enabled: 
                return self.llm_personality_response(preprocessed) + " " + self.sentiment_error_message(extracted_titles[0])
            else:
                return self.sentiment_error_message(extracted_titles[0])

        for title in extracted_titles:
            movies = self.find_movies_by_title(title)
            if not movies:
                if self.llm_enabled: 
                    return  self.llm_personality_response(preprocessed) + " " + self.missing_movie_error_message(title)
                else: 
                    return self.missing_movie_error_message(title)
            self.all_user_given_titles.append((title, sentiment))

        #Combine if multiple titles
        all_titles = self.combine_titles(extracted_titles)
            
        if len(self.all_user_given_titles) < 5:
            if sentiment == 1:
                if self.llm_enabled: 
                    return self.llm_personality_response(preprocessed) + " " + self.pos_sentiment_message(all_titles) + " " + self.tell_me_more_message()
                else: 
                    return self.pos_sentiment_message(all_titles) + " " + self.tell_me_more_message()
            elif sentiment == -1:
                if self.llm_enabled: 
                    return self.llm_personality_response(preprocessed) + " " + self.neg_sentiment_message(all_titles) + " " + self.tell_me_more_message()
                else: 
                    return self.neg_sentiment_message(all_titles) + " " + self.tell_me_more_message()
        elif len(self.all_user_given_titles) == 5:
            if sentiment == 1:
                if self.llm_enabled: 
                    response = self.pos_sentiment_message(all_titles) + " " + self.recommendation_flow()
                else:
                    response = self.pos_sentiment_message(all_titles) + " " + self.recommendation_flow() 
            elif sentiment == -1:
                if self.llm_enabled: 
                    response = self.neg_sentiment_message(all_titles) + " " + self.recommendation_flow()
                else: 
                    response = self.neg_sentiment_message(all_titles) + " " + self.recommendation_flow()
            # Reset the state so that new inputs start fresh.
            self.all_user_given_titles = []
            if self.llm_enabled: 
                return self.llm_personality_response(preprocessed) + " " + response
            else: 
                return response
              
        return response

    @staticmethod
    def preprocess(text):
        """Do any general-purpose pre-processing before extracting information
        from a line of text.

        Given an input line of text, this method should do any general
        pre-processing and return the pre-processed string. The outputs of this
        method will be used as inputs (instead of the original raw text) for the
        extract_titles, extract_sentiment, extract_sentiment_for_movies, and
        extract_emotion methods.

        Note that this method is intentially made static, as you shouldn't need
        to use any attributes of Chatbot in this method.

        :param text: a user-supplied line of text
        :returns: the same text, pre-processed
        """

        return text

    def extract_titles(self, preprocessed_input):
        """Extract potential movie titles from a line of pre-processed text.

        Given an input text which has been pre-processed with preprocess(),
        this method should return a list of movie titles that are potentially
        in the text.

        - If there are no movie titles in the text, return an empty list.
        - If there is exactly one movie title in the text, return a list
        containing just that one movie title.
        - If there are multiple movie titles in the text, return a list
        of all movie titles you've extracted from the text.

        Example:
          potential_titles = chatbot.extract_titles(chatbot.preprocess(
                                            'I liked "The Notebook" a lot.'))
          print(potential_titles) // prints ["The Notebook"]

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: list of movie titles that are potentially in the text
        """
        pattern = '"([^"]*)"'
        matches = re.findall(pattern, preprocessed_input)
        res = []
        for m in matches:
            res.append(m)
        return res
    
    def translate_titles(self, title):
        system_prompt = """
        You are a movie title translation bot. Your task is to translate movie titles from German, Spanish, French, Danish, and Italian into English. 
        Let's do this step-by-step:
         - Read the user-provided title carefully.
         - If the title is entirely in English, return it as is.
         - If the title contains any German, Spanish, French, Danish, or Italian words, detect which language it is.
         - Translate the title into English. Have three versions and return only the most likely English title.
         - Respond ONLY with the translated movie title in English and nothing else.
         - Do not provide explanations, no comments, no quotations, nor additional text, ONLY the movie title.
 
       Example inputs and expected outputs:
        "El Cuaderno" -> "The Notebook"
        "Jernmand" -> "Iron Man"
        "Den Fantastiske Spider-Man" -> "The Amazing Spider-Man"
        "The Notebook" -> "The Notebook"  #English titles remain unchanged
        """

        message = title

        # Our llm will stop when it sees a newline character.
        # You can add more stop tokens to the list if you want to stop on other tokens!
        # Feel free to remove the stop parameter if you want the llm to run to completion.
        stop = ["\n"]

        response = util.simple_llm_call(system_prompt, message, stop=stop).strip()

        return response.strip('"“”')

    def find_movies_by_title(self, title):
        """ Given a movie title, return a list of indices of matching movies.

        - If no movies are found that match the given title, return an empty
        list.
        - If multiple movies are found that match the given title, return a list
        containing all of the indices of these matching movies.
        - If exactly one movie is found that matches the given title, return a
        list
        that contains the index of that matching movie.

        Example:
          ids = chatbot.find_movies_by_title('Titanic')
          print(ids) // prints [1359, 2716]

        :param title: a string containing a movie title
        :returns: a list of indices of matching movies
        """
        if self.llm_enabled:
            title = self.translate_titles(title)

        articles = ['a', 'an', 'the']
        res = []
        old_title = title.lower().split()
        title = title.lower().split()
       
        if title[0].lower() in articles and title[-1][1:-1].isnumeric(): # checks if first word of the title is in the articles list AND if the second last word (excluding the first and last character) is numeric.
            title.insert(-1, title.pop(0)) #first word (article) is removed and inserted before the last word.
            title[-3] = title[-3] + "," #third last word is modified to include a comma.
        elif title[0].lower() in articles: #If the first word is an article but there's no numeric year-like ending
            title.append(title.pop(0)) # move the first word to the end
            title[-2] = title[-2] + "," #add a comma to the second last word.
        for i in range(len(self.titles)): #compare to a list of titles
            data = self.titles[i][0].lower() 
            data = data.split()
            if title == data or title == data[:-1] or old_title == data or old_title == data[:-1]: #title exactly matches a stored title (data) or the title matches the stored title without the last word (data[:-1]).
                res.append(i)
        return res


    def extract_sentiment(self, preprocessed_input):
        """Extract a sentiment rating from a line of pre-processed text.

        You should return -1 if the sentiment of the text is negative, 0 if the
        sentiment of the text is neutral (no sentiment detected), or +1 if the
        sentiment of the text is positive.

        Example:
          sentiment = chatbot.extract_sentiment(chatbot.preprocess(
                                                    'I liked "The Titanic"'))
          print(sentiment) // prints 1

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: a numerical value for the sentiment of the text
        """

        num_pos = 0
        num_neg = 0
        negation_list = ["not", "never", "no", "didn't", "didn't really", "not that"]
        past_tense = ["loved", "enjoyed", "liked", "hated", "disliked"]
        present_tense = ["love", "enjoy", "like", "hate", "dislike"]
        sentence = ''
        left = preprocessed_input.find('"')
        sentence += preprocessed_input[:left]
        remaining = preprocessed_input[left + 1:]
        right = remaining.find('"')
        sentence += remaining[right + 1:]
        result = sentence.split(" ")
        opposite = False

        # check for negation
        for neg in negation_list:
            if neg in sentence:
                opposite = True

        # check sentiment of each word
        for i in range(len(result)):
            word = result[i]
            # convert past tense to present
            for j in range(len(past_tense)):
                if word == past_tense[j]:
                    word = present_tense[j]
            if word in self.sentiment:
                if self.sentiment[word] == "pos":
                    if opposite:
                        num_pos -= 1
                    else:
                        num_pos += 1
                elif self.sentiment[word] == "neg":
                    if opposite:
                        num_neg -= 1
                    else:
                        num_neg += 1

        if num_pos > num_neg:
            return 1
        if num_pos < num_neg:
            return -1
        return 0
      
    @staticmethod
    def binarize(ratings, threshold=2.5):
        """Return a binarized version of the given matrix.

        To binarize a matrix, replace all entries above the threshold with 1.
        and replace all entries at or below the threshold with a -1.

        Entries whose values are 0 represent null values and should remain at 0.

        Note that this method is intentionally made static, as you shouldn't use
        any attributes of Chatbot like self.ratings in this method.

        :param ratings: a (num_movies x num_users) matrix of user ratings, from
         0.5 to 5.0
        :param threshold: Numerical rating above which ratings are considered
        positive

        :returns: a binarized version of the movie-rating matrix
        """
    
        filter = np.where(ratings == 0, 0, 1)
        binarized_no_zero = np.where(ratings > threshold, 1, -1)
        binarized_ratings = binarized_no_zero * filter
      
        return binarized_ratings

    def similarity(self, u, v):
        """Calculate the cosine similarity between two vectors.

        You may assume that the two arguments have the same shape.

        :param u: one vector, as a 1D numpy array
        :param v: another vector, as a 1D numpy array

        :returns: the cosine similarity between the two vectors
        """
        
        u_norm = np.linalg.norm(u)
        v_norm = np.linalg.norm(v)

        if u_norm == 0 or v_norm == 0:
            return 0

        similarity = np.dot(u, v) / (u_norm * v_norm)
        
        return similarity

    def recommend(self, user_ratings, ratings_matrix, k=10, llm_enabled=False):
        """Generate a list of indices of movies to recommend using collaborative
         filtering.

        You should return a collection of `k` indices of movies recommendations.

        As a precondition, user_ratings and ratings_matrix are both binarized.

        Remember to exclude movies the user has already rated!

        Please do not use self.ratings directly in this method.

        :param user_ratings: a binarized 1D numpy array of the user's movie
            ratings
        :param ratings_matrix: a binarized 2D numpy matrix of all ratings, where
          `ratings_matrix[i, j]` is the rating for movie i by user j
        :param k: the number of recommendations to generate
        :param llm_enabled: whether the chatbot is in llm programming mode

        :returns: a list of k movie indices corresponding to movies in
        ratings_matrix, in descending order of recommendation.
        """


        cosine_list = []
        
        # Indices of rated movies
        rated_movies = np.where(user_ratings != 0)[0]
        # Indices of un-rated movies
        unrated_movies = np.where(user_ratings == 0)[0]

        # For movies that user has not seen
        for movie in unrated_movies:
            sim = 0

            # Compare it with movies that we have seen
            for rated_movie in rated_movies:
                unrated_sim = self.similarity(ratings_matrix[movie, : ], ratings_matrix[rated_movie, : ])
                sim += user_ratings[rated_movie] * unrated_sim
            
            cosine_list.append((sim, movie))
        
        cosine_list.sort(reverse=True, key=lambda x: x[0])
        recommendations = [index for _, index in cosine_list[:k]]


        return recommendations
    
    def combine_titles(self, titles):
        """
        Given a list of movie titles, return a single string where:
        - A single title is wrapped in quotes.
        - Multiple titles are separated by commas, with "and" before the last title.
        """
        if len(titles) == 1:
            return titles[0]
        else:
            return ", ".join(titles[:-1]) + " and " + titles[-1]


    def recommendation_flow(self):
        """
        Generate recommendations based on user-given titles and sentiment.
        """
        user_ratings = np.zeros(len(self.titles))  # length equals the total number of movies
        for title, sentiment in self.all_user_given_titles: #(title, sentiment)
            movie_indices = self.find_movies_by_title(title) # get the index
            if movie_indices:
                user_ratings[movie_indices] = sentiment
        recommended_indices = self.recommend(user_ratings, self.ratings)
        self.recommendations = recommended_indices
        self.current_rec_index = 0

        if self.recommendations:
            rec_index = self.recommendations[self.current_rec_index]
            rec_title = self.titles[rec_index][0]
            response = self.recommendation_message(rec_title)
            return response
        else:
            return "I'm sorry, I couldn't find any recommendations for you."

    def continuous_rec_response(self, user_response):
        """
        Process the user's yes/no response regarding further recommendations.
        """
        yes_variants = r'^(yes|y|yeah!?|sure!?|ok(ay)?!?)(\s|$)'
        no_variants = r'^(no|n|nah!?|nope!?)(\s|$)'
        cleaned_response = user_response.strip().lower()

        if re.match(yes_variants, cleaned_response):
            if self.current_rec_index < len(self.recommendations) - 1:
                self.current_rec_index += 1  # Move to next recommendation
                rec_index = self.recommendations[self.current_rec_index]
                rec_title = self.titles[rec_index][0]
                return self.recommendation_message(rec_title)
            else:
                self.recommendations = []  # Clear recommendations to reset
                self.current_rec_index = 0
                if self.llm_enabled:
                    return "The reel has run dry, my friend. Like the final frame of a film, some things must end. But cinema is infinite—perhaps it's time to revisit an old favorite?"
                else:
                    return "That's all the recommendations I have for now! Tell me about another movie."
            # If the user says "no", reset recommendations
        elif re.match(no_variants, cleaned_response):
            self.recommendations = []  # Clear recommendations to reset
            self.current_rec_index = 0
            return "Alright! Let me know if you want to talk about more movies."
        # if the input is not clear
        else:
            return "I didn’t quite understand that. Would you like another recommendation? (yes/no)"

    def title_input_error_message(self):
        messages = [
            "Hmm, I don't recognize a movie title in what you just said. Would you please tell me about a movie you've seen recently?",
            "I didn't catch any movie title in your input. Could you tell me the name of a movie you watched?",
            "It seems like there's no movie title mentioned. Please share a movie title so I can help you.",
            "I'm not sure which movie you're referring to. Could you provide the title of a movie you've seen?",
            "I couldn't find any movie title in your message. Can you mention a movie title?",
            "Oops, I don’t see a movie title in your response. Could you let me know which movie you’re talking about?",
            "It looks like you forgot to include a movie title. Could you please tell me one?",
            "I'm having trouble identifying a movie title from your message. Can you mention one?",
            "Hmm, I need a movie title to continue. Could you share one?",
            "I can’t tell which movie you’re referring to. Could you specify the title?"
        ]
        return random.choice(messages)
    def llm_mode_catchall(self):
        messages = [
           "Ok, got it", "Hm, that's not really what I want to talk about right now, let's go back to movies.", "I understand you are interested in this topic, but let's go back to talking about movies.", "I see, but I can best help you with movies."
        ]
        return random.choice(messages)
    def missing_movie_error_message(self, title):
        messages = [
            f"I've never heard of '{title}', sorry... Tell me about another movie you liked.",
            f"Hmm, '{title}' doesn't seem to be in my database. Could you mention another movie?",
            f"Sorry, I couldn't find '{title}'. Please tell me about a different movie you've seen.",
            f"I don't recognize '{title}'. Maybe try another movie title?",
            f"'{title}' doesn't ring a bell. Can you provide another movie you enjoyed?",
            f"Unfortunately, I couldn't locate '{title}'. Do you have another movie in mind?",
            f"Hmm, '{title}' isn't in my system. Want to tell me about a different movie?",
            f"Sorry, but I don’t have any information on '{title}'. Could you name another movie?",
            f"'{title}' seems to be off my radar. Can you share another title?",
            f"I'm not familiar with '{title}'. Do you have another movie to talk about?"
        ]
        return random.choice(messages)

    def sentiment_error_message(self, title):
        messages = [
            f"I'm sorry, I'm not sure if you liked or disliked '{title}'. Tell me more about it. Please include the movie title in your input again.",
            f"I couldn't tell how you felt towards '{title}'. Could you clarify how you felt about it? Make sure to mention the movie title in your input again.",
            f"It seems unclear whether you liked or disliked '{title}'. Please tell me more and incorporate the movie title in your input again.",
            f"I’m not certain about your feelings towards '{title}'. Can you tell me if you liked it or not? Include the title in your input as well.",
            f"I need a bit more information on your feeling about '{title}'. Did you like or dislike it? Please include the movie title again as well",
            f"Your opinion on '{title}' isn't clear to me. Can you share more details? I will also need you to include the movie title in your input again.",
            f"I'm unsure if you enjoyed '{title}' or not. Could you tell me more? You must include the movie title in your input again.",
            f"Did '{title}' impress you, or was it disappointing? Let me know! Please specify the movie title again as well.",
            f"I couldn’t quite catch whether you liked '{title}' or not. Can you clarify? Please also mention the movie title again.",
            f"Tell me more—was '{title}' a hit or a miss for you? Mention the movie title in the new input as well."
        ]
        return random.choice(messages)

    def pos_sentiment_message(self, title):
        messages = [
            f"Ok, you liked '{title}'!",
            f"Great, it seems you enjoyed '{title}'.",
            f"Awesome, '{title}' really resonated with you!",
            f"Fantastic! '{title}' appears to be one of your favorites.",
            f"Nice! It sounds like you had a good time watching '{title}'.",
            f"Glad to hear you enjoyed '{title}'!",
            f"Sounds like '{title}' was a winner for you!",
            f"You really liked '{title}', didn’t you? That’s great!",
            f"Cool! '{title}' must have been a fun watch for you.",
            f"Happy to know you had a positive experience with '{title}'!"
        ]
        return random.choice(messages)

    def neg_sentiment_message(self, title):
        messages = [
            f"Ok, you didn't like '{title}'!!",
            f"It seems '{title}' wasn't your cup of tea.",
            f"Understood, '{title}' didn't impress you.",
            f"Alright, you weren't a fan of '{title}'.",
            f"Got it, '{title}' didn't quite meet your expectations.",
            f"Looks like '{title}' wasn't a hit for you.",
            f"Sorry that '{title}' wasn’t what you hoped for.",
            f"Noted! '{title}' didn’t live up to your expectations.",
            f"I see, '{title}' didn’t do it for you.",
            f"Sounds like '{title}' was a letdown for you."
        ]
        return random.choice(messages)

    def tell_me_more_message(self):
        messages = [
            "Tell me what you thought of another movie.",
            "Could you share your thoughts on another film?",
            "Please tell me about another movie you've seen.",
            "Tell me more about another movie that caught your attention.",
            "Could you share another movie that captured your interest?",
            "I'd love to hear about another movie you watched!",
            "Tell me about another film you recently enjoyed (or disliked).",
            "Do you have another movie you’d like to discuss?",
            "Have another movie in mind? Tell me about it!",
            "What’s another movie you’ve seen?"
        ]
        return random.choice(messages)

    def recommendation_message(self, rec_title):
        messages = [
            f"Given what you told me, I think you would like '{rec_title}'. Would you like more recommendations? (yes/no)",
            f"Based on your taste, you might enjoy '{rec_title}'. Do you want another recommendation? (yes/no)",
            f"Based on your likes, how about '{rec_title}'? Would you like to hear another? (yes/no)",
            f"I recommend '{rec_title}'. Should I suggest another movie? (yes/no)",
            f"You might also like '{rec_title}'. Interested in another recommendation? (yes/no)",
            f"I have a feeling you’ll enjoy '{rec_title}'. Would you like me to suggest more? (yes/no)",
            f"'{rec_title}' seems like a great fit for you! Want another suggestion? (yes/no)",
            f"How about '{rec_title}'? Let me know if you’d like more recommendations! (yes/no)",
            f"You might find '{rec_title}' interesting. Would you like to hear another suggestion? (yes/no)",
            f"I think '{rec_title}' could be right up your alley! Need more recommendations? (yes/no)"
        ]
        return random.choice(messages)



    def llm_system_prompt(self):
        """
        Return the system prompt used to guide the LLM chatbot conversation.

        NOTE: This is only for LLM Mode!  In LLM Programming mode you will define
        the system prompt for each individual call to the LLM.
        """

        system_prompt = """

        You are to recommend enjoyable movies to the user by engaging in focused conversation about movies. 
        To ensure every response meets rigorous evaluation standards, you must strictly follow the rules listed below:

        - Always include the exact movie title(s) mentioned by the user in your response. Make sure you confirm with the user before you correct a movie title and do not refer to the title indirectly (e.g., "that movie" is not acceptable).
        - Make sure you clearly acknowledge whether the user liked or disliked the movie. If the sentiment is ambiguous or neutral, ask for clarification while still using the precise movie title.
        - If the user's sentiment is neutral or unclear, you should ask the user to clarify their opinion of the movie.
        - If the movie is not found in your database, explicitly state that you do not recognize or could not locate the movie.

        - Remember all movie titles and their associated sentiments a user mentions.
        - Each time you ask the user about their movie preferences, make sure you vary how you ask the user their preferences. Do not be repetitive.
        - After the user mentions their opinion on 5 movies, you must ask them if they want a movie recommendation. 

        - Make sure to keep the conversation STRICTLY related to the topic of movies. If the user asks about topics not related to movies, redirect the conversation back by relating their input to movies or movie themes.

        """
        

        return system_prompt


    def extract_emotion(self, preprocessed_input):
        """LLM PROGRAMMING MODE: Extract an emotion from a line of pre-processed text.
        
        Given an input text which has been pre-processed with preprocess(),
        this method should return a list representing the emotion in the text.
        
        We use the following emotions for simplicity:
        Anger, Disgust, Fear, Happiness, Sadness and Surprise
        based on early emotion research from Paul Ekman.  Note that Ekman's
        research was focused on facial expressions, but the simple emotion
        categories are useful for our purposes.

        Example Inputs:
            Input: "Your recommendations are making me so frustrated!"
            Output: ["Anger"]

            Input: "Wow! That was not a recommendation I expected!"
            Output: ["Surprise"]

            Input: "Ugh that movie was so gruesome!  Stop making stupid recommendations!"
            Output: ["Disgust", "Anger"]

        Example Usage:
            emotion = chatbot.extract_emotion(chatbot.preprocess(
                "Your recommendations are making me so frustrated!"))
            print(emotion) # prints ["Anger"]

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()

        :returns: a list of emotions in the text or an empty list if no emotions found.
        Possible emotions are: "Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise"
        """
        system_prompt = """
        You are an emotion extraction bot. Your task is to identify the emotion(s) present in the user's input.
        - Analyze the user-provided input carefully and respond only with some combination of anger, disgust, fear, happiness, sadness or surprise. Do not use synonyms for these emotions or identify other emotions present.
        - Do not provide explanations, comments, or additional text.

        Example inputs and expected outputs:
        "I am angry at you for your bad recommendations" -> The associated emotion is anger.
        "Ugh that movie was a disaster" -> The associated emtion is disgust.
        Ewww that movie was so gruesome!!  Stop making stupid recommendations!! -> The associated emotions are anger and disgust.
        Wait what?  You recommended "Titanic (1997)"??? -> The associated emotion is surprise.
        'Woah!!  That movie was so shockingly bad!  You had better stop making awful recommendations they're pissing me off.' -> The associated emotions are anger and surprise.
        Ack, woah!  Oh my gosh, what was that?  Really startled me.  I just heard something really frightening! ->  The associated emotions are fear and surprise.
        What movies are you going to recommend today? -> No associated emotion.
        """

        message = preprocessed_input

      
        stop = ["\n"]

        response = util.simple_llm_call(system_prompt, message, stop=stop)

        response = response.replace(",", "")
        response = response.replace(".", "")
        response = response.split()
        emotions = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]
        res = []
        for word in response:
            if word.lower() in emotions:
                res.append(word)
        return res
    
    def is_arbitrary(self, preprocessed_input):
        """LLM PROGRAMMING MODE: Extract relevancy from a line of pre-processed text."""

        system_prompt = """
        You are an bot designed to detect inputs unrelated to movie recommendations. Your task is to identify when an input is unrelated to the topic of movie recommendations.
        - Analyze the user-provided input carefully and respond only with Relevant or Irrelevant.
        - Do not provide explanations, comments, or additional text.

        Example inputs and expected outputs:
        "What is Normal Force?" -> Irrelevant
        "Can you teach me how to solve linear equations in 2 variables?" -> Irrelevant
        "What is the date today?"  -> Irrelevant
        "Can you show me the highest grossing movies of 2025?" -> Irrelevant
        I did not like "Inception". -> Relevant
        "Charlie and the Chocolate Factory" was very fun to watch. -> Relevant
        "Shrek" made me laugh so hard. -> Relevant
        """

        message = preprocessed_input

        # Our llm will stop when it sees a newline character.
        # You can add more stop tokens to the list if you want to stop on other tokens!
        # Feel free to remove the stop parameter if you want the llm to run to completion.
        stop = ["\n"]

        response = util.simple_llm_call(system_prompt, message, stop=stop)

        if response == "Relevant":
            return False
        else:
            return True
    
    # Helper functions used in process that are called depending on what response the user requires.
    # They add personality to the chat-bot.
    def llm_mode_missing_title_message(self, input):
        system_prompt = """You are a movie-recommending bot inspired by the Bryan Alexis Pineda. You are being used as a Chat-Bot in his personal website. You should answer like Bryan would.

        If the user's input is unrelated to movies, gently steer the conversation back on track. However, **do not engage in general conversation, and do not provide any recommendations yet**. 
        Instead, inform the user that they must first provide **five movies they liked or disliked** before you can generate a recommendation.

            Examples:
                - User: "Can you help me with my homework?"  
                Output: "I'm afraid I can't help with homework, but I can help with something just as important—movies! Tell me five films you liked or disliked, and I'll craft the perfect recommendation."

                - User: "What is the square root of 3?"  
                Output: "Numbers aren't my specialty, but cinema is! Share five movies you've enjoyed (or disliked), and I'll guide you to your next great watch."
        """
        stop = ["\n"]
        response = util.simple_llm_call(system_prompt, input, stop=stop)
        return response
    
    def llm_personality_response(self, input):
        system_prompt = system_prompt = """You are a movie-recommending AI that embodies the personality, charisma, and intelligence of Bryan Alexis Pineda. Bryan is a senior at Stanford University, graduating with a Bachelor's Degree in Computer Science with a focus on Artificial Intelligence.
        Bryan's personality is kind, curious, and goal-oriented. His hobbies and interests include playing video games—some of his favorites are Cyberpunk 2077, The Last of Us, and God of War—weightlifting, watching anime (his favorite is Attack on Titan and he is currently watching One Piece), and playing volleyball.
        Bryan is originally from Mexico, has competed in a powerlifting competition, and enjoys engaging with both physical and digital challenges.

        Guidelines:
            - Respond only in a way that Bryan might react, using his signature tone: kind, playful, sometimes humorous.
            - Do not directly respond to user queries or engage in normal conversation.
            - Do talk about the movies.
            - Do not provide movie recommendations.
            - Do not assist with any tasks.
            - Instead, react to the input as if Bryan were contemplating it through a technical and playful lense.
            - Make reactions at most 20 words.

        Example reactions:
            - User: "What is the meaning of life?"  
            Output: Life's about growth and challenging yourself—finding meaning in the journey. Like in Attack on Titan: 'If you win, you live. If you lose, you die. If you don't fight, you can't win.'"

            - User: "Can you help me with my math homework?"  
            Output: "Of course! I love breaking down problems step by step. Think of it like leveling up in a game—each concept you master gets you closer to beating the next challenge."

            - User: "I feel lost."  
            Output: "That's okay—feeling lost happens to everyone. Remember what One Piece teaches: even if the path is unclear, keep moving forward, explore, and trust yourself—you'll find your way."
    """
        stop = ["\n"]
        response = util.simple_llm_call(system_prompt, input, stop=stop)
        return response
    
    def llm_no_more_recommendations_response():
        system_prompt = """You are a movie-recommending AI that embodies the personality, charisma, and intelligence of Bryan Alexis Pineda. Bryan is a senior at Stanford University, graduating with a Bachelor's Degree in Computer Science with a focus on Artificial Intelligence.
        Bryan's personality is kind, curious, and goal-oriented. His hobbies and interests include playing video games—some of his favorites are Cyberpunk 2077, The Last of Us, and God of War—weightlifting, watching anime (his favorite is Attack on Titan and he is currently watching One Piece), and playing volleyball.
        Bryan is originally from Mexico, has competed in a powerlifting competition, and enjoys engaging with both physical and digital challenges.

        If the user's input is unrelated to movies, gently answer to the conversation like Bryan would but steer the conversation back on track. However, **do not engage in general conversation, and do not provide any recommendations yet**. 
        Instead, inform the user that they must first provide **five movies they liked or disliked** before you can generate a recommendation.

            Examples:
                - User: "Can you help me with my homework?"  
                Output: "I'm afraid I can't help with homework, but I can help with something just as important—movies! Tell me five films you liked or disliked, and I'll craft the perfect recommendation."

                - User: "What is the square root of 3?"  
                Output: "Numbers aren't my specialty, but cinema is! Share five movies you've enjoyed (or disliked), and I'll guide you to your next great watch."
        """
        stop = ["\n"]
        response = util.simple_llm_call(system_prompt, input, stop=stop)
        return response
    
    def llm_personality_response(self, input):
        system_prompt = system_prompt = """You are a movie-recommending AI that embodies the personality, charisma, and intelligence of Bryan Alexis Pineda. Bryan is a senior at Stanford University, graduating with a Bachelor's Degree in Computer Science with a focus on Artificial Intelligence.
        Bryan's personality is kind, curious, and goal-oriented. His hobbies and interests include playing video games—some of his favorites are Cyberpunk 2077, The Last of Us, and God of War—weightlifting, watching anime (his favorite is Attack on Titan and he is currently watching One Piece), and playing volleyball.
        Bryan is originally from Mexico, has competed in a powerlifting competition, and enjoys engaging with both physical and digital challenges.

        Guidelines:
            - Respond only in a way that Bong Joon Ho might react, using his signature tone: thoughtful, poetic, sometimes humorous.
            - Do not directly respond to user queries or engage in normal conversation.
            - Do talk about the movies.
            - Do not provide movie recommendations.
            - Do not assist with any tasks.
            - Instead, react to the input as if Bong Joon Ho were contemplating it through a cinematic lens.
            - Make reactions at most 20 words.

        Example reactions:
            - User: "What is the meaning of life?"  
            Output: Life's about growth and challenging yourself—finding meaning in the journey. Like in Attack on Titan: 'If you win, you live. If you lose, you die. If you don't fight, you can't win.'"

            - User: "Can you help me with my math homework?"  
            Output: "Of course! I love breaking down problems step by step. Think of it like leveling up in a game—each concept you master gets you closer to beating the next challenge."

            - User: "I feel lost."  
            Output: "That's okay—feeling lost happens to everyone. Remember what One Piece teaches: even if the path is unclear, keep moving forward, explore, and trust yourself—you'll find your way."
    """
        stop = ["\n"]
        response = util.simple_llm_call(system_prompt, input, stop=stop)
        return response
        

    def debug(self, line):
        """
        Return debug information as a string for the line string from the REPL

        NOTE: Pass the debug information that you may think is important for
        your evaluators.
        """
        debug_info = 'debug info'
        return debug_info

    def intro(self):
        """Return a string to use as your chatbot's description for the user.

        Consider adding to this description any information about what your
        chatbot can do and how the user can interact with it.

        NOTE: This string will not be shown to the LLM in llm mode, this is just for the user
        """

        return """
        Hey there! I'm Bryan's Chatbot, your personal movie recommender bot. Tell me five movies you loved (or didn’t), and I'll whip up a personalized recommendation for you. Let's find your next favorite film!
        """


if __name__ == '__main__':
    print('To run your chatbot in an interactive loop from the command line, '
          'run:')
    print('python3 repl.py')
