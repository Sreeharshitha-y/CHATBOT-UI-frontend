import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from PIL import Image
from io import BytesIO
import pytesseract
import pdf2image
from pdf2image import convert_from_path
import os

# Download NLTK data
nltk.download('punkt')

# Set the path to Poppler binaries
poppler_path = "/usr/bin"  # Use the default path in Colab

# Optionally, set the environment variable
os.environ["PYPDF2_BIN_PATH"] = poppler_path
os.environ["PATH"] += os.pathsep + poppler_path

pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'


# Function to read PDF files that returns text corpus.
def get_pdf_data(user_resp):
    PDF_file = "pdf1.pdf"
    pages = convert_from_path(PDF_file, 500, poppler_path=poppler_path)
    image_counter = 1
    for page in pages:
        filename = "page_" + str(image_counter) + ".jpg"
        page.save(filename, 'JPEG')
        image_counter = image_counter + 1

    filelimit = image_counter - 1
    corpus = ''
    for i in range(1, filelimit + 1):
        filename = "page_" + str(i) + ".jpg"
        text = str(((pytesseract.image_to_string(Image.open(filename)))))
        text = text.replace('-\n', '')
        corpus += text

    sent_tokens = nltk.sent_tokenize(corpus)
    return sent_tokens

# Cleaning the data and making it usable.
def LemNormalize(corpus):
    return nltk.word_tokenize(corpus.lower().translate(str.maketrans('', '', string.punctuation)))

# Greeting Inputs
GREETING_INPUTS = ["hi", "hello", "hola", "greetings", "wassup", "hey"]
# Greeting responses back to the user
GREETING_RESPONSES=["howdy", "hi", "hey", "what's good", "hello", "hey there"]
# Function to return a random greeting response to a user's greeting
def greeting(sentence):
    # If the user's input is a greeting, then return a randomly chosen greeting response
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)
        

# Create a text similarity detection function that matches the user inputs and returns similar sentences.
def response(user_response):
    # The user's response / query
    user_response = user_response.lower()  # Make the response lowercase
    # Set the chatbot response to an empty string
    robo_response = ''
    sent_tokens = get_pdf_data(user_response)
    # Append the user's response to the sentence list
    sent_tokens.append(user_response)
    # Create a TfidfVectorizer Object
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    # Convert the text to a matrix of TF-IDF features
    tfidf = TfidfVec.fit_transform(sent_tokens)
    # Get the measure of similarity (similarity scores)
    vals = cosine_similarity(tfidf[-1], tfidf)
    # Get the index of the most similar text/sentence to the user's response
    try:
        idx = vals.argsort()[0][-2]
    except IndexError:
        robo_response = robo_response + "I apologize, I don't understand."
        return robo_response
    # Reduce the dimensionality of vals
    flat = vals.flatten()
    # Sort the list in ascending order
    flat.sort()
    # Get the most similar score to the user's response
    score = flat[-2]
    # If the variable 'score' is 0 then there is no text similar to the user's response
    if score == 0:
        robo_response = robo_response + "I apologize, I don't understand."
    else:
        robo_response = robo_response + sent_tokens[idx]

    # Remove the user's response from the sentence tokens list
    sent_tokens.remove(user_response)

    return robo_response


# A function that handles the input of the users.
flag = True
print("ChatBot: Hi! I will answer your queries. Please Ask. If you want to exit, type Bye!")
while flag:
    user_response = input()
    user_response = user_response.lower()
    if user_response != 'bye':
        if user_response == 'thanks' or user_response == 'thank you':
            flag = False
            print("ChatBot: You are welcome!")
        else:
            if greeting(user_response) is not None:
                print("ChatBot: " + greeting(user_response))
            else:
                print("ChatBot: " + response(user_response))
    else:
        flag = False
        print("ChatBot: Chat with you later!")



