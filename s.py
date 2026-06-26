import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    L = []
    for word in text:
        if word.isalnum():
            L.append(word)

    text = L.copy()
    L.clear()

    for word in text:
        if word not in stopwords.words('english') and word not in string.punctuation:
            L.append(word)

    text = L.copy()
    L.clear()

    for word in text:
        L.append(ps.stem(word))

    return " ".join(L)

tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

msg = "won 25 lakh in lucky draw"

msg = transform_text(msg)
print(msg)

vector = tfidf.transform([msg])

print(model.predict(vector))