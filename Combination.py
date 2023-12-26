from flask import Flask, render_template, request
import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

popular_df = pd.read_csv("Popular.csv")

filepath = "Pivot_File_User_Ratings.pkl"

with open(filepath, 'rb') as f:
    Pivot_Table = pd.read_pickle(f)

# Sample list of book names
book_list = popular_df["Book-Title"] + " By " + popular_df["Book-Author"]

Similarity_Score = cosine_similarity(Pivot_Table)


def preprocess_text(text):  # extract tokens/ keywords
    # Tokenization
    tokens = word_tokenize(text.lower())

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.isalnum() and token not in stop_words]

    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]

    return tokens


def search_books_nlp(query, book_list):  # from tokens extracted, if any token matches then we will return book name
    query_tokens = preprocess_text(query)
    results = []

    for book in book_list:
        book_tokens = preprocess_text(book)
        if any(token in book_tokens for token in query_tokens):
            results.append(book.split(" By ")[0])

    return results


def recommend(Book_Name):  # from returned book names,   recommend similar books
    # Book_Name has to be in the csv file. else we will get error
    # fetch index

    index = np.where(Pivot_Table.index == Book_Name)[0][0]

    Similar_Items = sorted(list(enumerate(Similarity_Score[index])), key=lambda x: x[1], reverse=True)[1:4]

    itemss = []
    for i in Similar_Items:
        itemss.append(Pivot_Table.index[i[0]])  # we will get indexes

    return itemss


def FetchAndRecommend(user_input):
    search_results = search_books_nlp(user_input, book_list)

    similarList = []
    foundList = []

    if search_results:
        # print("\n\nSearch results: \n\n")
        for result in search_results:
            # print(result)
            foundList.append(result)
            similarList.append(recommend(result))

        similars = np.array(similarList)
        similars = np.ravel(similars)

        foundList = set(foundList)

        similars = set(similars)

        similars = similars - foundList

        similars = list(similars)

        # print("\n\nYou might also like\n\n")
        # for i in similars: print(i)

        return foundList, similars


    else:

        return None, None



app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')



@app.route('/recommend_books', methods = ["POST"])
def recommendWebsite():
    user_Given_input = request.form.get("user_input")


    a, b = FetchAndRecommend(user_Given_input)

    try:
        SearchedResultIP = popular_df[popular_df['Book-Title'].isin(a)]
        SearchedResultIP = SearchedResultIP.to_numpy()

    except: SearchedResultIP= []

    try:
        RecommendsIP = popular_df[popular_df['Book-Title'].isin(b)]
        RecommendsIP = RecommendsIP.to_numpy()

    except: RecommendsIP = []

    #print(SearchedResultIP)
    #print(RecommendsIP)

    if len(SearchedResultIP) > 0: ResTr = 1
    else: ResTr = 0

    if len(RecommendsIP) > 0: RecTr = 1
    else: RecTr = 0


    return render_template('index.html',
                           SearchedResult=SearchedResultIP,
                           Recommends=RecommendsIP,
                           SearchedResultTrue = ResTr,
                           RecommendsTrue = RecTr
                           )



if __name__ == "__main__":
    app.run(debug = True)