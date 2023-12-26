import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
nltk.download('punkt')
nltk.download('stopwords')

popular_df = pd.read_csv("Popular.csv")

filepath = "Pivot_File_User_Ratings.pkl"

with open(filepath, 'rb') as f:
    Pivot_Table = pd.read_pickle(f)



# Sample list of book names
book_list = popular_df["Book-Title"]  + " By " + popular_df["Book-Author"]


Similarity_Score = cosine_similarity(Pivot_Table)


def preprocess_text(text):    # extract tokens/ keywords
    # Tokenization
    tokens = word_tokenize(text.lower())

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.isalnum() and token not in stop_words]

    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]

    return tokens



def search_books_nlp(query, book_list):   # from tokens extracted, if any token matches then we will return book name
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
        #print("\n\nSearch results: \n\n")
        for result in search_results:
            #print(result)
            foundList.append(result)
            similarList.append(recommend(result))

        similars = np.array(similarList)
        similars = np.ravel(similars)

        foundList = set(foundList)

        similars = set(similars)

        similars = similars - foundList

        similars = list(similars)

        #print("\n\nYou might also like\n\n")
        #for i in similars: print(i)

        return {"Found": foundList, "Recommends":similars}


    else:

        return None, None

aa = FetchAndRecommend("Harry")

a = aa["Found"]
b = aa["Recommends"]

SearchRes = popular_df[popular_df['Book-Title'].isin(a)]
RecommendRes = popular_df[popular_df['Book-Title'].isin(b)]

print(SearchRes.to_numpy())

print(RecommendRes.to_numpy())