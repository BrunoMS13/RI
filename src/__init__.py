import xml.etree.ElementTree as ET
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.feature_extraction.text import TfidfVectorizer


def get_info(queries) -> list:
    temp_list = []
    with open(queries, 'r') as queries_reader:
        txt = queries_reader.read()
    root = ET.fromstring(txt)
    for query in root.iter('TOP'):
        q_num = query.find('NUM').text
        q_title = query.find('TITLE').text
        temp_list.append("Query " + q_num + ": " + q_title)
    return temp_list


class Main:
    # Set all queries at root.

    queries = "C:/Users/silva/OneDrive/Ambiente de Trabalho/RI/info_folders/topics-2014_2015-summary.topics"
    documents = "C:/Users/silva/OneDrive/Ambiente de Trabalho/RI/info_folders/topics-2014_2015-description.topics"

    corpus = get_info(queries)

    index = TfidfVectorizer(ngram_range=(1, 2), analyzer='word', stop_words=None)
    index.fit(corpus)

    X = index.transform(corpus)

    query = get_info(documents)
    query_tfidf = index.transform(query)

    doc_scores = 1 - pairwise_distances(X, query_tfidf, metric='cosine')
    print(doc_scores)
