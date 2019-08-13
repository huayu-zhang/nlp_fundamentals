import re
import spacy
from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score


def trim_news(news):
    starting_char = re.search(r"Lines: \w+", news)
    if starting_char is None:
        return news
    return news[starting_char.end():]


# Topic classification using entity recognition

# News data import

news_train = [trim_news(news) for news in fetch_20newsgroups(subset='train').data]
topic_train = fetch_20newsgroups(subset='train').target
news_test = [trim_news(news) for news in fetch_20newsgroups(subset='test').data]
topic_test = fetch_20newsgroups(subset='test').target


# nlp pipeline with spaCy, using small pre-trained web text model
# Named entities are recognized and saved

nlp = spacy.load('en_core_web_sm')

train_entities = []
train_entities_labels = []

for doc in nlp.pipe(news_train):
    entities = ' '.join([ent.text for ent in doc.ents])
    entity_labels = ' '.join([ent.label_ for ent in doc.ents])
    train_entities.append(entities)
    train_entities_labels.append(entity_labels)

test_entities = []
test_entities_labels = []

for doc in nlp.pipe(news_test):
    entities = ' '.join([ent.text for ent in doc.ents])
    entity_labels = ' '.join([ent.label_ for ent in doc.ents])
    test_entities.append(entities)
    test_entities_labels.append(entity_labels)

# Vectorize list of entities and entity labels
# Naive bayes classifier

nb_pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('nb', MultinomialNB())
])

nb_pipeline.fit(X=train_entities_labels, y=topic_train)
topic_pred = nb_pipeline.predict(X=test_entities_labels)

model_accuracy = accuracy_score(y_pred=topic_pred, y_true=topic_test)

