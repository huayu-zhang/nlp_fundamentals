import os
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix


def read_txt(path):
    f = open(path).read()
    return f


def tokenize_rm_stopwords(text, sw):
    words = word_tokenize(text)
    words_rm_stopwords = [w for w in words if w not in sw]
    return words_rm_stopwords


def filter_tag(tagged_words, tag):
    filtered = [w for w, t in tagged_words if t == tag]
    return filtered


def lemmatize(words, lemmatizer):
    lemma = [lemmatizer(word) for word in words]
    return lemma


# Data import
train_files_pos = ['./aclImdb/train/pos/' + file for file in os.listdir('aclImdb/train/pos')]
train_files_neg = ['./aclImdb/train/neg/' + file for file in os.listdir('aclImdb/train/neg')]
test_files_pos = ['./aclImdb/test/pos/' + file for file in os.listdir('aclImdb/test/pos')]
test_files_neg = ['./aclImdb/test/neg/' + file for file in os.listdir('aclImdb/test/neg')]

train_txt = [read_txt(file) for file in train_files_pos + train_files_neg]
train_y = ['positive' for i in range(train_files_pos.__len__())] + ['negative' for i in range(train_files_neg.__len__())]
test_txt = [read_txt(file) for file in test_files_pos + test_files_neg]
test_y = ['positive' for i in range(test_files_pos.__len__())] + ['negative' for i in range(test_files_neg.__len__())]


# Tokenization
set_stop_words = set(stopwords.words('english'))

train_txt_tokenize = [tokenize_rm_stopwords(text=text, sw=set_stop_words) for text in train_txt]
test_txt_tokenize = [tokenize_rm_stopwords(text=text, sw=set_stop_words) for text in test_txt]


# Tagging
train_txt_tag = [pos_tag(tokens) for tokens in train_txt_tokenize]
test_txt_tag = [pos_tag(tokens) for tokens in test_txt_tokenize]


# Filter for JJ (adjectives)
train_txt_filtered = [filter_tag(i, 'JJ') for i in train_txt_tag]
test_txt_filtered = [filter_tag(i, 'JJ') for i in test_txt_tag]


# Lemmatization
wnl = WordNetLemmatizer()
train_txt_lemma = [lemmatize(words=words, lemmatizer=wnl.lemmatize) for words in train_txt_filtered]
test_txt_lemma = [lemmatize(words=words, lemmatizer=wnl.lemmatize) for words in test_txt_filtered]


# Counts and NB model with scikit learn

train_txt_sk = [' '.join(words) for words in train_txt_lemma]
test_txt_sk = [' '.join(words) for words in test_txt_lemma]

text_pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('nb', MultinomialNB())
])

text_pipeline.fit(X=train_txt_sk, y=train_y)
pred = text_pipeline.predict(X=test_txt_sk)

cfm = confusion_matrix(y_true=test_y, y_pred=pred)
print(cfm)
