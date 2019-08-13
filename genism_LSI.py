from sklearn.datasets import fetch_20newsgroups
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import remove_stopwords
from nltk.stem import WordNetLemmatizer
from gensim.corpora.dictionary import Dictionary
from gensim.models import LsiModel
from gensim.models import LdaMulticore


def lemmatize(words, lemmatizer):
    lemma = [lemmatizer(word) for word in words]
    return lemma


def tokenize_lemmatize(text, lemmatizer):
    text_rm = remove_stopwords(text)
    text_token = simple_preprocess(text_rm)
    text_lemma = lemmatize(text_token, lemmatizer)
    return text_lemma


news_train = fetch_20newsgroups(subset='train')


# Tokenization and lemmatization
wnl = WordNetLemmatizer()
news_train_lemma = [tokenize_lemmatize(article, wnl.lemmatize) for article in news_train.data]


# Build a genism corpara structure
dict_train = Dictionary(news_train_lemma)
mmCorpus_train = [dict_train.doc2bow(article) for article in news_train_lemma]

# Latent Semantic Analysis
lsi_train = LsiModel(corpus=mmCorpus_train,
                     num_topics=40,
                     id2word=dict_train
                     )

for i in range(40):
    print('topic' + i.__str__() + ' :')
    print(lsi_train.print_topic(i))


# Latent Dirichlet Allocation

lda_train = LdaMulticore(corpus=mmCorpus_train,
                         num_topics=40,
                         id2word=dict_train,
                         workers=5
                        )

for i in range(40):
    print('topic' + i.__str__() + ' :')
    print(lda_train.print_topic(i))
