import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF, TruncatedSVD, LatentDirichletAllocation

from gensim import matutils

class topic_model():

    def __init__(
        self, 
        data=None, 
        vectorizer=CountVectorizer(), 
        model=None, 
        num_topics=10,
        random_state=None
        ):
        '''
        Vectorize and topic model a given text
        :param data: list of document texts
        :param vectorizer: which vectorizor to use:
        :param model: which model to use: 'lda', 'lsa' or 'nmf'
        '''
        self.data = data
        self.vectorizer = vectorizer
        self.model = model
        self.num_topics = num_topics
        self.random_state = random_state      

    def get_topics(self):
        self.doc_word = self.vectorizer.fit_transform(self.data)
        self.doc_word_df = pd.DataFrame(self.doc_word.toarray(),columns = self.vectorizer.get_feature_names())
        self.feature_names = self.vectorizer.get_feature_names()
        # self.corpus = matutils.Sparse2Corpus(self.doc_word.transpose())
        # self.id2word = dict((v, k) for k, v in self.vectorizer.vocabulary_.items())
        #return self.doc_word, self.doc_word_df, self.feature_names

    #def display_topics(self):
        if self.model == "HOLD lda":
            lda = LatentDirichletAllocation(n_components=self.num_topics)
            lda.fit_transform(self.doc_word)
            for idx, topic in enumerate(lda.components_):
                print ("Topic ", idx, " ".join(self.feature_names[i] for i in topic.argsort()[:-10 - 1:-1]))

        # elif self.model == "corex":
        #     model = corextopic.Corex(n_hidden=self.num_topics, words=self.words, seed=1, max_iter=200)
        #     model.fit(self.doc_word, words=self.words, docs=self.data)
        #     topics = model.get_topics()
        #     print(topics)
        #     for n,topic in enumerate(topics):
        #         topic_words,_ = zip(*topic)
        #     print('{}: '.format(n) + ','.join(topic_words))

        else:
            if self.model == "lsa":
                model = TruncatedSVD(self.num_topics,random_state=self.random_state)
            elif self.model == "nmf":
                model = NMF(self.num_topics,random_state=self.random_state)
            elif self.model == 'lda':
                model = LatentDirichletAllocation(self.num_topics,random_state=self.random_state)

            else: print("Please define a valid model.")

            doc_topic = model.fit_transform(self.doc_word)
            for ix, topic in enumerate(model.components_):
                print("\nTopic ", ix)
                print(", ".join([self.feature_names[i] for i in topic.argsort()[:-16:-1]]))