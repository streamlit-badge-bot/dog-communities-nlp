# Topic Modeling Exploration of Reddit Dog Communities

## Goal
The goal of this project is to explore reddit dog communities using natural language processing and unsupervised learning. Pulling one year's worth of data from the top seven highest-subscribed reddit dog breed communities, I used NLP and topic modeling techniques to identify topics most commonly discussed among subreddit communities, and derived ["doggolingo"](https://www.npr.org/sections/alltechconsidered/2017/04/23/524514526/dogs-are-doggos-an-internet-language-built-around-love-for-the-puppers) terms from the corpus. Finally, I built [an app](https://share.streamlit.io/labb0t/doggolingo-explained/main) to allow users to explore the meaning of different doggolingo terms.

## Methodologies
1. Pulled all 2019 post and comment data from the seven most highly subscribed dog breed subreddits from Googe's BigQuery, in total covering roughly 80K reddit posts.
2. Used SpaCy pipelines to preprocess the text corpus.
3. Ran topic modeling on the corpus using Count Vectorizer, TF-IDF, LSA, NMF, LDA, and CorEx.
4. Used a different a different SpaCy pipeline to pre-process the corpus to derive "doggolingo" words.
5. Built a [doggolingo exploration app](https://share.streamlit.io/labb0t/doggolingo-explained/main) using streamlit.

## Outline of Files
- [all_breeds_topic_modeling notebook](https://github.com/labb0t/dog-communities-nlp/blob/main/all_breeds_topic_modeling.ipynb): data preprocessing and topic modeling.
- [doggolingo notebook](https://github.com/labb0t/dog-communities-nlp/blob/main/doggolingo.ipynb): data processing to derive "doggolingo" and wordcloud generation, as well as data cleaning and preparation for streamlit app.
- [presentation pdf](https://github.com/labb0t/dog-communities-nlp/blob/main/reddit_dogs_presentation.pdf): final presentation for Metis program
- See the [doggolingo explained](https://github.com/labb0t/doggolingo-explained) repo for code and final data for the streamlit app.
