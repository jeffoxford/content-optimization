from pandas.core.frame import DataFrame
from serpapi import GoogleSearch
import os
import csv
import time
import pandas as pd
import numpy as np
from requests import get
import re 
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from pprint import pprint
from requests import session
import textrazor
import pandas as pd
import streamlit as st
from keybert import KeyBERT
textrazor.api_key = "cbf491222196a84d4bbcf85f575a75ea323c329bb97a2bb280404dac"

app_secret = 'e478f284e8aa736bc21fd8691ae7d08f14680d2e6a1fac7a8d6ad1f51e1b358f'
client = textrazor.TextRazor(extractors=["entities", "topics"])
client.set_cleanup_mode(cleanup_mode='cleanHTML')
client.set_cleanup_return_cleaned(True)
TOP_K_KEYWORDS = 15

from sklearn.feature_extraction.text import TfidfVectorizer

############ tfidf ###############################
def sort_coo(coo_matrix):
    """Sort a dict with highest score"""
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""
    
    #use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []
    
    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    #create a tuples of feature, score
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results
def get_keywords(vectorizer, feature_names, doc):
    """Return top k keywords from a doc using TF-IDF method"""

    #generate tf-idf for the given document
    tf_idf_vector = vectorizer.transform([doc])
    
    #sort the tf-idf vectors by descending order of scores
    sorted_items=sort_coo(tf_idf_vector.tocoo())
    #extract only TOP_K_KEYWORDS
    keywords=extract_topn_from_vector(feature_names,sorted_items,TOP_K_KEYWORDS)
    return keywords
def get_tfidf_vectorizer(corpora):
    vectorizer = TfidfVectorizer(stop_words='english',ngram_range=(1,3), smooth_idf=True, use_idf=True)
    vectorizer.fit_transform(corpora)
    feature_names = vectorizer.get_feature_names()
    return vectorizer, feature_names


def get_keywords_summary(entity_df,threshold):
    url_df = entity_df['url'].unique()
    text_df =  entity_df['text'].unique()

    vectorizer = TfidfVectorizer(analyzer='word', stop_words='english', ngram_range=(1,1),  smooth_idf=True, use_idf=True)
    vectorizer.fit_transform(list(text_df))
    feature_names = vectorizer.get_feature_names_out()

    vectorizer_2 = TfidfVectorizer(analyzer='word',stop_words='english',  ngram_range=(2,2),  smooth_idf=True, use_idf=True)
    vectorizer_2.fit_transform(list(text_df))
    feature_names_2 = vectorizer_2.get_feature_names_out()

    vectorizer_3 = TfidfVectorizer(analyzer='word',stop_words='english',  ngram_range=(3,3),  smooth_idf=True, use_idf=True)
    vectorizer_3.fit_transform(list(text_df))
    feature_names_3 = vectorizer_3.get_feature_names_out()

    keywords_df = pd.DataFrame(columns=['text','keywords'])
    result = []
    corpora = text_df
    for doc in corpora:
        keywords = get_keywords(vectorizer, feature_names, doc)
        keywords_df.loc[len(keywords_df)] = {'text':doc,'keywords':keywords}

        keywords = get_keywords(vectorizer_2, feature_names_2, doc)
        keywords_df.loc[len(keywords_df)] = {'text':doc,'keywords':keywords}
        keywords = get_keywords(vectorizer_3, feature_names_3, doc)
        keywords_df.loc[len(keywords_df)] = {'text':doc,'keywords':keywords}

        


    unique_keys = set()
    for i in range(len(keywords_df)):
        for item in keywords_df.iloc[i]['keywords']:
            unique_keys.add(item)

    keywords_summary_df = pd.DataFrame(columns=['keyword','no_documents','computed_score','average_weight','max'])

    for u_key in unique_keys:
        scores = []
        cnt = 0
        for i in range(len(keywords_df)):
            for keyword in keywords_df.iloc[i]['keywords']:
                if keyword == u_key:
                    cnt = cnt + 1
                    scores.append(keywords_df.iloc[i]['keywords'][u_key])
                    break
        computed_score = np.average(scores) * np.power(cnt,2)
        if computed_score > threshold:
            keywords_summary_df.loc[len(keywords_summary_df)] = {'keyword': u_key, 'no_documents':cnt,'computed_score':computed_score, 'average_weight':np.average(scores),'max': np.max(scores)}
    return keywords_summary_df

def apply_keybert(data_df):
        url_df = data_df.groupby(['url','text'])['url'].unique()
        text_df =  data_df.groupby(['url','text'])['text'].unique()

        kw_extractor = KeyBERT()
        key_df = pd.DataFrame(columns=['url','text','keywords'])
        for j in range(len(url_df)):
            keywords = kw_extractor.extract_keywords(text_df[j], keyphrase_ngram_range=(1,3),stop_words='english',diversity=0.6,top_n=10,use_mmr=True,
            )
            key_df.loc[len(key_df)] = {'url': url_df[j],'text':text_df[j],'keywords':keywords}
        return key_df

def search_keywords(input_text,num_pages):
    entity_df = pd.DataFrame(columns=['url','entity','entity_type','text'])
    gsearch = GoogleSearch({
        "q": input_text, 
        "location": "Austin,TX,Texas,United States",
        "num" : num_pages,
        "api_key": app_secret
    })
    result = gsearch.get_dict()

    final_results= []

    df = pd.DataFrame(columns=['link','title','text'])
    count = 0
    for item in result['organic_results']:
        page_url = item['link']
        title=item['title']
        response = client.analyze_url(item['link'])
        
        
        for entity in response.entities():
            if len(entity.freebase_types) > 0:
                entity_df.loc[len(entity_df)] = {'url': page_url,'title':title, 'entity': entity.id, 'entity_type':str(entity.freebase_types[0]), 'text':response.cleaned_text }
    
        
        
    return entity_df


def main():
    threshold = st.slider('Score threshold (avg_weight * number_of_doc^2):',0.0, 1.0,0.2,0.05)
    no_pages = st.selectbox(
    'Number of pages:',
    
    ('5', '10', '15','20','25','30'))
    user_input = st.text_input('Google Search', 'unlock iphone')

    if  st.button("Search",no_pages) and len(user_input) > 3 :
        entity_df =  search_keywords(user_input,no_pages)
        #st.write(mydf.groupby(['url','entity','entity_type'])['entity'].count().reset_index(
        #        name='Count').sort_values(['url','Count'], ascending=False))

        #corpora = mydf['text'].unique()
        ### apply tfidf on corpora 
        keywords_summary_df = get_keywords_summary(entity_df, threshold)
        st.title('Kewords extraction using TF-IDF')

        tfidf_csv = keywords_summary_df.sort_values('computed_score',ascending=False).to_csv().encode('utf-8')  
        st.write(keywords_summary_df.sort_values('computed_score',ascending=False).head())
        
        st.download_button(
        label="Download full data as CSV",
        data=tfidf_csv,
        file_name='tfidf_keywords.csv',
        mime='text/csv',
        )

    

if __name__ == '__main__':
    main()