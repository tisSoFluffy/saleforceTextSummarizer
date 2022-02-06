# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 16:57:18 2022

@author: joshu
"""
# importing libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import pandas as pd
import re

# Input text - to summarize
df = pd.read_csv('corpus_1.csv')

###Clean up file
def clean_file(x: str) -> str:
    '''This function is used to remove all newline characters within the dataframes and all caps words.'''
    #remove newlines
    s = x.replace('\n', '')
    #remove all caps
    s = re.sub("[A-Z]{2,}","", s)
    #remove phone numbers
    s = re.sub("\d+-\d+-\d+-\d{4}", "",s)
    return s

def build_summary(x: str) -> str:
    '''Using Cos Similarity to build summary of webpage'''
    words = word_tokenize(x)
    freqTable = dict()
    for word in words:
    	word = word.lower()
    	if word in stopWords:
    		continue
    	if word in freqTable:
    		freqTable[word] += 1
    	else:
    		freqTable[word] = 1
    
    # Creating a dictionary to keep the score
    # of each sentence
    
    sentences = sent_tokenize(x)
    sentenceValue = dict()
    
    for sentence in sentences:
    	for word, freq in freqTable.items():
    		if word in sentence.lower():
    			if sentence in sentenceValue:
    				sentenceValue[sentence] += freq
    			else:
    				sentenceValue[sentence] = freq
    
    
    
    ###Utilize cos similarity
    
    sumValues = 0
    for sentence in sentenceValue:
    	sumValues += sentenceValue[sentence]
    
    # Average value of a sentence from the original text
    
    average = int(sumValues / len(sentenceValue))
    
    # Storing sentences into our summary.
    summary = ''
    for sentence in sentences:
    	if (sentence in sentenceValue) and (sentenceValue[sentence] > (1.2 * average)):
    		summary += " " + sentence
    return summary

#clean the dataframe text
df['cleaned_text'] = df['1'].apply(lambda x: clean_file(x))

# Tokenizing the text
stopWords = set(stopwords.words("english"))

#build summary of webpage
df['summary'] = df['cleaned_text'].apply(lambda x: build_summary(x))

df.to_csv('summary.csv', index=False)
