'''
This file contains for function on feature extraction.
Create some feature from the preprocessed data, before that, clean data.
'''


import string, itertools, generate_candidate, math, re
import pandas as pd
from pandas import DataFrame as df
from itertools import chain
from nltk import word_tokenize, pos_tag, pos_tag_sents
from nltk.stem.porter import *
from nltk.chunk.regexp import RegexpParser
from nltk.chunk import tree2conlltags
from itertools import chain
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
from pytopicrank import TopicRank
from nltk.stem.porter import *


#using porter stemmer
stemmer = PorterStemmer()


def clean_corpus(corpus):

	'''
	Merge and filter some tokens from corpus and stem it.
	'''
	
	clean = []
	for doc in corpus:
		cleaned_words = " ".join([word for word in word_tokenize(doc.lower()) if re.search(r"\b[A-Za-z-]+\b", word) and len(word) > 2])
		stemmed_words = [stemmer.stem(word) for word in cleaned_words.split()]
		clean.append(" ".join([word for word in stemmed_words]))
	return clean
	
	
def calculate_tf(section, vocab, type):

	'''
	calculate_tf(section = name of data, 
		vocab = (name of variable for noun phrase, 'None' for ngram), 
		type = ('ngram' for ngram, and 'np' for noun phrase))
	
	noun phrase and ngram have different function because for noun phrase,
		vocabulary must be feed into the system.
		
	other settings are similar like calculate tf-idf.
	'''
	
	#eliminate ngram which starts or ends from stopwords
	class NewCountVectorizer(CountVectorizer):
		def _word_ngrams(self, tokens, stop_words = None):
			# First get tokens without stop words
			tokens  =  super(CountVectorizer, self)._word_ngrams(tokens, None)
			if stop_words is not None:
				new_tokens = []
				for token in tokens:
					split_words  =  token.split(' ')
					# Only check the first and last word for stop words
					if len(token) > 2 and split_words[0] not in stop_words and split_words[-1] not in stop_words:
						#stem every word in token
						if len(split_words) == 1 and len(split_words[0]) > 2:
							new_tokens.append(stemmer.stem(token))
						#eliminate candidate that contains ' , because 's, this noise must be deleted
						elif len(split_words) == 2 and split_words[-1] == "'":
							del(token)
						#delete candidate (2 tokens) that contains below than 3 characters
						elif len(split_words[0]) < 3 and len(split_words[1]) < 3:
							del(token)
						#removing 's in candidate
						elif split_words[1] == "'" and split_words[2] == "s":
							new_tokens.append(stemmer.stem(split_words[0]) + split_words[1] + split_words[2])
						else:
							new_tokens.append(' '.join(list(stemmer.stem(word) for word in word_tokenize(token))))
				return new_tokens
			return tokens
    
	class StemmedCountVectorizer(CountVectorizer):
	
		'''
		This functions is for extracting noun phrase candidate
		'''
		
		def build_tokenizer(self):
			tokenizer = super(CountVectorizer, self).build_tokenizer()
			return lambda doc: (stemmer.stem(token) for token in tokenizer(doc) if token not in np_stop_words)

	#initialise stop words and only allow certain stop words in noun phrase
	ngram_stop_words = text.ENGLISH_STOP_WORDS
	np_stop_words = set(text.ENGLISH_STOP_WORDS)
	s = ['of','in','on','for']
	np_stop_words = np_stop_words.difference(s)
    
	'''
	In term of stop words, because in ngram candidates, stop words in the middle is allowed,
	on nounphrase, some stop words that are more likely to be keywords, are included
	'''
	
	def get_values(tf, section):
		matrix = tf.fit_transform(section)
		feature_names = tf.get_feature_names()

		tf_values = []
		for doc in range(0,len(section)):
			feature_index = matrix[doc,:].nonzero()[1]
			tf_doc = zip(feature_index, [matrix[doc, x] for x in feature_index])
			names_tf = [(w, s) for w, s in [(feature_names[i], s) for (i, s) in tf_doc]]
			tf_values.append(names_tf)
		return tf_values
		
	#regular expression is used to extract only words in text, hypen is acceptable between the words
	if type == 'ngram':
		tf = NewCountVectorizer(ngram_range = (1,5), stop_words = ngram_stop_words, token_pattern = r"(?u)\b[A-Za-z-]+\b")
		tf_values = get_values(tf, section)
		return tf_values
	elif type == 'np':
		tf = StemmedCountVectorizer(ngram_range = (1,5), stop_words = np_stop_words, vocabulary = vocab, token_pattern = r"(?u)\b[A-Za-z-]+\b")
		tf_values = get_values(tf, section)
		return tf_values
	
def create_supervised_list(train_label, tf_corpus):

	'''
	create_supervised_list(train_label = name of train_label,
							tf_corpus = name of train tf corpus)
	create a dictionary for value of supervised keyphraseness from training data.
	'''

	#select distinct keyphrase in label
	train_label = set(list(chain.from_iterable(train_label)))
	
	#flatten tf_corpus
	tf_corpus = list(chain.from_iterable(tf_corpus))
	
	#merge the value from same keyphrase that found in label
	dict_tf_corpus = dict()
	for key, val in tf_corpus:
		if key in train_label:
			if key in dict_tf_corpus:
				dict_tf_corpus[key] += val
			else:
				dict_tf_corpus[key] = val
			
	#map keyphrase, value as tuple in array
	supervised_corpus = [(key, value) for key, value in dict_tf_corpus.items()]
	return dict_tf_corpus	

	
def create_features(corpus, candidates, label, supervised_keyphraseness, tf_corpus, topic_rank, name, n_keyphrase):
    
	'''
	create_features(
					corpus = name of file name,
					candidates = name of file name,
					label = name of file name ('train_label' for training, 'test_label' for testing),
					supervised_keyphraseness = name of file of supervised_keyphraseness
					tf_corpus = name of file name,
					topic_rank = name of file name,
					name = '' string for the name of the result from this extraction,
					n_keyphrase = number of keyphrase)
					
	This function create 10 features.
	The features are grouped into some category based on their same needs from the file/variable
	'''
		
	def feature_candidate(supervised_keyphraseness, candidates, corpus, topic_rank, n_keyphrase):
	
		'''
		feature_candidate(supervised_keyphraseness = variable name of supervised keyphraseness dictionary, 
						candidates = variable name of candidates,
						corpus = variable name of corpus,
						topic_rank = variable name of topic rank,
						n_keyphrase = number of keyphrase 
						)
		'length', 'supervised_key', 'distance', 'back_distance',
		'spread', 'topic_rank',
		this function creates 6 features
		1. length : how many word in the candidate
		2. supervised key : how many time the candidate occurred as a keyphrase in training data
		3. distance : how many preceding words of the candidate / total words in a document
		4. back distance : how many following words of the candidate / total words in a document
		5. spread : how many word between first and last occurrence
		6. topic rank : binary number of occurrence the candidate in topic rank keywords
		'''
		
		feature = []
		
		#merge all topic rank keywords based on how many number of keyphrase
		topic_rank_keyphrase = [list(chain.from_iterable(n_doc[:n_keyphrase])) for n_doc in topic_rank] 
		
		#clean corpus
		cleaned_corpus = clean_corpus(corpus)
		
		for n_doc in range(len(candidates)):
			doc = []
			matches = []
			
			#count how many words in a document
			corpus_words = len(cleaned_corpus[n_doc].split(" "))

			for n_cand in range(len(candidates[n_doc])):
				
				#feature length of candidate
				length = len(candidates[n_doc][n_cand][0].split())
				
				
				#feature of supervised keyphraseness
				#store candidate which found in supervised keyphraseness list
				temp = [value for term, value in supervised_keyphraseness.items() if candidates[n_doc][n_cand][0] == term]
				
				#if dont appear, put 0
				if len(temp) == 0:
					supervised_value = 0
				#if candidates appears, put the tf in training
				else:
					supervised_value = temp[0]
					
					
				#position feature
				#find the position of first and last index in document
				first_index = cleaned_corpus[n_doc].lower().find(candidates[n_doc][n_cand][0])
				last_index = cleaned_corpus[n_doc].lower().rfind(candidates[n_doc][n_cand][0])
				
				#calculate how many preceding and following words
				preceding_words = len(cleaned_corpus[n_doc][:first_index].split(" ")) - 1
				following_words = len(cleaned_corpus[n_doc][:last_index].split(" ")) - 1
				
				#calculate distance, back distance and spread
				distance = float("{0:.6F}".format(preceding_words/corpus_words))
				back_distance = float("{0:.6F}".format(following_words/corpus_words))
				spread = len(cleaned_corpus[n_doc][first_index:last_index].split(" ")) - 1
				
				
				#topic rank feature
				for topic in topic_rank_keyphrase[n_doc]:
				
					#match the candidates which contain with topic rank keyword, 
					#but only longer to the left from topic rank keyword
					if re.findall(r'.*'+topic+'$', candidates[n_doc][n_cand][0]):
						matches.append(candidates[n_doc][n_cand][0])
				if candidates[n_doc][n_cand][0] in matches:
					topic_rank = 1
				else:
					topic_rank = 0
				
				doc.append((length, supervised_value, distance, back_distance, spread, topic_rank))
			feature.append(doc)
		return feature
      
	  
	def feature_frequency(tf_corpus, candidates): 
		
		'''
		feature_frequency(tf_corpus = variable name of tf corpus,
						candidates = variable name of candidate
						)
		create 3 features: tf, GDC and DPM-index.
		tf = term frequency candidate per document
		DPM-index = measure the phrase is the real keyphrase or not, based on the subterm's occurrences
		GDC = evaluate degree of assosiation between the phrase, Is it only coincide, or the phrase is really really
		'''
		
		frequency = []
		for n_doc in tf_corpus:
			doc = {}
			
			#store all candidate per document
			cand_perdoc = [x[0] for x in n_doc]

			for n_cand, value in n_doc:
			
				#feature term frequency
				term_tf = value
				
				
				#feature DPM-index
				matches = []

				#find the longer phrase from the candidate, but the candidate become the last word in the phrase
				for n_cand2 in cand_perdoc:
					if re.findall(r'.*'+n_cand+'$', n_cand2):
						matches.append((n_cand2))
						
				#only store the occurrence of superterm, excluding the candidate 
				tf_superterm_values = [x[1] for x in n_doc if x[0] in matches and x[0] != n_cand]
				if len(tf_superterm_values) == 0:
					tf_superterm_values = [0]
					
				#calculate the DPM index        
				dpm = float("{0:.5F}".format(1 - max (s_tf / term_tf for s_tf in tf_superterm_values)))
            
			
				#feature GDC
				#split the candidate into some tokens
				length_term = len(n_cand.split(" ")) 
				
				#if candidate only consist from one word, calculate GDC with 0.10 from value.
				#otherwise, it will be the same with tf from its candidate
				if length_term == 1:
					gdc = float("{0:.5F}".format( 0.1 * ( length_term * math.log10 (term_tf) * term_tf ) / term_tf )) 
				
				#if candidate consists more than one words
				else:
				
					#find the value of total occurrences from all the unigram that construct the phrase 
					matched_word = [word for word in [word for word in n_cand.split(" ")] 
                            if word in [value[0] for value in n_doc]]
					freq_word = sum([value[1] for value in n_doc if value[0] in matched_word])
					gdc = float("{0:.5F}".format(( length_term * math.log10 (term_tf) * term_tf ) / freq_word ))
				
				#check if GDC value is inf or -inf
				if gdc == float("-inf") or gdc == float("inf"):
					gdc = 0.0
					
				doc[n_cand] = ((term_tf, dpm, gdc))
			frequency.append(doc) 
		
		#mapping the result with candidate order
		feature = []
		for n_doc in range(len(candidates)):
			doc = []
			for n_cand in candidates[n_doc]:
				for term, value in frequency[n_doc].items():
					if n_cand[0] == term:
						doc.append((value))
			feature.append(doc)
		return feature
    
	#assign the function into variable
	feature1 = feature_candidate(supervised_keyphraseness, candidates, corpus, topic_rank, n_keyphrase)
	feature2 = feature_frequency(tf_corpus, candidates)
	
    #add values of all features into candidate list
	for n_doc in range(0, len(candidates)):
		for n_candidate in range(0, len(candidates[n_doc])):
			candidates[n_doc][n_candidate] = candidates[n_doc][n_candidate] + (feature1[n_doc][n_candidate][0],)    #length  			2
			candidates[n_doc][n_candidate] = candidates[n_doc][n_candidate] + (feature1[n_doc][n_candidate][1],)    #supervised  		3
			candidates[n_doc][n_candidate] = candidates[n_doc][n_candidate] + (feature1[n_doc][n_candidate][2],) 	#distance			4
			candidates[n_doc][n_candidate] = candidates[n_doc][n_candidate] + (feature1[n_doc][n_candidate][3],) 	#back_distance		5
			candidates[n_doc][n_candidate] = candidates[n_doc][n_candidate] + (feature1[n_doc][n_candidate][4],) 	#spread				6
			candidates[n_doc][n_candidate] = candidates[n_doc][n_candidate] + (feature1[n_doc][n_candidate][5],) 	#topicrank			7
			candidates[n_doc][n_candidate] = candidates[n_doc][n_candidate] + (feature2[n_doc][n_candidate][0],) 	#tf      			8
			candidates[n_doc][n_candidate] = candidates[n_doc][n_candidate] + (feature2[n_doc][n_candidate][1],) 	#DPM 				9
			candidates[n_doc][n_candidate] = candidates[n_doc][n_candidate] + (feature2[n_doc][n_candidate][2],) 	#GDC				10

			
    #create example as well as label
	x_data = []
	y_label = []
	
	
	#initialise header for csv
	header = ['candidates', 'tf-idf', 'length', 'supervised_key', 'distance', 'back_distance',
		'spread', 'topic_rank', 'tf',  'DPM-index', 'GDC']

    #merge all features and create label
	for n_doc in range(len(candidates)):
		for n_candidate in range(len(candidates[n_doc])):
			keyphrase_document = list(label[n_doc])
			if candidates[n_doc][n_candidate][0] not in keyphrase_document:
				y_label.append(0)
			else:
				y_label.append(1)            
			x_data.append(list(candidates[n_doc][n_candidate]))
    
	#store the feature and candidates into pandas
	data = df.from_records(x_data, columns = header)
	
	#merge the label into data and store into csv
	data['label'] = pd.Series(y_label).values
	csv = data.to_csv('%s_data.csv' % name, encoding = 'utf-8')
    
	return 'complete to create machine learning data'
	

	

