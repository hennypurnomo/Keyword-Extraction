'''
This file is created to generate candidate.
Because to generate candidate, it incorporate with TFIDFVectorizer, therefore, 
candidate list comprise with tuple of candidate and tf-idf value.
Since the model has two type of candidate, which are ngram and noun phrase, 
on noun phrase, it must use noun phrase vocabulary.
'''


import string, itertools
from nltk import word_tokenize, pos_tag, pos_tag_sents
from nltk.stem.porter import *
from nltk.chunk.regexp import RegexpParser
from nltk.chunk import tree2conlltags
from itertools import chain
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text

stemmer = PorterStemmer() #stemming using Porter

def calculate_tfidf(corpus, vocab, type):

	'''
	calculate_tfidf(corpus = name of corpus, 
					vocabulary = ('None' for ngram, 'name of variable' for nounphrase),
					type = ('ngram' for ngram, 'np' for noun phrase))
				
	Each text in corpus is extracted into ngram / noun phrase (depends on the type).
	On ngram, this system filters unwanted candidate like below than 3 characters,
		while on nounphrase, because its vocabulary has been filtered.
	The candidates from tfidfvectorizer are stemmed.
	Then, the result of vectorizer per document is mapped into list of tuple, 
		each tuple contains ('candidate', value of tfidf).
	'''
	
	
	class NewTfidfVectorizer(TfidfVectorizer):
	
		'''
		eliminate ngram which starts or ends from stopwords
		from https://stackoverflow.com/questions/49746555/sklearn-tfidfvectorizer-generate-custom
		-ngrams-by-not-removing-stopword-in-the/49775000#49775000
		This function is for extracting ngram candidate.
		'''
	
		def _word_ngrams(self, tokens, stop_words = None):
            # First get tokens without stop words
			tokens  =  super(TfidfVectorizer, self)._word_ngrams(tokens, None)
			if stop_words is not None:
				new_tokens = []
				for token in tokens:
					split_words  =  token.split(' ')
                    # Only check the first and last word for stop words
					if len(token) > 2 and split_words[0] not in stop_words and split_words[-1] not in stop_words:
                        #stem every word in token and eliminate first token that contain words below 2 characters
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
			
	class StemmedTfidfVectorizer(TfidfVectorizer):
	
		'''
		This functions is for extracting noun phrase candidate
		'''
	
		def build_tokenizer(self):
			tokenizer = super(TfidfVectorizer, self).build_tokenizer()
			return lambda doc: (stemmer.stem(token) for token in tokenizer(doc) if token not in np_stop_words)

	'''
	In term of stop words, because in ngram candidates, stop words in the middle is allowed,
	on nounphrase, some stop words that are more likely to be keywords, are included
	'''
	ngram_stop_words = text.ENGLISH_STOP_WORDS
	np_stop_words = set(text.ENGLISH_STOP_WORDS)
	s = ['of','in','on','for']
	np_stop_words = np_stop_words.difference(s)
	
	
	def get_values(tfidf, corpus):
	
		'''
		Store tuple of candidate and tfidf value into a list 
		'''
		
		#count the tfidf from text and get the candidates
		matrix = tfidf.fit_transform(corpus)
		feature_names = tfidf.get_feature_names()

		#how to print tf-idf from https://stackoverflow.com/questions/34449127/
		#sklearn-tfidf-transformer-how-to-get-tf-idf-values-of-given-words-in-documen
		#mapping the candidates per document into list
		candidates = []
		for doc in range(0,len(corpus)):
			feature_index = matrix[doc,:].nonzero()[1]
			tfidf_doc = zip(feature_index, [matrix[doc, x] for x in feature_index])
			names_tfidf = [(w, s) for w, s in [(feature_names[i], s) for (i, s) in tfidf_doc]]
			candidates.append(names_tfidf)
		return candidates
    
	#regular expression is used to extract only words in text, hypen is acceptable between the words
	if type == 'ngram':
		tfidf = NewTfidfVectorizer(ngram_range = (1,5), stop_words = ngram_stop_words, token_pattern = r"(?u)\b[A-Za-z-]+\b")
		candidates = get_values(tfidf, corpus)
		return candidates
	elif type == 'np':
		tfidf = StemmedTfidfVectorizer(ngram_range = (1,5), stop_words = np_stop_words, vocabulary = vocab, token_pattern = r"(?u)\b[A-Za-z-]+\b")
		candidates = get_values(tfidf, corpus)
		return candidates	
	
	
def create_phrase_vocabulary(raw_data):
    
	'''
	Extract vocabulary of nounphrase, because tfidfvectorizer only automatically extract ngram,
		if we want to use different format or different vocabulary, vocabulary must be created.
	'''
    
	#grammar to extract the noun phrase
	grammar = r'NP: {(<JJ.*>* <VBN>? <NN.*>+ <IN>)? <JJ.*>* <VBG>? <NN.*>+}' 

	#set the punctuation and chunker
	punct  =  set(string.punctuation) 
	chunker  =  RegexpParser(grammar)
    
	def lambda_unpack(f):
		#function to unpack the tuple
		return lambda args:f(*args)
    
	#tokenize and create pos tags per sentence, then get its IOB tag
	postag_sents  =  pos_tag_sents(word_tokenize(sent) for sent in raw_data)
	noun_phrases  =  list(chain.from_iterable(tree2conlltags(chunker.parse(tagged_sent)) for tagged_sent in postag_sents)) 
    
    #join B-NP and I-NP tags as one noun phrase excluding O tags    
	merged_nounphrase  =  [' '.join(stemmer.stem(word) for word, pos, chunk in group).lower() for key, group in
                    itertools.groupby(noun_phrases, lambda_unpack(lambda word, pos, chunk: chunk !=  'O')) if key]
    
    #filter the term below than two characters and punctuation
	all_nounphrases = [cand for cand in merged_nounphrase
            if len(cand) > 2 and not all(char in punct for char in cand)]
    
    #select distinct noun phrases
	vocabulary = (list(set(all_nounphrases)))
	return vocabulary
	

