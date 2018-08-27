'''
This file contains all the function for preprocessing, like clean the text, 
	load the files, load the gold standard, and create the corpus. 
There are three types of loading files, such as xml for model, text files for model,
	and xml files for evaluation.
'''

import os, re, string, json
import xml.etree.ElementTree as et
from collections import OrderedDict
from nltk import word_tokenize, pos_tag, pos_tag_sents
from nltk.corpus import wordnet, stopwords
from itertools import chain


def clean(input_list):
	'''
	Eliminate unwanted characters from text file format.
	'''
    
	result = []
    #remove unwanted character per line
	for line in input_list:
		clean = re.sub("(\.)?\n",'', line) #remove \n
		clean = re.sub("('s)",'', clean) #remove 's
		clean = re.sub("\[([0-9]{1,2}\,?\s?)+\]",'', clean) #remove [2]
		clean = re.sub("\(([0-9]{1,2}\,?\s?)+\)",'', clean) #remove (2)
		clean = re.sub("([Ff]ig.|[Ff]igure|[Tt]ab.|[Tt]able)\s?[0-9]{1,2}",'', clean) #remove fig. 2 etc
		clean = re.sub(r"\b((https?://|www.)[^\s]+)",'', clean) #remove email
		result.append(clean)
	return result

	
def clean_xml(input_list): 
	'''
	Eliminate unwanted characters from xml format.
	'''
	
	result = []
    #remove unwanted character per line
	for line in input_list:
		clean = re.sub("(-LSB-|-LRB-|-LCB-)",'(', line) #substitute brackets
		clean = re.sub("(-RSB-|-RRB-|-RCB-)",')', clean) #substitute brackets
		clean = re.sub("('s)",'', clean) #remove 's
		clean = re.sub("\[([0-9]{1,2}\,?\s?)+\]",'', clean) #remove reference like [2]
		clean = re.sub("\(([0-9]{1,2}\,?\s?)+\)",'', clean) #remove number (2)
		clean = re.sub("([Ff]ig.|[Ff]igures?|[Tt]ab.|[Tt]ables?)\s?\d{1,2}",'', clean) #remove fig. 2 etc
		clean = re.sub(r"\b((https?://|www.)[^\s]+)",'', clean) #remove email
		result.append(clean)
	return result
	

def load_files(path):
	'''
	Load text files and parse into some selected section. 
	Store per document into a list.
	'''
    
	#create dictionary per document, extract title without its extension
	raw = []
	for file in path:
		dict_doc = {'doc_id': None, 'title': None, 'abstract': None, 'introduction': None, 'full-text': None}
		file_id = os.path.basename(file).rstrip('.txt.final')
		dict_doc['doc_id'] = file_id
		
        #open and clean the text
		source = open(file,encoding = 'utf-8').readlines()
		source = clean(source)
        
        #parse first two lines in each document, the second sentence is the candidates of title
		beginning = re.sub("\n", "", source[0])
		candidate = re.sub("\n", "", source[1])
		h_candidate = word_tokenize(re.sub("-",' ',candidate))
        
		#check the title candidate, 
		#if total words that exist on wordnet more than half or the same number of its total sentence length, 
		#this candidate is a continuation from title 
		title = []
		name = []
		for word in h_candidate:
			if wordnet.synsets(word): 
				title.append(word)
			else:
				name.append(word)
			if len(title) > len(name): 
				newtitle = beginning+' '+candidate
			elif len(title) == len(name):
				newtitle = beginning
			else:
				newtitle = beginning

		dict_doc['title'] = newtitle
        
		#extract only content of document, find some index for some sections
		content = source[2:]
        ######check header, inconsistency all file
		r_intro = re.compile("^1\.?\s[A-Z]+")
		r_after_intro = re.compile("^2\.?\s[A-Z]+")
		r_ref = re.compile("[0-9]{1,2}?\.?\s?R[EFERENCES|eferences]") #detect reference
        #r_header = re.compile("[0-9]{1,2}?\.?\s?[A-Z]")
		in_abstract = content.index('ABSTRACT')
		in_authorkey = content.index('Categories and Subject Descriptors')
        
        #find the first index that contain the selected sections
		list_intro = [i for i, item in enumerate(content) if re.search(r_intro, item)]
		in_intro = list_intro[0]
		list_after_intro = [i for i, item in enumerate(content) if re.search(r_after_intro, item)]
		in_after_intro = list_after_intro[0]
		list_ref = [i for i, item in enumerate(content) if re.search(r_ref, item)]
		in_ref = list_ref[0]
        
		#save only content without the header of section
		abstract = content[in_abstract + 1:in_authorkey] #eliminate keyword and category
		intro = content[in_intro + 1:in_after_intro]
		body = content[in_after_intro + 1:in_ref]      
        
		#store text per section in dictionary
		list_title = []
		list_title.append(newtitle)
		full_text = list(chain(list_title, abstract, intro, body))
		dict_doc['abstract'] = abstract
		dict_doc['introduction'] = intro
		dict_doc['body'] = body
		dict_doc['full_text'] = full_text
       
        #per sentence in a document
		raw.append(dict_doc)
	return raw
	
	
def load_xml(path):
	
	'''
	Load and parse xml files into some sections.
	Store per document into a list.
	'''
	
	#create dictionary per document, extract title without its extension
	all_files = []
	for file in path:
		tree = et.parse(file)
		dict_doc = {'doc_id': None, 'title': None, 'abstract': None, 
                 'introduction': None, 'method': None, 
                 'evaluation': None, 'related work': None, 
                 'conclusions': None, 'full_text': None, 'glabels': None}
		file_id = os.path.basename(file.rstrip('.xml'))
		dict_doc['doc_id'] = file_id
        
		#parse the xml file per sentence into some sections
		#sections are title, abstract, introduction, method, evaluation, related work, conclusions, and unknown category
		title = []
		abstract = []
		introduction = []
		method = []
		evaluation = []
		related_work = []
		conclusions = []
		unknown = []
		full = []
		for sentence in tree.iterfind('./document/sentences/sentence'):
			if sentence.attrib['section'] == 'title' and sentence.attrib['type'] != 'sectionHeader':
				title.append(' '.join([x.text for x in sentence.findall("tokens/token/word")]))
			elif sentence.attrib['section'] == 'abstract' and sentence.attrib['type'] != 'sectionHeader':
				abstract.append(' '.join([x.text for x in sentence.findall("tokens/token/word")]))
			elif sentence.attrib['section'] == 'introduction' and sentence.attrib['type'] != 'sectionHeader':
				introduction.append(' '.join([x.text for x in sentence.findall("tokens/token/word")]))
			elif sentence.attrib['section'] == 'method' and sentence.attrib['type'] != 'sectionHeader':
				method.append(' '.join([x.text for x in sentence.findall("tokens/token/word")]))
			elif sentence.attrib['section'] == 'evaluation' and sentence.attrib['type'] != 'sectionHeader':
				evaluation.append(' '.join([x.text for x in sentence.findall("tokens/token/word")]))
			elif sentence.attrib['section'] == 'related work' and sentence.attrib['type'] != 'sectionHeader':
				related_work.append(' '.join([x.text for x in sentence.findall("tokens/token/word")]))
			elif sentence.attrib['section'] == 'conclusions' and sentence.attrib['type'] != 'sectionHeader':
				conclusions.append(' '.join([x.text for x in sentence.findall("tokens/token/word")]))
			else:
				unknown.append(' '.join([x.text for x in sentence.findall("tokens/token/word")]))
        
		#join and clean all sentences into a text, store to dictionary
		dict_doc['title'] = ' '.join(title)
		dict_doc['abstract'] = ' '.join(clean_xml(abstract))
		dict_doc['introduction'] = ' '.join(clean_xml(introduction))
		dict_doc['method'] = ' '.join(clean_xml(method))
		dict_doc['evaluation'] = ' '.join(clean_xml(evaluation))
		dict_doc['related work'] = ' '.join(clean_xml(related_work))
		dict_doc['conclusions'] = ' '.join(clean_xml(conclusions))
		dict_doc['unknown'] = ' '.join(clean_xml(unknown))
		full = title + abstract + introduction + method + evaluation + related_work + conclusions
		dict_doc['full_text'] = ' '.join(clean_xml(full))
        
		all_files.append(dict_doc)
	return all_files
	

def load_xml_non_title(path):
	
	'''
	Load and parse xml files for evaluation (inspec and news dataset).
	Those files do not have title.
	Store per document into a list.	
	'''
	
	#create dictionary per document, extract title without its extension
	all_files = []
	for file in path:
		tree = et.parse(file)
		dict_doc = {'doc_id': None, 'full_text': None}
		file_id = os.path.basename(file.rstrip('.xml'))
		dict_doc['doc_id'] = file_id
        
		#parse the file per sentence
		sentences = []
		for sentence in tree.iterfind('./document/sentences/sentence'):
			sentences.append(' '.join([x.text for x in sentence.findall("tokens/token/word")]))
		
		#join all sentences into a text
		dict_doc['full_text'] = ' '.join(clean_xml(sentences))
        
		all_files.append(dict_doc)
		
	return all_files
	

def create_corpus(raw_data):
	'''
	Create a corpus for text file format.
	To feed the data into tfidfvectorizer or countvectorizer, 
	it should consist of a list that contain list per document.
	'''

	#merge all content into a text
	train_data = []
	for doc in raw_data:
		train_data.append(' '.join(doc['full_text']))
	return train_data

	
def create_xml_corpus(raw_data):
	'''
	Create a corpus to xml format
	'''
	
	#merge all content into a text
	train_data = []
	for doc in raw_data:
		train_data.append(doc['full_text'])
	return train_data
	
	
def extract_keyphrase(gold_data):
    '''
	Extract the gold standard for model.
	Both of xml and text file version share the same label.
	'''
	
	#some keywords contain + and /, parse it into to words
    r_plus = re.compile("^.*\+.*$")
    r_slash = re.compile("^.*\s.*\/.*$")
    
	#extract the keywords and save into a list per document
    gold_standard = []
    for line in gold_data.split('\n'):
        doc = []      
        for key in line[6:].split(','):
            if key[0] == ' ':
                doc.append(key[1:])
            elif re.search(r_plus, key):
                split = []
                for element in key.split('+'):
                    doc.append(element)
            elif re.search(r_slash, key):
                split = []
                for element in key.split('/'):
                    doc.append(element)
            else:
                doc.append(key)
        gold_standard.append(doc)
    return gold_standard
	
	
def extract_json_label(file, raw_data, file_type):
	'''
	extract_json_label(file = name of directory, file_type = ('default' for inspec, and 'news' for news))
	Extract the gold standart for evaluation.
	'''

	#load the json file
	load = json.load(file)
	
	#for inspec dataset, order the name of file incrementally (only extract the number of name file)
	#store into a list
	if file_type == 'default':
		order = OrderedDict(sorted(load.items(), key = lambda x: int(x[0])))	
		gold_standard = []
		for key, value in order.items():
			gold_standard.append(list(chain.from_iterable(value)))
		return gold_standard
	else:
		#this setting only for news, because the number can be used in different file name
		#the labels then are stored into a list
		order = OrderedDict(load.items())
		label = []
		for n_doc in range(0, len(raw_data)):
			match = [list(chain.from_iterable(value)) for key, value in order.items() if key == raw_data[n_doc]['doc_id']]
			label.append(list(chain.from_iterable(match)))
		return label
    
	