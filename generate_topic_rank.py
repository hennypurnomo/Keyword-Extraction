'''
To generate the keywords from topic rank, this file has been created.
Because it takes a times to run it, this keywords must be runned before create the feature.
Store all extracted topic rank keywords into pickle.
'''

import utils
from itertools import chain
from pytopicrank import TopicRank


def calculate_topic_rank(corpus, data):
	'''
	calculate_topic_rank(corpus = name of corpus, data = ('default' for inspect, semeval dataset, 'news' for news dataset))
	'''
	
	# 15 - number of keywords in inspect label is between 5, 10 and the model label is 5, 10, 15, therefore, extract 15.
	# But this number will be adjusted (same like how many number of keyphrase) when extract the features
	# 50 - number of keywords in news label is 50
	if data == 'default':
		n_topics = 15
	elif data == 'news':
		n_topics = 50
		
	#store the keywords into a list
	all_topics = []
	for n_doc in corpus:
		all_topics.append(TopicRank(n_doc).get_top_n(n = n_topics))
	return all_topics

#Read pickle file from the model
txt_train_data = utils.open_pickle('./pickle/txt train data')
txt_test_data = utils.open_pickle('./pickle/txt test data')
xml_train_data = utils.open_pickle('./pickle/xml train data')
xml_test_data = utils.open_pickle('./pickle/xml test data')

#Read pickle file from the news dataset
news_train_data = utils.open_pickle('./pickle/500N-KPCrowd/train data')
news_test_data = utils.open_pickle('./pickle/500N-KPCrowd/test data')

#Read pickle file from the inspec dataset
inspec_train_data = utils.open_pickle('./pickle/inspec/train data')
inspec_test_data = utils.open_pickle('./pickle/inspec/test data')


#Processing on the model data
txt_train_topics = calculate_topic_rank(txt_train_data, data='default')
txt_test_topics = calculate_topic_rank(txt_test_data, data='default')
xml_train_topics = calculate_topic_rank(xml_train_data, data='default')
xml_test_topics = calculate_topic_rank(xml_test_data, data='default')

#Processing on the news data
news_train_topics = calculate_topic_rank(news_train_data, data='news')
news_test_topics = calculate_topic_rank(news_test_data, data='news')

#Processing on the inspec data
inspec_train_topics = calculate_topic_rank(inspec_train_data, data='default')
inspec_test_topics = calculate_topic_rank(inspec_test_data, data='default')

#create all pickle for semeval dataset
utils.create_pickle(txt_train_topics, './pickle/txt train topics')
utils.create_pickle(txt_test_topics, './pickle/txt test topics')
utils.create_pickle(xml_train_topics, './pickle/xml train topics')
utils.create_pickle(xml_train_topics, './pickle/xml test topics')

#create all pickles for news dataset
utils.create_pickle(news_train_topics, './pickle/500N-KPCrowd/train topics')
utils.create_pickle(news_test_topics, './pickle/500N-KPCrowd/test topics')

#create all pickle for inspec dataset
utils.create_pickle(inspec_train_topics, './pickle/inspec/train topics')
utils.create_pickle(inspec_test_topics, './pickle/inspec/test topics')

print("done")