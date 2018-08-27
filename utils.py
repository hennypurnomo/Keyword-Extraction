'''This file is created to create and open pickle'''

import pickle

#create pickle from data
def create_pickle(data, name):

	'''
	create_pickle(data = name of variable data,
				name = location and string of pickle file)
	'''
    
	with open('%s.pickle' % name,'wb') as handle:
		result = pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
	return result

#open pickle from data
def open_pickle(name):

	'''
	open_pickle(name = location and string of pickle file)
	'''
	
	with open('%s.pickle' % name,'rb') as handle:
		result = pickle.load(handle)
	return result
