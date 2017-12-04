from __future__ import print_function
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
import multiprocessing as mp

# for explanation of keys of POS tagger
# http://www.comp.leeds.ac.uk/amalgam/tagsets/upenn.html

fname = '../data/text8_proc'
fnameout = '../data/text8_proc_lemmatized'

lemmmatizer = WordNetLemmatizer()
swords = set(stopwords.words("english"))
num_words = 1000

pos = ['NN', 'NNP', 'NNPS', 'NNS']

# doc = ifile.read(num_words)


def worker(i, num_processes, fname, fnameout):

	ifile = open(fname, 'r')
	# doc = ifile.read()
	doc = ifile.read(1000)
	N = len(doc)
	chunk_size = int(N) / int(num_processes)
	start = i * chunk_size
	if i == num_processes-1: 
	    end = N
        else:
            end = (i+1) * chunk_size

	doc_proc = [a[0] for a in filter(lambda x: x[1] in pos,
                                         nltk.pos_tag(doc[start:end].split(' ')))]
	ifile.close()

	fnmo = fnameout + '_' + '%d' % i
	ofile = open(fnmo, 'w')
	# for word in ifile.read().split(' '):
	for word in doc_proc:
		nword = lemmmatizer.lemmatize(word.lower())
		nword = nword if nword not in swords else ''
		# print(word, nword)
		if nword != '':
			ofile.write(nword)
			ofile.write(' ')
	ofile.close()

num_processes = 8
jobs = []
for i in range(num_processes):
    name = "process_%d" % i
    p = mp.Process(name=name,
		   target=worker,
		   args=(i, num_processes,
			 fname, fnameout
			 ))
    jobs.append(p)
    p.start()

for p in jobs:
    print('joining %s : %d' % (p.name, p.pid))
    p.join()
print('All threads completed')
