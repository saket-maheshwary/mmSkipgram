from __future__ import print_function
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

# for explanation of keys of POS tagger
# http://www.comp.leeds.ac.uk/amalgam/tagsets/upenn.html

fname = '../data/text8_proc'
fnameout = '../data/text8_proc_lemmatized'

lemmmatizer = WordNetLemmatizer()
swords = stopwords.words("english")
num_words = 1000

pos = ['NN', 'NNP', 'NNPS', 'NNS']

ifile = open(fname, 'r')
# doc = ifile.read(num_words)
doc = ifile.read()
doc_proc = [a[0] for a in filter( lambda x: x[1] in pos,
              nltk.pos_tag(doc.split(' ')))]
ifile.close()

ifile = open(fname, 'r')
ofile = open(fnameout, 'w')
# for word in ifile.read().split(' '):
for word in doc_proc:
	nword = lemmmatizer.lemmatize(word.lower())
	nword = nword if nword not in swords else ''
	# print(word, nword)
	if nword != '':
		ofile.write(nword)
		ofile.write(' ')
ifile.close()
ofile.close()
