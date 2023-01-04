#!wget -nc http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip
#!unzip -nq spa-eng.zip
#!ls
#ls spa-eng
#!head spa-eng/spa.txt

# compile eng-spa translations
eng2spa = {}
for line in open('spa-eng/spa.txt'):
  line = line.rstrip()
  eng, spa = line.split("\t")
  if eng not in eng2spa:
    eng2spa[eng] = []
  eng2spa[eng].append(spa)

eng2spa

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w')

tokens = tokenizer.tokenize('¿Qué me cuentas?'.lower())
sentence_bleu([tokens], tokens)

sentence_bleu([['hi']], ['hi'])

