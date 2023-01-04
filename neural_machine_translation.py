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

smoother = SmoothingFunction()
sentence_bleu(['hi'], 'hi', smoothing_function=smoother.method4)

sentence_bleu(['hi there'.split()], 'hi there'.split())

sentence_bleu(['hi there friend'.split()], 'hi there friend'.split())

sentence_bleu([[1,2,3,4]], [1,2,3,4])

eng2spa_tokens = {}
for eng, spa_list in eng2spa.items():
  spa_list_tokens = []
  for text in spa_list:
    tokens = tokenizer.tokenize(text.lower())
    spa_list_tokens.append(tokens)
  eng2spa_tokens[eng] = spa_list_tokens

#!pip install transformers sentencepiece transformers[sentencepiece]

from transformers import pipeline
translator = pipeline("translation", model='Helsinki-NLP/opus-mt-en-es', device=0)

