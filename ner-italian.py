#!/usr/bin/env python3

import nltk 
from nltk import sent_tokenize, word_tokenize, ne_chunk
from nltk.corpus import alpino

# Load the Alpino corpus for Italian
italian_corpus = alpino.words()

# Your Italian text
italian_text = "Il Presidente della Repubblica Italiana, Sergio Mattarella, ha tenuto un discorso oggi a Roma."
/Users/jacopo/ner-italian.py
# Tokenize the text into sentences and words
sentences = sent_tokenize(italian_text)
tokenized_words = [word_tokenize(sentence) for sentence in sentences]

# Perform named entity recognition
for words in tokenized_words:
	tagged_words = nltk.pos_tag(words)
	named_entities = ne_chunk(tagged_words)
	
	# Filter entities with label 'PERSON'
	for subtree in named_entities:
		if isinstance(subtree, nltk.Tree) and subtree.label() == 'PERSON':
			person_name = ' '.join([token[0] for token in subtree.leaves()])
			print(f"Person: {person_name}")
		if isinstance(subtree, nltk.Tree) and subtree.label() == 'ORGANIZATION':
			org_name = ' '.join([token[0] for token in subtree.leaves()])
			print(f"Organisation: {org_name}")
		if isinstance(subtree, nltk.Tree) and subtree.label() == 'GPE':
			place_name = ' '.join([token[0] for token in subtree.leaves()])
			print(f"Place: {place_name}")
	