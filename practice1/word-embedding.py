#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 16:15:22 2021

@author: alejandro
"""

from gensim import models
import pandas as pd 
import re 

data = pd.read_csv("../data/quotes_dataset.csv") 
data.dropna()

# Grab a bunch of quotes from famous authors
quotes = data.iloc[0:400000,0].values
quotes_split = []

# Split in words so we can create the vectors without having for example "love" and "love:" as different items
for quote in quotes:
    quote = quote.lower()
    quotes_split.append(list(filter(None, re.split(',|:|;|_|-|!|\.| |\)|\(|\?|\"', quote))) )

model = models.Word2Vec(sentences=quotes_split, size=100, window=5, min_count=1, workers=4)

# Save the model for later retrieving
model.save("../models/famous_quotes_model.wv")

# Play with it, find similarities
model.wv.most_similar(positive='love')
# Out[78]: 
# [('loving', 0.6740010976791382),
#  ('friendship', 0.6651197671890259),
#  ('kindness', 0.6522172093391418),
#  ('hate', 0.6392741799354553),
#  ('trust', 0.6313838362693787),
#  ('forgiveness', 0.6226910352706909),
#  ('compassion', 0.6096630096435547),
#  ('loved', 0.5879759192466736),
#  ('respect', 0.5869451761245728),
#  ('affection', 0.5687838792800903)]