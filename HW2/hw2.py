#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import nltk
import random
import csv
import pickle
from collections import Counter

from nltk.data import load
from sklearn.model_selection import train_test_split

# In[2]:


nltk.download('treebank')
nltk.download('tagsets')


# In[3]:


tagdict = load('help/tagsets/upenn_tagset.pickle')


# In[4]:


TAGS = list(tagdict.keys())
TAGS.append("UNK")


# In[5]:


len(TAGS)


# In[6]:


def readFile(fileName):
    finalList = []
    temp = []
    with open("data/"+fileName) as tsv:
        for line in csv.reader(tsv, dialect="excel-tab"): #You can also use delimiter="\t" rather than giving a dialect.            
            if len(line) == 0:  # New Line
                if fileName != "test" and len(temp) > 0 and temp[-1][1] != ".":
                    # Missing period at the end of sentence, add it in
                    temp.append(tuple(["","."]))
                finalList.append(temp)
                temp = []
            else:
                if fileName == "test":
                    temp.append(tuple([line[1]]))
                else:
                    # Train or Dev
                    temp.append(tuple([line[1], line[2]]))

    return finalList


# In[31]:


trainList = readFile("train")
testList = readFile("test")
devList = readFile("dev")


# In[8]:


def combineSentences(l):
    finalList = []
    for i in l:
        for j in i:
            finalList.append(j)
    return finalList


# In[270]:


combinedDevData = combineSentences(devList)


# In[271]:


sentence_dev_tags = [tag[1] for tag in combinedDevData]
sentence_dev_words = [tag[0] for tag in combinedDevData]


# In[272]:


combinedData = combineSentences(trainList)

TRAINING_TAGS = [tag[1] for tag in combinedData]
TRAINING_WORDS = [tag[0] for tag in combinedData]


# In[273]:

print("Utilizing 1 as threshold")
# Getting rid of tags labeled unknown
unknown_count = 0
for ind, tag in enumerate(TRAINING_TAGS):
    if tag not in set(TAGS):
        unknown_count += 1
        TRAINING_TAGS[ind] = "UNK"
print("Unknown Counts", unknown_count, "out of", len(TRAINING_TAGS))


# In[274]:


counter = Counter(TRAINING_WORDS)


# In[277]:


remove_set = set()
threshold = 1

for c,v in counter.items():
    if v <= threshold:
        remove_set.add(c)
remove_count = len(remove_set)

print("To remove", remove_count )
print("Total Uknown", str(remove_count+unknown_count))
print("Unique Words", len(list(set(TRAINING_WORDS))), "out of all words", len(TRAINING_WORDS))
print("Tags", TAGS)

# In[279]:


filehandler = open("tags_transition_prob_6.obj","rb")
tags_transition_prob = pickle.load(filehandler)
filehandler.close()


# In[167]:




# In[280]:


filtered_tags = TRAINING_TAGS
tag_count = len(filtered_tags)

dictCount = dict((i, filtered_tags.count(i)) for i in set(filtered_tags))

filtered_tags_dict = {}
for tag in TAGS:
    filtered_tags_dict[tag] = [t for t in combinedData if t[1] == tag]

filtered_tags_dict = {}
for tag in TAGS:
    filtered_tags_dict[tag] = [TRAINING_WORDS[ind] for ind, t in enumerate(TRAINING_TAGS) if t == tag]


filehandler = open("emission_prob_6.obj","rb")
emission_dict = pickle.load(filehandler)
filehandler.close()


# In[ ]:


initial_prob = {}
initial_total = 0
for d in trainList:
    d = d[0]
    if d[1] not in initial_prob:
        initial_prob[d[1]] = 1
    else:
        temp = initial_prob[d[1]]
        initial_prob[d[1]] = temp + 1
    initial_total += 1

# In[57]:


default = len([t for t in TRAINING_TAGS if t == 'UNK'])/len(TRAINING_TAGS)


# ### Greedy Decoding

# In[268]:


period_ind = TAGS.index(".")
greedy_predicted_tags = []
prev_ind = period_ind
for word in sentence_dev_words:
    final_prob = 0
    final_tag = ""
    final_tag_ind = -1
    for tag_ind, tag in enumerate(TAGS): # s2
        
        # How many tags go to the next tag
        transition_prob = tags_transition_prob[prev_ind][tag_ind]
        # What tag is to what word
        emission_prob = default
        if word in emission_dict:
            emission_prob = emission_dict[word][tag_ind]

        curr_final_prob = transition_prob * emission_prob

        if final_prob < curr_final_prob:
            final_prob = curr_final_prob
            final_tag = tag
            final_tag_ind = tag_ind

    greedy_predicted_tags.append(final_tag)
    prev_ind = final_tag_ind

count = 0
for i, pred in enumerate(greedy_predicted_tags):
    if pred == sentence_dev_tags[i]:
        count += 1
print("Greedy Dev Final Score", (count/len(greedy_predicted_tags)*100))


# ### Viterbi

# In[253]:

scores= []

for s in devList:
    sentence_words = [t[0] for t in s]
    sentence_tags = [t[1] for t in s]

    final_prob_matrix = [[0 for _ in range(len(TAGS))] for _ in range(len(sentence_words))]
    tag_ind_matrix = [[0 for _ in range(len(TAGS))] for _ in range(len(sentence_words)-1)]

    # just do initial state here
    period_ind = TAGS.index(".")
    for tag_ind, tag in enumerate(TAGS):
    #     transition_prob = tags_transition_prob[period_ind][tag_ind]
        word = sentence_words[0]
        
        transition_prob = 0
        if tag in initial_prob:
            transition_prob = initial_prob[tag]/initial_total
            
        emission_prob = 0.003008641237291414
        if word in emission_dict:
            emission_prob = emission_dict[word][tag_ind]

        final_prob_matrix[0][tag_ind] = emission_prob * transition_prob

    # Prob for not initial state
    for word_ind in range(1, len(sentence_words)):
        word = sentence_words[word_ind]
        for tag_ind, curr_tag in enumerate(TAGS): # s2 
            max_transition_prob = 0
            max_transition_ind = 45
            
            # How many words are in a certain tag
            emission_prob = 0.003008641237291414
            if word in emission_dict:
                emission_prob = emission_dict[word][tag_ind]

            for prev_ind, prev_tag in enumerate(TAGS):
                # How many tags go to the next tag
                transition_prob = tags_transition_prob[prev_ind][tag_ind]
                prev_prob = final_prob_matrix[word_ind-1][prev_ind]

                temp_prob = prev_prob * transition_prob
                if max_transition_prob < temp_prob:
                    max_transition_prob = temp_prob
                    max_transition_ind = prev_ind

            final_prob_matrix[word_ind][tag_ind] = emission_prob * max_transition_prob
    #         print(word_ind, tag_ind, emission_prob, max_transition_prob)
            tag_ind_matrix[word_ind-1][tag_ind] = max_transition_ind
    

    viterbi_predicted_tags = []

    max_value = -1
    max_ind = 45
    for ind, value in enumerate(final_prob_matrix[-1]):
        if value > max_value:
            max_value = value
            max_ind = ind
    viterbi_predicted_tags.append(TAGS[max_ind])

    prev_index = max_ind
    for n in range(len(sentence_words)-2, -1, -1):
        index = tag_ind_matrix[n][prev_index]
        viterbi_predicted_tags.append(TAGS[index])
        prev_index = index

    viterbi_predicted_tags.reverse()

    count = 0
    for i, pred in enumerate(viterbi_predicted_tags):
        if pred == sentence_tags[i]:
            count += 1
    score = (count/len(viterbi_predicted_tags)*100)
    scores.append(score)


# In[267]:
print("Viterbi Dev Final Score", sum(scores)/len(scores))



### File Creations

# HMM creation
import json

tag_dict = {}
for i, tag in enumerate(TAGS):
    for j, tag2 in enumerate(TAGS):
        tag_dict[str(tag+"-"+tag2)] = tags_transition_prob[i][j]
        
emission_d = {}
for e in emission_dict:
    for i, tag in enumerate(TAGS):
        emission_d[str(e+"-"+tag)] = emission_dict[e][i]
        
final_dict = {
    "transition": tag_dict,
    "emission": emission_d
}
with open("hmm.json", "w") as outfile:
    json.dump(final_dict, outfile)

# Vocab Creation
counter = Counter(TRAINING_TAGS)
vocab_tags = []
for i, tag in enumerate(TAGS):
    vocab_tags.append(str(i+1)+"\t"+str(tag)+"\t"+str(counter[tag]))
with open("vocab.txt", "w") as outfile:
    for line in vocab_tags:
        outfile.write(f'{line}\n')

# Viterbi Creation
final_hmm_list = []

for s in testList:
    sentence_words = [t[0] for t in s]

    final_prob_matrix = [[0 for _ in range(len(TAGS))] for _ in range(len(sentence_words))]
    tag_ind_matrix = [[0 for _ in range(len(TAGS))] for _ in range(len(sentence_words)-1)]

    # just do initial state here
    period_ind = TAGS.index(".")
    for tag_ind, tag in enumerate(TAGS):
    #     transition_prob = tags_transition_prob[period_ind][tag_ind]
        word = sentence_words[0]
        
        transition_prob = 0
        if tag in initial_prob:
            transition_prob = initial_prob[tag]/initial_total
            
        emission_prob = 0.003008641237291414
        if word in emission_dict:
            emission_prob = emission_dict[word][tag_ind]

        final_prob_matrix[0][tag_ind] = emission_prob * transition_prob

    # Prob for not initial state
    for word_ind in range(1, len(sentence_words)):
        word = sentence_words[word_ind]
        for tag_ind, curr_tag in enumerate(TAGS): # s2 
            max_transition_prob = 0
            max_transition_ind = 45
            
            # How many words are in a certain tag
            emission_prob = 0.003008641237291414
            if word in emission_dict:
                emission_prob = emission_dict[word][tag_ind]

            for prev_ind, prev_tag in enumerate(TAGS):
                # How many tags go to the next tag
                transition_prob = tags_transition_prob[prev_ind][tag_ind]
                prev_prob = final_prob_matrix[word_ind-1][prev_ind]

                temp_prob = prev_prob * transition_prob
                if max_transition_prob < temp_prob:
                    max_transition_prob = temp_prob
                    max_transition_ind = prev_ind

            final_prob_matrix[word_ind][tag_ind] = emission_prob * max_transition_prob
    #         print(word_ind, tag_ind, emission_prob, max_transition_prob)
            tag_ind_matrix[word_ind-1][tag_ind] = max_transition_ind
    

    viterbi_predicted_tags = []

    max_value = -1
    max_ind = 45
    for ind, value in enumerate(final_prob_matrix[-1]):
        if value > max_value:
            max_value = value
            max_ind = ind
    viterbi_predicted_tags.append(TAGS[max_ind])

    prev_index = max_ind
    for n in range(len(sentence_words)-2, -1, -1):
        index = tag_ind_matrix[n][prev_index]
        viterbi_predicted_tags.append(TAGS[index])
        prev_index = index

    viterbi_predicted_tags.reverse()

    for word_ind, word in enumerate(sentence_words):
        final_tag = viterbi_predicted_tags[word_ind]
        final_hmm_list.append(str(word_ind+1)+"\t"+word+"\t"+final_tag)
    final_hmm_list.append("")

with open("viterbi.out", "w") as outfile:
    for line in final_hmm_list:
        outfile.write(f'{line}\n')

# Greedy Creation
final_greedy_list = []

for s in testList:
    sentence_words = [t[0] for t in s]
    
    period_ind = TAGS.index(".")
    greedy_predicted_tags = []
    prev_ind = period_ind
    
    for word_ind, word in enumerate(sentence_words):
        final_prob = 0
        final_tag = ""
        final_tag_ind = -1
        for tag_ind, tag in enumerate(TAGS): # s2
            # How many tags go to the next tag
            transition_prob = 0
            if word_ind == 0:            
                if tag in initial_prob:
                    transition_prob = initial_prob[tag]/initial_total
            else:
                transition_prob = tags_transition_prob[prev_ind][tag_ind]
                
            # What tag is to what word
            emission_prob = 0.003008641237291414
            if word in emission_dict:
                emission_prob = emission_dict[word][tag_ind]

            curr_final_prob = transition_prob * emission_prob

            if final_prob < curr_final_prob:
                final_prob = curr_final_prob
                final_tag = tag
                final_tag_ind = tag_ind

        greedy_predicted_tags.append(final_tag)
        prev_ind = final_tag_ind
        
        final_greedy_list.append(str(word_ind+1)+"\t"+word+"\t"+final_tag)
    final_greedy_list.append("")

with open("greedy.out", "w") as outfile:
    for line in final_greedy_list:
        outfile.write(f'{line}\n')