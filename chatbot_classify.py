#!/usr/bin/python
# -*- coding: UTF-8 -*-
import nltk
from nltk.stem.lancaster import LancasterStemmer

import os
import json
import datetime
import numpy as np
import time
stemmer = LancasterStemmer()

# 3 classes of training data
training_data = []
training_data.append({"class":"greeting", "sentence":"how are you?"})
training_data.append({"class":"greeting", "sentence":"how is your day?"})
training_data.append({"class":"greeting", "sentence":"good day"})
training_data.append({"class":"greeting", "sentence":"how is it going today?"})

training_data.append({"class":"goodbye", "sentence":"have a nice day"})
training_data.append({"class":"goodbye", "sentence":"see you later"})
training_data.append({"class":"goodbye", "sentence":"have a nice day"})
training_data.append({"class":"goodbye", "sentence":"talk to you soon"})

training_data.append({"class":"sandwich", "sentence":"make me a sandwich"})
training_data.append({"class":"sandwich", "sentence":"can you make a sandwich?"})
training_data.append({"class":"sandwich", "sentence":"having a sandwich today?"})
training_data.append({"class":"sandwich", "sentence":"what's for lunch?"})
print ("%s sentences in training data" % len(training_data))

# process data
words = []
classes = []
documents = []
ignore_words = ['?']
for pattern in training_data:
  w = nltk.word_tokenize(pattern['sentence'])
  words.extend(w)
  documents.append((w, pattern['class']))
  if pattern['class'] not in classes:
    classes.append(pattern['class'])

words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = list(set(words))

classes = list(set(classes))
print (len(documents), "documents")
print (len(classes), "classes", classes)
print (len(words), "unique stemmed words", words)


training = []
output = []
output_empty = [0] * len(classes)

for doc in documents:
  bag = []
  pattern_words = doc[0]
  pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
  for w in words:
    bag.append(1) if w in pattern_words else bag.append(0)

  training.append(bag)
  output_row = list(output_empty)
  output_row[classes.index(doc[1])] = 1
  output.append(output_row)


i = 0
w = documents[i][0]
print ([stemmer.stem(word.lower()) for word in w ])
print training[i]
print output[i]

def sigmoid(x):
  output = 1 / (1+np.exp(-x))
  return output
# sigmoid函数求导
def sigmoid_output_to_derivative(output):
  return output * (1 - output)

def clean_up_sentence(sentence):
  sentence_words = nltk.word_tokenize(sentence)
  sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
  return sentence_words

def bow(sentence, words, show_details=False):
  sentence_words = clean_up_sentence(sentence)
  bag = [0] * len(words)
  for s in sentence_words:
    for i, w in enumerate(words):
      if w == s:
        bag[i] = 1
        if show_details:
          print ("found in bag: %s" % w)
  return np.array(bag)


def think(sentence, show_details=False):
  x = bow(sentence.lower(), words, show_details)
  if show_details:
    print ("sentence: ", sentence, "\n bow: ", x)
  l0 = x
  l1 = sigmoid(np.dot(l0, synapse_0))
  l2 = sigmoid(np.dot(l1, synapse_1))
  return l2


def train(X, y, hidden_neurons=10, alpha=1, epochs=50000, dropout=False, dropout_percent=0.5):
  print ("Training with %s neurons, alpha:%s, dropout:%s %s" % (hidden_neurons, str(alpha), dropout, dropout_percent if dropout else ''))
  print ("Input matrix: %sx%s  Output matrix: %sx%s" % (len(X), len(X[0]), 1, len(classes)))
  np.random.seed(1)
  last_mean_error = 1
  # numpy.random.random返回的是一个元素值在半开区间[0.0, 1.0)  这里将其转换成了元素范围在区间[-1.0, 1.0)
  # len(X[0])=26  20个隐层神经元 最后随机出来26 * 20的元素值在[-1.0, 1.0)之间的矩阵  输入到隐层
  synapse_0 = 2 * np.random.random((len(X[0]), hidden_neurons)) - 1   #  
  # 隐层到输出 20个隐层神经元 3种类别输出  20 * 3
  synapse_1 = 2 * np.random.random((hidden_neurons, len(classes))) - 1
  print synapse_0
  print len(synapse_0)
  print np.shape(synapse_0)
  # print synapse_0
  print len(synapse_1)
  print np.shape(synapse_1)
  # print synapse_1
  # 权重初始化
  prev_synapse_0_weight_update = np.zeros_like(synapse_0)
  prev_synapse_1_weight_update = np.zeros_like(synapse_1)

  synapse_0_direction_count = np.zeros_like(synapse_0)
  synapse_1_direction_count = np.zeros_like(synapse_1)
  # 迭代50000次
  for j in iter(range(epochs+1)):
    layer_0 = X 
    # print ("layer_0: ", layer_0)
    # (12 x 26) * (26 x 20) = (12 x 20)
    layer_1 = sigmoid(np.dot(layer_0, synapse_0))
    # print ("layer_1: ", layer_1)
    if dropout:
      layer_1 *= np.random.binomial([np.ones((len(X), hidden_neurons))], 1-dropout_percent)[0] * (1.0 / (1-dropout_percent))
    # (12 x 20) * (20 x 3) = (12 x 3)
    layer_2 = sigmoid(np.dot(layer_1, synapse_1))
    if j == 1000:
      print '**************'
      print np.shape(layer_1)
      print np.shape(synapse_1)
      print np.shape(layer_2)
    # print ("layer_2: ", layer_2)
    # 误差
    layer_2_error = y - layer_2
    # print ("layer_2_error: ", layer_2_error)
    # 每隔10000次迭代打印输出一次
    if (j % 10000) == 0 and j > 5000:
      # 损失误差越来越小，每次迭代进行更新；如果此次第10000次的迭代误差大于上次的，直接break掉
      if np.mean(np.abs(layer_2_error)) < last_mean_error:
        print ("delta after " + str(j) + " iterations: " + str(np.mean(np.abs(layer_2_error))))
        last_mean_error = np.mean(np.mean(np.abs(layer_2_error)))
      else:
        print ("break: ", np.mean(np.abs(layer_2_error)), ">", last_mean_error)
        break
    # 反向传播  根据误差和梯度去更新权重  这里没有做正则化处理
    if j == 1000:
      print '============='
      print np.shape(layer_2)
      print np.shape(sigmoid_output_to_derivative(layer_2))
      print np.shape(layer_2_error)
    # 误差与输出层的求导进行元素乘积
    layer_2_delta = layer_2_error * sigmoid_output_to_derivative(layer_2)
    # (12 x 3)  *  (3 x 20) = (12 x 20)
    layer_1_error = layer_2_delta.dot(synapse_1.T)
    # layer_1是12 x 20  layer_1的误差与layer_1隐藏层的输出进行元素积
    layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)
    # 更新权重  layer_0输入 12 x 26  
    # （26 x 12）* (12 x 20) = 26 x 20  
    synapse_0_weight_update = layer_0.T.dot(layer_1_delta)
    # （20 x 12）* (12 x 3) = 20 x 3 
    synapse_1_weight_update = layer_1.T.dot(layer_2_delta)

    if j == 1000:
      print '&&&&&&&&&&&&'
      print np.shape(synapse_0_weight_update)
      print np.shape(synapse_1_weight_update)

    if j > 0:
      synapse_0_direction_count += np.abs(((synapse_0_weight_update > 0) + 0) - ((prev_synapse_0_weight_update > 0) + 0))
      synapse_1_direction_count += np.abs(((synapse_1_weight_update > 0) + 0) - ((prev_synapse_1_weight_update > 0) + 0))   

    #  
    synapse_1 += alpha * synapse_1_weight_update
    synapse_0 += alpha * synapse_0_weight_update

    prev_synapse_0_weight_update = synapse_0_weight_update
    prev_synapse_1_weight_update = synapse_1_weight_update
  
  now = datetime.datetime.now()

  # persist synapses  固化模型
  synapse = {'synapse0': synapse_0.tolist(), 'synapse1': synapse_1.tolist(),
              'datetime': now.strftime("%Y-%m-%d %H:%M"),
              'words': words,
              'classes': classes
            }
  synapse_file = "synapses.json"

  with open(synapse_file, 'w') as outfile:
    json.dump(synapse, outfile, indent=4, sort_keys=True)
  print ("saved synapses to: ", synapse_file)


X = np.array(training)
y = np.array(output)

start_time = time.time()
train(X, y, hidden_neurons=20, alpha=0.1, epochs=100000, dropout=True, dropout_percent=0.2)
elapsed_time = time.time() - start_time
print ("processing time: ", elapsed_time, "senconds")


ERROR_THRESHOLD = 0.2
synapse_file = 'synapses.json'
with open(synapse_file) as data_file:
  synapse = json.load(data_file)
  synapse_0 = np.asarray(synapse['synapse0'])
  synapse_1 = np.asarray(synapse['synapse1'])


def classify(sentence, show_details=False):
  results = think(sentence, show_details)

  results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]
  results.sort(key=lambda x: x[1], reverse=True)
  return_results = [[classes[r[0]], r[1]] for r in results]
  print ("%s \n classification: %s" % (sentence, return_results))
  return return_results

classify("sudo make me a sandwich")
classify("how are you today?")
classify("talk to you tomorrow")
classify("who are you?")
classify("make me some lunch")
# 分不出来
classify("I love you")
classify("Are you ok?")
print ()
classify("how was your lunch?", show_details=True)



