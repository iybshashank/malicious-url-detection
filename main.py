import tkinter as tk  
from tkinter import ttk
from tkinter import font
from tkinter.constants import BOTTOM
from tensorflow.keras.preprocessing import text
from tensorflow.python.platform.tf_logging import warning
  
win = tk.Tk()# Application Name  
win.title("Python GUI App")# Label  
win.geometry("1200x600")
win.configure(bg="white")
lbl = tk.Label(win, text = "Deep URL Detect",
					fg="#209E62",
					bg="white",
					font="Qwigley 50 bold")
lbl.pack(pady=40)

def url_prediction():

	import tkinter as tk  
	from tkinter import ttk
	from numpy import loadtxt
	from keras.models import load_model
	import pandas as pd
	# import library
	import numpy as np
	from sklearn.metrics import precision_recall_fscore_support,accuracy_score,precision_score,precision_score,f1_score,recall_score

	import re
	import matplotlib.image as mpimg
	import matplotlib.pyplot as plt
	import plotly.graph_objects as go
	import plotly.express as px
	from plotly.subplots import make_subplots
	import seaborn as sns
	import gc
	import random
	import os
	import pickle
	import tensorflow as tf
	from tensorflow.python.util import deprecation
	from urllib.parse import urlparse
	import tldextract

	from sklearn.model_selection import train_test_split
	from tensorflow.keras.preprocessing.text import Tokenizer
	from tensorflow.keras.preprocessing.sequence import pad_sequences
	from tensorflow.keras import models, layers, backend, metrics
	from tensorflow.keras.callbacks import EarlyStopping
	from keras.utils.vis_utils import plot_model
	from PIL import Image
	from sklearn.metrics import confusion_matrix, classification_report

	os.environ['PYTHONHASHSEED'] = '0'
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
	np.random.seed(0)
	random.seed(0)
	tf.random.set_seed(0)
	#LOAD DATA
	model = load_model('model.h5')
	#SUMMARIZE MODEL
	model.summary()
	from keras.models import Model
	model2= Model(model.input,model.get_layer('embedding').output)

	#LOAD DATASET
	data = pd.read_csv('input/data.csv')
	data.head()
	val_size = 0.2
	train_data, val_data = train_test_split(data, test_size=val_size, stratify=data['label'], random_state=0)
	fig = go.Figure([go.Pie(labels=['Train Size', 'Validation Size'], values=[train_data.shape[0], val_data.shape[0]])])
	fig.update_layout(title='Train and Validation Size')
	#PERCENTAGE OF CLASS (GOOD AND BAD)
	fig = go.Figure([go.Pie(labels=['Good', 'Bad'], values=data.label.value_counts())])
	fig.update_layout(title='Percentage of Class (Good and Bad)')

	#urlimp=urlimp_value.urlimp2
	urlimp=name.get()
	def parsed_url(url):
	    # extract subdomain, domain, and domain suffix from url
	    # if item == '', fill with '<empty>'
	    subdomain, domain, domain_suffix = ('<empty>' if extracted == '' else extracted for extracted in tldextract.extract(urlimp))
	    
	    return [urlimp,subdomain, domain, domain_suffix]
	def extract_url(urlimp):
	    # parsed url
	    extract_url_data = [parsed_url(urlimp) for urlimp in urlimp]
	    extract_url_data = pd.DataFrame(extract_url_data, columns=['url','subdomain', 'domain', 'domain_suffix'])
	    # concat extracted feature with main data
	    return extract_url_data

	def get_frequent_group(data, n_group):
	    # get the most frequent
	    data = data.value_counts().reset_index(name='values')
	    
	    # scale log base 10
	    data['values'] = np.log10(data['values'])
	    
	    # calculate total values
	    # x_column (subdomain / domain / domain_suffix)
	    x_column = data.columns[1]
	    data['total_values'] = data[x_column].map(data.groupby(x_column)['values'].sum().to_dict())
	    
	    # get n_group data order by highest values
	    data_group = data.sort_values('total_values', ascending=False).iloc[:, 1].unique()[:n_group]
	    data = data[data.iloc[:, 1].isin(data_group)]
	    data = data.sort_values('total_values', ascending=False)
	    
	    return data

	def plot(data, n_group, title):
	    data = get_frequent_group(data, n_group)
	    fig = px.bar(data, x=data.columns[1], y='values', color='label')
	    fig.update_layout(title=title)

	# extract url
	data = extract_url(data)
	train_data = extract_url(train_data)
	val_data = extract_url(val_data)
	fig = go.Figure([go.Bar(
	    x=['domain', 'Subdomain', 'Domain Suffix'], 
	    y = [data.domain.nunique(), data.subdomain.nunique(), data.domain_suffix.nunique()]
	)])

	tokenizer = Tokenizer(filters='', char_level=True, lower=False, oov_token=1)

	# fit only on training data
	#TOKENIZATION
	tokenizer.fit_on_texts(train_data['url'])
	n_char = len(tokenizer.word_index.keys())

	train_seq = tokenizer.texts_to_sequences(train_data['url'])
	val_seq = tokenizer.texts_to_sequences(val_data['url'])

	print('Before tokenization: ')
	print(train_data.iloc[0]['url'])
	print('\nAfter tokenization: ')
	print(train_seq[0])
	#PADDING
	sequence_length = np.array([len(i) for i in train_seq])
	sequence_length = np.percentile(sequence_length, 99).astype(int)
	print(f'Before padding: \n {train_seq[0]}')
	train_seq = pad_sequences(train_seq, padding='post', maxlen=sequence_length)
	val_seq = pad_sequences(val_seq, padding='post', maxlen=sequence_length)
	print(f'After padding: \n {train_seq[0]}')
	unique_value = {}
	for feature in ['subdomain', 'domain', 'domain_suffix']:
	    # get unique value
	    label_index = {label: index for index, label in enumerate(train_data[feature].unique())}
	    
	    # add unknown label in last index
	    label_index['<unknown>'] = list(label_index.values())[-1] + 1
	    
	    # count unique value
	    unique_value[feature] = label_index['<unknown>']
	    
	    # encode
	    train_data.loc[:, feature] = [label_index[val] if val in label_index else label_index['<unknown>'] for val in train_data.loc[:, feature]]
	    val_data.loc[:, feature] = [label_index[val] if val in label_index else label_index['<unknown>'] for val in val_data.loc[:, feature]]
	    
	train_data.head()
	val_x = [val_seq, val_data['subdomain'], val_data['domain'], val_data['domain_suffix']]

	val_pred = model2.predict(val_x)


	import numpy as np

	arr = val_pred[0][0]
	max_value = np.max(arr)
	max_value1 = np.mean(arr)

	if max_value1>0:
		blank.insert(tk.END, "  https://www.", 'warning1a')
		blank.insert(tk.END, urlimp, 'warning1a')
		blank.tag_config('warning1a', font="Calibri 10 bold")
		blank.insert(tk.END, " is \n")
		malicious="""  Malicious \n"""
		blank.insert(tk.END,malicious,'warning1')
		blank.tag_config('warning1', foreground="red")

	else:
		blank.insert(tk.END, "  https://www.", 'warning2a')
		blank.insert(tk.END, urlimp, 'warning2a')
		blank.tag_config('warning2a', font="Calibri 10 bold")
		blank.insert(tk.END, " is \n")
		safe="""  Safe \n"""
		blank.insert(tk.END,safe, 'warning')
		blank.tag_config('warning', foreground="#0F783F")


lb2 = tk.Label(win, text = "Enter URL to Predict",
					fg="Black",
					bg="white",
					font="Calibri 12 bold")
lb2.pack(pady=20)
name = tk.StringVar()  
nameEntered = tk.Entry(win, width = 30,
							textvariable = name,
							font="Calibri 20",
							highlightthickness=2,
							highlightcolor= "white",
							borderwidth=0)
nameEntered.pack()

button = tk.Button(win, text = "Predict",
						fg="white",
						bg="#077D3F",
						font="Calibri 20 bold",
						width=30, height=2,
						command = url_prediction,
						highlightthickness=0,
						borderwidth=0)
button.pack(padx=10, pady=0)

blank = tk.Text(win,font="Calibri 10",
					bg ="#F0F0F0",
					width=200,
					height=100,
					highlightthickness=2,
					highlightcolor= "white",
					borderwidth=0)
blank.pack(pady=10)

win.mainloop()   