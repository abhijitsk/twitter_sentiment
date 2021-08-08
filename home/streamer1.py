import pandas as pd 
import string
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import numpy as np 
from nltk import word_tokenize
from nltk.corpus import stopwords
import re
from wordcloud import WordCloud
from .dbMongo import SentimentAnalysis


dataset = pd.read_csv('finalSentimentdata2.csv')
data_for_pie = dataset['sentiment'].value_counts()
sa = SentimentAnalysis()
data_for_word = dataset['text'].apply(sa.without_stem)
complete_words = sa.generate_complete_text(data_for_word)










def get_graph():
    buffer  = BytesIO()
    plt.savefig(buffer, format = 'png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png)
    graph = graph.decode('utf-8')
    buffer.close()
    return graph


def plotting():
    x = np.array(data_for_pie.values)
    y = np.array(data_for_pie.index)
    plt.switch_backend('AGG')
    plt.pie(x, labels = y)
    graph = get_graph()
    return graph

def plot_wordcloud():

    wordCloud = WordCloud(width = 500, height = 300, random_state = 21, max_font_size = 119 ).generate(complete_words)
    plt.switch_backend('AGG')
    plt.imshow(wordCloud, interpolation = "bilinear") # need to explain this line of code
    plt.axis('off')
    plt.title('WORD CLOUD', size = 20, color = "blue")
    graph = get_graph()
        
    return graph




