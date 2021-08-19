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


''' dataset = pd.read_csv('finalSentimentdata2.csv')
data_for_pie = dataset['sentiment'].value_counts()
sa = SentimentAnalysis()
data_for_word = dataset['text'].apply(sa.without_stem)
complete_words = sa.generate_complete_text(data_for_word)'''

#training_data =  pd.read_csv('test_data_senti.csv')





'''
def process_forKeras(predictions):
    check_label = ['anger','fear','joy','sad']
    predictions_list = [check_label[np.argmax(text)] for text in predictions]
    


    return predictions_list

'''




def get_graph():
    buffer  = BytesIO()
    plt.savefig(buffer, format = 'png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png)
    graph = graph.decode('utf-8')
    buffer.close()
    return graph


def plotting(predictions):
    data_for_cloud = np.unique(predictions, return_counts =True)
    x = data_for_cloud[1].astype(np.int)
    y = data_for_cloud[0]
    
    

    plt.switch_backend('AGG')
    plt.pie(x, autopct = '%1.0f%%')
    plt.legend(y)
    graph = get_graph()
    return graph

def plot_wordcloud(complete_words):

    wordCloud = WordCloud(width = 500, height = 300, random_state = 21, max_font_size = 119 ).generate(complete_words)
    plt.switch_backend('AGG')
    plt.imshow(wordCloud, interpolation = "bilinear") # need to explain this line of code
    plt.axis('off')
    plt.title('WORD CLOUD', size = 20, color = "blue")
    graph = get_graph()
        
    return graph




