import pandas as pd 
import string
import matplotlib.pyplot as plt
import base64
from io import BytesIO

dataset = pd.read_csv('finalSentimentdata2.csv')


def get_graph():
    buffer  = BytesIO()
    plt.savefig(buffer, format = 'png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png)
    graph = graph.decode('utf-8')
    buffer.close()
    return graph


def plotting(x,y):
    plt.switch_backend('AGG')
    plt.plot(x,y)
    graph = get_graph()
    return graph
