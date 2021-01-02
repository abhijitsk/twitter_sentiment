from django.shortcuts import render
from . import dbMongo

# Create your views here.

def home(request):
    tweetext = dbMongo.listForWeb
    return render(request,'home.html',{'tweet':tweetext})
