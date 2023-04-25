import csv
import re
from nltk.stem.porter import *
stemmer = PorterStemmer()
from nltk.corpus import stopwords
st = stopwords.words('english')

def preprocess(tweet):
    tweet = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '_url_', tweet)
    tweet = re.sub('(@[A-Za-z0-9\_]+)', '_username_', tweet)
    tweet = tweet.lower()
    tweet = re.sub(r'(coronavirus|corona|covid\W*19)', 'covid', tweet)
    tweet = re.sub(r'\W', " ", tweet)
    tweet = re.sub(r'\d+', '_number_', tweet)
    tweet = re.sub(r'\s{2,}', " ", tweet)
    #tweet = ' '.join([w for w in tweet.split() if not w in st])
    tweet = ' '.join([stemmer.stem(w) for w in tweet.split()])
    tweet = tweet.strip()
    return tweet

weka_outfile = open('covid_diagnosis_tweets_training.arff','w', encoding='UTF-8')
weka_outfile.write('@relation covid_diagnosis\n')
# weka_outfile.write('@attribute tweet_id string\n')
# weka_outfile.write('@attribute user_id string\n')
weka_outfile.write('@attribute text string\n')
# weka_outfile.write('@attribute created_at string\n')
weka_outfile.write('@attribute class_ {0,1}\n\n')
weka_outfile.write('@data\n')

with open('covid_diagnosis_tweets_training.tsv', 'r', encoding='UTF-8') as filename_input:
    filereader = csv.reader(filename_input, delimiter='\t')
    for row in filereader:
        tweet_id = '\"'+ row[0] +'\"'
        user_id = '\"'+row[1]+'\"'
        text = '\"'+preprocess(row[2])+'\"'
        # text = '\"' + re.sub('\"', '_qt_', tweet) + '\"'
        created_at = '\"' + row[3] + '\"'
        label = row[4]

        if not re.search(r'^\"text\"$', text):
            print(tweet_id + '\t' + user_id + '\t' + text + '\t' + created_at + '\t' + '\t' + label)
            weka_outfile.write(text + ',' + label + '\n')

weka_outfile.close()