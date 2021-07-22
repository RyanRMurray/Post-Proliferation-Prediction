import os
import json
from urllib.parse import urlparse
import re
from collections import defaultdict

def preproc_text(tweet : json) -> str:
    text = tweet['text']

    #replace urls
    if 'entities' in tweet:
        if 'urls' in tweet['entities']:
            for url in tweet['entities']['urls']:
                domain = urlparse(url['expanded_url']).netloc
                text = text.replace(url['url'], domain)

        #replace usernames
        if 'mentions' in tweet['entities']:
            for mention in tweet['entities']['mentions']:
                text = text.replace('@'+mention['username'], '<username>')

        #replace hashtags    
        if 'hashtags' in tweet['entities']:
            for ht in tweet['entities']['hashtags']:
                text = text.replace('#'+ht['tag'], '<hashtag> ' + ht['tag'].lower())

    #replace allcaps
    for allcap in set(re.findall('[A-Z]{2,}', text)):
        text = text.replace(allcap, '<allcaps> ' + allcap.lower())

    #replace times
    text = re.sub("[0-9]{1,2}:[0-9]{2}", '<time>', text)

    #replace dates
    text = re.sub("[0-9]{1,4}(-|\/)[0-9]{1,2}((-|\/)[0-9]{1,4})?", "<date>", text)

    #replace numbers
    text = re.sub("-?[0-9][0-9,\.]+", '<number>', text)

    return text.lower()


def generate_normalised(paths : [str], output : str):
    word_corpus = defaultdict(int)
    file_counter = 1
    file_num = len(paths)

    with open('temp_corpus.json', 'w') as outfile:
        outfile.write("[\n")
        for path in paths:
            counter = 0
            with open(path, "r") as infile:
                j = json.load(infile)
            
            tweet_num = len(j)
            
            for tweet in j:
                processed_text = preproc_text(tweet)
                outfile.write('{{"id": "{}", "text": {}}},\n'.format(tweet['id'], json.dumps(processed_text)))
                for word in re.split("\s+", processed_text):
                    word_corpus[word] += 1

                counter += 1
                print("First Pass: Normalised {}/{} tweets for file {}/{}".format(counter, tweet_num, file_counter, file_num), end='\r')
                
        outfile.write("]")
    print()
    print("First Pass: Complete")
    print("{} unique words in corpus.".format(len(word_corpus.keys())))

    #todo, delete words with <10 occurences, save final file
    
    



def main():
    print("testing")
    generate_normalised(["./Data/200ktweets26062021.json"], "eggs")
    print("complete")

main()