import os
import json
from urllib.parse import urlparse
import re

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


def main():
    print("testing")
    with open("./Data/200ktweets26062021.json", "r") as f:
        j = json.load(f)
        print("loaded")
        for (n,tweet) in zip(range(100), j):
            print("{}/100".format(n))
            print(tweet['text'])
            print("-->")
            print(preproc_text(tweet))

main()