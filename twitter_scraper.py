from typing import Collection
from searchtweets import ResultStream, load_credentials
from pathlib import Path
import os
import json
import time
import requests

#collection parameters
collection_target = 210
collection_query  = {"query": "the OR i OR to OR a OR and OR is OR in OR has:media", "tweet.fields": "author_id,created_at,in_reply_to_user_id,entities,public_metrics"}
collection_header = {"Authorization": "Bearer {}".format(os.environ.get("BEARER_TOKEN"))}

def make_query(header,query):
    result = requests.request("GET", "https://api.twitter.com/2/tweets/search/recent", headers=header,params=query)
    if result.status_code != 200:
        raise Exception(result.status_code,result.text)
    return result.json()

def main():
    tweet_ids = set()

    #check if collection file exists
    #if so, record the ids of the tweets
    collections_path = Path("./Data/collection.json")

    if collections_path.is_file():
        start = 0
        with open(collections_path, "r") as file:
            collection = json.loads(file.read())

        for tweet in collection["tweets"]:
            tweet_ids.add(tweet["id"])
    else:
        collection = json.loads('{"tweets":[]}')
    
    #if we dont have enough tweets, collect more
    #note: we can query recent 450 times every 15 minutes.
    query_start = time.time()
    while len(tweet_ids) < collection_target:

        try:
            result = make_query(collection_header,collection_query)
        except Exception as e:
            #wait for next round of queries
            query_end = time.time()
            print("Rate limit exceeded!")
            elapsed  = query_end - query_start
            timeout = 60*16 - elapsed
            time.sleep(timeout)
            query_start = time.time()
        else:
            for tweet in result["data"]:
                if tweet["id"] not in tweet_ids:
                    tweet_ids.add(tweet["id"])
                    collection["tweets"].append(tweet)
            print("Collected {} out of {} tweets".format(len(tweet_ids,collection_target)))
    
    with open(collections_path, "w") as file:
        file.write(json.dumps(collection))


    print("Test done!")


    
main()