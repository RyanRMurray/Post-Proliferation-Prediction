from typing import Collection
from searchtweets import ResultStream, load_credentials
from pathlib import Path
from collections import deque
import os
import json
import time
import requests
import zipfile


collections_path = Path("./Data/collection.json")
updates_path = Path("./Data/updates.json")

#collection parameters
header = {"Authorization": "Bearer {}".format(os.environ.get("BEARER_TOKEN"))}
collection_endpoint = "https://api.twitter.com/2/tweets/search/recent"
#number of tweets to collect
collection_target = 20_000
#update rate in seconds
update_rate = 30 * 60
#number of times to update
update_iterations = 240
#number of updates before compression
update_split = 10

def make_query(endpoint, header,query):
    result = requests.request("GET", endpoint, headers=header,params=query)
    if result.status_code != 200:
        raise Exception(result.status_code,result.text)
    return result.json()

def main():
    tweet_ids = set()

    #check if collection file exists
    #if so, record the ids of the tweets
    #if collections_path.is_file():
    #    for line in open(collections_path, "r"):
    #        tweet_ids.add(int(json.loads(line)["id"]))
    
    #if we dont have enough tweets, collect more
    #note: we can query recent 450 times every 15 minutes.
    update_start = time.time()
    query_start = time.time()
    newest_id = "0"
    
    with open("./Data/collection.json", "a") as file:
        file.write("[\n")
    
    #initial collection query: note no since_id field
    collection_query  = {
        #note: get every tweet tagged english. weird negated statement because API doesn't let you just ask for every tweet in recent
        "query": "lang:en the -the",
        "max_results": "100",
        "tweet.fields": "author_id,created_at,in_reply_to_user_id,entities,public_metrics"
    }
    while len(tweet_ids) < collection_target:
        try:
            time.sleep(2)
            result = make_query(collection_endpoint, header,collection_query)
        except Exception as e:
            #wait for next round of queries
            query_end = time.time()
            print("Rate limit exceeded!")
            print(e)
            elapsed  = query_end - query_start
            timeout = 60*16 - elapsed
            time.sleep(max(0,timeout))
            query_start = time.time()
        else:
            #update our newest tweet id and query
            if result["meta"]["result_count"] > 0:
                newest_id = result["meta"]["newest_id"]
                collection_query  = {
                    #note: get every tweet tagged english. weird negated statement because API doesn't let you just ask for every tweet in recent
                    "query": "lang:en the -the",
                    #ensures we dont poll the same tweets
                    "since_id": newest_id,
                    "max_results": "100",
                    "tweet.fields": "author_id,created_at,in_reply_to_user_id,entities,public_metrics"
                }
                #store our new tweets
                with open(collections_path, "a") as file:
                    for tweet in result["data"]:
                        file.write(json.dumps(tweet))
                        if len(tweet_ids) < collection_target or tweet is not result["data"][-1]:
                            file.write(",")
                        file.write("\n")
                        tweet_ids.add(int(tweet["id"]))
            print("Collected {} out of {} tweets".format(len(tweet_ids),collection_target))

    with open("./Data/collection.json", "a") as file:
        file.write("]")
    
    ids_ascending = [str(x) for x in sorted(tweet_ids)]
    #now that we have our tweets, wait for the update rate to elapse before beginning update step
    timeout = update_rate - (time.time() - update_start)
    print("Waiting {} seconds for next update".format(str(timeout)))
    time.sleep(max(0,timeout))

    zf = zipfile.ZipFile("./Data/collection.zip", "w")
    zf.write("./Data/collection.json", compress_type=zipfile.ZIP_DEFLATED)
    zf.close()
    os.remove("./Data/collection.json")

    #begin update iteration.
    day = 1
    with open("./Data/updates-{}.json".format(str(day)), "a") as file:
        file.write("[\n")
    #for this step we need the likes, rt, comment count of each tweet
    #we also need to purge any tweets that are deleted in this time in accordance with the twitter API
    deleted_ids = []
    for u in range(1,update_iterations+1):
        update_start = time.time()
        to_update = deque(ids_ascending)
        
        #note, we can make 300 requests for 100 tweets each every 15 minutes. Since we are polling for
        #less than 30_000 tweets, we dont need to worry about hitting the limit
        while len(to_update) > 0:
            batch = ",".join([to_update.popleft() for _ in range(0,min(100, len(to_update)))])
            update_query = {
                "ids":batch,
                "tweet.fields":"public_metrics"
            }
            timestamp = time.time()
            result = make_query("https://api.twitter.com/2/tweets", header, update_query)

            #record updates
            with open("./Data/updates-{}.json".format(str(day)), "a") as file:
                for tweet in result["data"]:
                    update = {
                        "timestamp" : str(timestamp),
                        "id"        : tweet["id"],
                        "public_metrics" : tweet["public_metrics"]
                    }
                    file.write(json.dumps(update))
                    if (u == 0) or (u % update_split != 0) or tweet is not result["data"][-1]:
                        file.write(",")
                    file.write("\n")

            #record and delete unavailable tweets
            if "errors" in result:
                for removed in result["errors"]:
                    deleted_ids.append(removed["value"])
                    ids_ascending.remove(removed["value"])

        print("Finished update {} of {}".format(u,update_iterations))

        if u > 0 and (u % update_split == 0):
            print("Compressing")
            with open("./Data/updates-{}.json".format(str(day)), "a") as file:
                file.write("]")
            zf = zipfile.ZipFile("./Data/updates-{}.zip".format(str(day)), "w")
            zf.write("./Data/updates-{}.json".format(str(day)), compress_type=zipfile.ZIP_DEFLATED)
            zf.close()
            os.remove("./Data/updates-{}.json".format(str(day)))
            day += 1
            with open("./Data/updates-{}.json".format(str(day)), "a") as file:
                file.write("[\n")

        if(u < update_iterations):
            #wait for next update
            update_end = time.time()
            timeout = update_rate - (update_end - update_start)
            print("Waiting {} seconds for next update".format(str(timeout)))
            time.sleep(timeout)
    

    with open("./Data/To delete.txt", "a") as file:
        for i in deleted_ids:
            file.write(i)
            file.write("\n")


    
main()