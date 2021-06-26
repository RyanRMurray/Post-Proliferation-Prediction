import csv
import json
from datetime import datetime

def convert_to_epoch(t):
    utc= datetime.strptime(t, "%Y-%m-%dT%H:%M:%S.%fZ")
    return (utc - datetime(1970,1,1)).total_seconds()

def filter_collection():
    #get deleted tweet IDs
    to_delete = set()
    
    with open("./Data/To delete.txt", "r") as file:
        for l in file:
            to_delete.add(l[:-1])
    
    print(to_delete)
    #get json object
    with open("./Data/collection.json", "r") as file:
        collection = json.load(file)

    with open("./Data/collection_filtered.json", "w") as file:
        file.write("[\n")

        for tweet in collection:
            tweet["created_at"] = convert_to_epoch(tweet["created_at"])
            if tweet["id"] not in to_delete:
                file.write(json.dumps(tweet))
                file.write(",\n")
    
        file.write("]")

def merge_updates(u : int):
    fields = [
        "timestamp", "post_id", "retweets", "likes", "quotes"
    ]
    #get deleted tweet IDs
    to_delete = set()
    
    with open("./Data/To delete.txt", "r") as file:
        for l in file:
            to_delete.add(l[:-1])

    with open("./Data/updates-all.csv", "w", newline="") as outfile:
        writer = csv.DictWriter(outfile, fields)
        writer.writeheader()

        for i in range(1,u+1):
            print("Reading update file {} of {}".format(i,u))
            with open("./Data/updates-{}.json".format(str(i)), "r") as infile:
                updates = json.load(infile)
                for update in updates:
                    if update["id"] not in to_delete:
                        writer.writerow({
                            "timestamp" : update["timestamp"],
                            "post_id" : update["id"],
                            "retweets" : update["public_metrics"]["retweet_count"],
                            "likes" : update["public_metrics"]["like_count"],
                            "quotes" : update["public_metrics"]["quote_count"],
                        })