from typing import Collection, List
from searchtweets import ResultStream, load_credentials
from pathlib import Path
from collections import deque
import os
import json
import time
import requests
import zipfile
import csv
import urllib
from PIL import Image
from datetime import datetime

collections_path = Path("./Data/collection.json")
updates_path = Path("./Data/updates.json")

# collection parameters
header = {"Authorization": "Bearer {}".format(os.environ.get("BEARER_TOKEN"))}
collection_endpoint = "https://api.twitter.com/2/tweets/search/recent"
# number of tweets to collect
collection_target = 20_000
# update rate in seconds
update_rate = 30 * 60
# number of times to update
update_iterations = 240
# number of updates before compression
update_split = 10


def make_query(endpoint, header, query):
    result = requests.request("GET", endpoint, headers=header, params=query)
    if result.status_code != 200:
        raise Exception(result.status_code, result.text)
    return result.json()


def convert_to_epoch(t):
    utc = datetime.strptime(t, "%Y-%m-%dT%H:%M:%S.%fZ")
    return (utc - datetime(1970, 1, 1)).total_seconds()


# get a number of pure (non-retweet) tweets
def get_tweets(target: int, outfile: str):
    query = {
        # Twitter does not have a command for general queries, so we use 'the -the',
        # which ends up not filtering anything
        "query": "lang:en -is:retweet the -the",
        "max_results": "100",
        # collect a wide net of data in case we develop more elaborate approaches later
        "tweet.fields": "author_id,created_at,entities,public_metrics,attachments,context_annotations",
    }
    query_start = time.time()
    newest_id = 0
    collected = 0

    with open(outfile, "w") as file:
        file.write("[\n")

        while collected < target:
            try:
                result = make_query(collection_endpoint, header, query)
            except Exception as e:
                # wait for next round of queries
                query_end = time.time()
                print("Rate limit exceeded!")
                print(e)
                elapsed = query_end - query_start
                timeout = 60 * 16 - elapsed
                time.sleep(max(0, timeout))
                query_start = time.time()
            else:
                # see if we got new tweets
                if result["meta"]["result_count"] > 0:
                    collected += int(result["meta"]["result_count"])
                    newest_id = result["meta"]["newest_id"]

                    for tweet in result["data"]:
                        tweet["created_at"] = convert_to_epoch(tweet["created_at"])
                        file.write(json.dumps(tweet))
                        # add delimiter
                        if collected < target or tweet is not result["data"][-1]:
                            file.write(",\n")

                # update query so we dont save the same tweets
                query = {
                    "query": "lang:en -is:retweet the -the",
                    "max_results": "100",
                    "since_id": newest_id,
                    "tweet.fields": "author_id,created_at,in_reply_to_user_id,entities,public_metrics,attachments,context_annotations,reply_settings",
                }

                print("Collected {} out of {} tweets".format(collected, target))

                # slight time delay to allow for more tweets
                time.sleep(10)

        file.write("\n]")


# get updates for tweets
def updates_for(tweets: str, to_file: str):
    fields = ["timestamp", "id", "likes", "retweets", "quotes"]

    # get every id to update
    with open(tweets, "r") as infile:
        ids_buffer = deque([t["id"] for t in json.load(infile)])

    with open(to_file, "w") as outfile:
        w = csv.DictWriter(outfile, fields)
        w.writeheader()

    while len(ids_buffer) > 0:
        update_start = time.time()

        batch = ",".join(
            [ids_buffer.popleft() for _ in range(0, min(100, len(ids_buffer)))]
        )
        query = {"ids": batch, "tweet.fields": "public_metrics"}
        timestamp = time.time()
        try:
            result = make_query("https://api.twitter.com/2/tweets", header, query)
        except:
            print("Limit rate exceeded!")
            elasped = time.time() - update_start
            timeout = 60 * 16 - elasped
            time.sleep(max(0, timeout))
            update_start = time.time()
            # redo request now that timeout has ended
            result = make_query("https://api.twitter.com/2/tweets", header, query)

        with open(to_file, "a", newline="") as outfile:
            writer = csv.DictWriter(outfile, fields)
            for tweet in result["data"]:
                writer.writerow(
                    {
                        "timestamp": timestamp,
                        "id": tweet["id"],
                        "likes": tweet["public_metrics"]["like_count"],
                        "retweets": tweet["public_metrics"]["retweet_count"],
                        "quotes": tweet["public_metrics"]["quote_count"],
                    }
                )

        if "errors" in result:
            with open("./Data/to_delete.txt", "a") as td:
                for removed in result["errors"]:
                    td.write(removed["value"])
                    td.write("\n")

    print("Completed update!")


def retrieve_images(tweets: str, directory: str):
    # get only tweets with media keys
    with open(tweets, "r") as infile:
        ids_buffer = deque(
            [
                t["id"]
                for t in json.load(infile)
                if ("attachments" in t and "media_keys" in t["attachments"])
            ]
        )
    update_start = time.time()
    while len(ids_buffer) > 0:
        batch = ",".join(
            [ids_buffer.popleft() for _ in range(0, min(100, len(ids_buffer)))]
        )
        query = {
            "ids": batch,
            "expansions": "attachments.media_keys",
            "media.fields": "type,url",
        }

        try:
            result = make_query("https://api.twitter.com/2/tweets", header, query)
        except:
            print("Limit rate exceeded!")
            elasped = time.time() - update_start
            timeout = 60 * 16 - elasped
            time.sleep(max(0, timeout))
            update_start = time.time()
            # redo request now that timeout has ended
            result = make_query("https://api.twitter.com/2/tweets", header, query)

        # first, associate media keys with tweets
        keys_to_ids = {}
        for tweet in result["data"]:
            if "attachments" in tweet:
                for k in tweet["attachments"]["media_keys"]:
                    keys_to_ids[k] = tweet["id"]

        # now we can retrieve images. Images are resized to 299x299 for resnet compatibility.
        for media in result["includes"]["media"]:
            if media["type"] == "photo":
                # only get the first image a tweet contains
                if not (
                    os.path.isfile(
                        "./"
                        + directory
                        + "/"
                        + keys_to_ids[media["media_key"]]
                        + ".jpg"
                    )
                ):
                    try:
                        img = Image.open(requests.get(media["url"], stream=True).raw)
                        resized = img.resize((299, 299), Image.ANTIALIAS)
                        if resized.mode in ("RGBA", "P"):
                            resized = resized.convert("RGB")
                        resized.save(
                            "./"
                            + directory
                            + "/"
                            + keys_to_ids[media["media_key"]]
                            + ".jpg"
                        )
                    except:
                        print(
                            "Couldn't retrieve image for {}".format(
                                keys_to_ids[media["media_key"]]
                            )
                        )
        print("{} tweets remain in queue".format(len(ids_buffer)))

    print("Done!")


# collect author details and add to records. Duplicated data, but simplifies input process later.
def author_details(directories: List[str]):
    file_count = len(directories)
    file_num = 0

    query_start = time.time()
    for d in directories:
        file_num += 1
        with open(d, "r+") as outfile:
            data = json.load(outfile)
            author_data = {}
            to_request = set()

            # get all unique authors
            for tweet in data:
                to_request.add(tweet["author_id"])

            # generate request queue
            author_count = len(to_request)
            author_num = 0
            queue = deque(to_request)

            # record data
            while len(queue) > 0:
                batch = ",".join(
                    [queue.popleft() for _ in range(0, min(100, len(queue)))]
                )
                query = {
                    "ids": batch,
                    "user.fields": "created_at,public_metrics,verified",
                }
                try:
                    result = make_query(
                        "https://api.twitter.com/2/users", header, query
                    )
                except:
                    print("Limit rate exceeded!")
                    elasped = time.time() - query_start
                    timeout = 60 * 16 - elasped
                    time.sleep(max(0, timeout))
                    query_start = time.time()
                    # redo request now that timeout has ended
                    result = make_query(
                        "https://api.twitter.com/2/users", header, query
                    )

                author_num += len(result["data"])
                for details in result["data"]:
                    author_data[details["id"]] = details

                print(
                    "Retrieved data for {}/{} authors in file {}/{}".format(
                        author_num, author_count, file_num, file_count
                    ),
                    end="\r",
                )

            print()
            print("Collected Author Data. Deleting authorless tweets and saving.")
            # delete tweets from deleted accounts
            data = [tweet for tweet in data if tweet["author_id"] in author_data]

            # add author data
            for tweet in data:
                tweet["author_data"] = author_data[tweet["author_id"]]

            outfile.seek(0)
            json.dump(data, outfile)
            outfile.truncate()
    print("Author collection complete!")


def collection_1():
    tweet_ids = set()

    # check if collection file exists
    # if so, record the ids of the tweets
    # if collections_path.is_file():
    #    for line in open(collections_path, "r"):
    #        tweet_ids.add(int(json.loads(line)["id"]))

    # if we dont have enough tweets, collect more
    # note: we can query recent 450 times every 15 minutes.
    update_start = time.time()
    query_start = time.time()
    newest_id = "0"

    with open("./Data/collection.json", "a") as file:
        file.write("[\n")

    # initial collection query: note no since_id field
    collection_query = {
        # note: get every tweet tagged english. weird negated statement because API doesn't let you just ask for every tweet in recent
        "query": "lang:en the -the",
        "max_results": "100",
        "tweet.fields": "author_id,created_at,in_reply_to_user_id,entities,public_metrics",
    }
    while len(tweet_ids) < collection_target:
        try:
            time.sleep(2)
            result = make_query(collection_endpoint, header, collection_query)
        except Exception as e:
            # wait for next round of queries
            query_end = time.time()
            print("Rate limit exceeded!")
            print(e)
            elapsed = query_end - query_start
            timeout = 60 * 16 - elapsed
            time.sleep(max(0, timeout))
            query_start = time.time()
        else:
            # update our newest tweet id and query
            if result["meta"]["result_count"] > 0:
                newest_id = result["meta"]["newest_id"]
                collection_query = {
                    # note: get every tweet tagged english. weird negated statement because API doesn't let you just ask for every tweet in recent
                    "query": "lang:en the -the",
                    # ensures we dont poll the same tweets
                    "since_id": newest_id,
                    "max_results": "100",
                    "tweet.fields": "author_id,created_at,in_reply_to_user_id,entities,public_metrics",
                }
                # store our new tweets
                with open(collections_path, "a") as file:
                    for tweet in result["data"]:
                        file.write(json.dumps(tweet))
                        if (
                            len(tweet_ids) < collection_target
                            or tweet is not result["data"][-1]
                        ):
                            file.write(",")
                        file.write("\n")
                        tweet_ids.add(int(tweet["id"]))
            print(
                "Collected {} out of {} tweets".format(
                    len(tweet_ids), collection_target
                )
            )

    with open("./Data/collection.json", "a") as file:
        file.write("]")

    ids_ascending = [str(x) for x in sorted(tweet_ids)]
    # now that we have our tweets, wait for the update rate to elapse before beginning update step
    timeout = update_rate - (time.time() - update_start)
    print("Waiting {} seconds for next update".format(str(timeout)))
    time.sleep(max(0, timeout))

    zf = zipfile.ZipFile("./Data/collection.zip", "w")
    zf.write("./Data/collection.json", compress_type=zipfile.ZIP_DEFLATED)
    zf.close()
    os.remove("./Data/collection.json")

    # begin update iteration.
    day = 1
    with open("./Data/updates-{}.json".format(str(day)), "a") as file:
        file.write("[\n")
    # for this step we need the likes, rt, comment count of each tweet
    # we also need to purge any tweets that are deleted in this time in accordance with the twitter API
    deleted_ids = []
    for u in range(1, update_iterations + 1):
        update_start = time.time()
        to_update = deque(ids_ascending)

        while len(to_update) > 0:
            batch = ",".join(
                [to_update.popleft() for _ in range(0, min(100, len(to_update)))]
            )
            update_query = {"ids": batch, "tweet.fields": "public_metrics"}
            timestamp = time.time()
            result = make_query(
                "https://api.twitter.com/2/tweets", header, update_query
            )

            # record updates
            with open("./Data/updates-{}.json".format(str(day)), "a") as file:
                for tweet in result["data"]:
                    update = {
                        "timestamp": str(timestamp),
                        "id": tweet["id"],
                        "public_metrics": tweet["public_metrics"],
                    }
                    file.write(json.dumps(update))
                    if (
                        (u == 0)
                        or (u % update_split != 0)
                        or tweet is not result["data"][-1]
                    ):
                        file.write(",")
                    file.write("\n")

            # record and delete unavailable tweets
            if "errors" in result:
                for removed in result["errors"]:
                    deleted_ids.append(removed["value"])
                    ids_ascending.remove(removed["value"])

        print("Finished update {} of {}".format(u, update_iterations))

        if u > 0 and (u % update_split == 0):
            print("Compressing")
            with open("./Data/updates-{}.json".format(str(day)), "a") as file:
                file.write("]")
            zf = zipfile.ZipFile("./Data/updates-{}.zip".format(str(day)), "w")
            zf.write(
                "./Data/updates-{}.json".format(str(day)),
                compress_type=zipfile.ZIP_DEFLATED,
            )
            zf.close()
            os.remove("./Data/updates-{}.json".format(str(day)))
            day += 1
            with open("./Data/updates-{}.json".format(str(day)), "a") as file:
                file.write("[\n")

        if u < update_iterations:
            # wait for next update
            update_end = time.time()
            timeout = update_rate - (update_end - update_start)
            print("Waiting {} seconds for next update".format(str(timeout)))
            time.sleep(timeout)

    with open("./Data/To delete.txt", "a") as file:
        for i in deleted_ids:
            file.write(i)
            file.write("\n")
