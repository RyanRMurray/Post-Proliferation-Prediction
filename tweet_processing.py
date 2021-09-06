import os
import json
import csv
from typing import List
from urllib.parse import urlparse
import re
from collections import defaultdict

# project default paramaters
tweet_files = [
    "1m_tweets_10_7_2021.json",
    "1m_tweets_11_7_2021.json",
    "1m_tweets_12_7_2021.json",
    "1m_tweets_13_7_2021.json",
    "1m_tweets_14_7_2021.json",
    "1m_tweets_15_7_2021.json",
    "1m_tweets_16_7_2021.json",
]

procced_tweets = [
    "./Processed/1m_tweets_10_7_2021.json",
    "./Processed/1m_tweets_11_7_2021.json",
    "./Processed/1m_tweets_12_7_2021.json",
    "./Processed/1m_tweets_13_7_2021.json",
    "./Processed/1m_tweets_14_7_2021.json",
    "./Processed/1m_tweets_15_7_2021.json",
    "./Processed/1m_tweets_16_7_2021.json",
]

update_files = [
    "./Data/1m_tweets_10_7_2021_updates.csv",
    "./Data/1m_tweets_11_7_2021_updates.csv",
    "./Data/1m_tweets_12_7_2021_updates.csv",
    "./Data/1m_tweets_13_7_2021_updates.csv",
    "./Data/1m_tweets_14_7_2021_updates.csv",
    "./Data/1m_tweets_15_7_2021_updates.csv",
    "./Data/1m_tweets_16_7_2021_updates.csv",
]


def preproc_text(tweet: json) -> str:
    text = tweet["text"]

    # replace urls
    if "entities" in tweet:
        if "urls" in tweet["entities"]:
            for url in tweet["entities"]["urls"]:
                domain = urlparse(url["expanded_url"]).netloc
                text = text.replace(url["url"], domain)

        # replace usernames
        if "mentions" in tweet["entities"]:
            for mention in tweet["entities"]["mentions"]:
                text = text.replace("@" + mention["username"], "<username>")

        # replace hashtags
        if "hashtags" in tweet["entities"]:
            for ht in tweet["entities"]["hashtags"]:
                text = text.replace("#" + ht["tag"], "<hashtag> " + ht["tag"].lower())

    # replace allcaps
    for allcap in set(re.findall("[A-Z]{2,}", text)):
        text = text.replace(allcap, "<allcaps> " + allcap.lower())

    # replace times
    text = re.sub(r"[0-9]{1,2}:[0-9]{2}", "<time>", text)

    # replace dates
    text = re.sub(r"[0-9]{1,4}(-|\/)[0-9]{1,2}((-|\/)[0-9]{1,4})?", "<date>", text)

    # replace numbers
    text = re.sub(r"-?[0-9][0-9,\.]+", "<number>", text)

    return text.lower()


def normalise_text(files: List[str]):
    word_corpus = defaultdict(int)
    file_counter = 1
    file_num = len(files)

    inputs = ["./Data/" + f for f in files]
    outputs = ["./Processed/" + f for f in files]

    for (infile, outfile) in zip(inputs, outputs):
        counter = 0
        with open(infile, "r") as infile:
            j = json.load(infile)

        tweet_num = len(j)

        with open(outfile, "w") as outfile:
            for tweet in j:
                processed_text = preproc_text(tweet)
                tweet["text"] = processed_text
                for word in re.findall(
                    r'\w+[!"#$%&\'()*+,\-./:;<=>?@[\]^_`{|}~]\w+|\<\w+\>|\w+|[!"#$%&\'()*+,\-./:;<=>?@[\]^_`{|}~]',
                    processed_text,
                ):
                    word_corpus[word] += 1

                counter += 1
                print(
                    "First Pass: Normalised {}/{} tweets for file {}/{}".format(
                        counter, tweet_num, file_counter, file_num
                    ),
                    end="\r",
                )
            json.dump(j, outfile)
        file_counter += 1

    print()
    print("First Pass: Complete")
    print("{} unique words in corpus.".format(len(word_corpus.keys())))

    # delete words with <10 occurrences
    file_counter = 1

    for outfile in outputs:
        counter = -0
        with open(outfile, "r+") as outfile:
            j = json.load(outfile)

            tweet_num = len(j)

            for tweet in j:
                tweet["text"] = " ".join(
                    map(
                        lambda t: "<unknown>" if (word_corpus[t] <= 10) else t,
                        tweet["text"].split(),
                    )
                )

                counter += 1
                print(
                    "Second Pass: Normalised {}/{} tweets for file {}/{}".format(
                        counter, tweet_num, file_counter, file_num
                    ),
                    end="\r",
                )

            outfile.seek(0)
            json.dump(j, outfile)
            outfile.truncate()

        file_counter += 1

    print()
    print("Second Pass: Complete")


def id_map(tweets: json):
    mapped = {}
    for t in tweets:
        mapped[t["id"]] = t

    return mapped


def build_data_set(tweets: List[str], updates: List[str], destination: str):
    # open update/files, save to file
    file_num = len(tweets)
    file_count = 0

    with open(destination, "w") as outfile:
        outfile.write("[\n")

        for (t, u) in zip(tweets, updates):
            file_count += 1
            print("Collecting data from file pair {}/{}".format(file_count, file_num))

            with open(t, "r") as infile_tweets, open(u, "r") as infile_updates:
                c_updates = csv.DictReader(infile_updates)
                j_tweets = id_map(json.load(infile_tweets))

                for update in c_updates:
                    final_metrics = {
                        "like_count": update["likes"],
                        "quote_count": update["quotes"],
                        "retweet_count": update["retweets"],
                    }
                    data = j_tweets[update["id"]]

                    data["initial_metrics"] = data["public_metrics"]
                    data["final_metrics"] = final_metrics

                    data.pop("public_metrics", None)
                    data.pop("entities", None)
                    data.pop("in_reply_to_user_id", None)

                    outfile.write(json.dumps(data))
                    outfile.write(",\n")

        outfile.write("]")

    print("Complete")
