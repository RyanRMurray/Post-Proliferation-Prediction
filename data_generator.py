import numpy as np
import random
import keras.utils.np_utils
from datetime import datetime
from nltk.tokenize import TweetTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from urllib.parse import urlparse

#details:
#   day-of-week of post,
#   time-of-day of post,
#   author account age,
#   author followers count,
#   author following count,
#   author tweet count,
#   author in list count,
#   author verified status
calc_detail_vector_size = lambda x : sum(range(x+1))
DETAIL_FEATURES = calc_detail_vector_size(8)
THRESHOLDS = [10, 100, 1000, 10000, 100000, 999999999999]
CATEGORIES      = 6
SEQ_LENGTH      = 356
IMAGE_SHAPE = (1)
DETAIL_MAXIMUMS = np.array([
        6,
        81181,
        628,
        73061888,
        2043341,
        39488624,
        210150,
        1,
        450186,
        3768,
        301304384,
        10216705,
        236931744,
        1260900,
        6,
        48864680,
        4294081024,
        4292340480,
        4294827008,
        4269498368,
        81150,
        4136565248,
        292197760,
        3396021760,
        36986400,
        187,
        4272793344,
        4294827776,
        4289864448,
        73061888,
        4293038080,
        3989597696,
        2043341,
        4256888832,
        6970388,
        210150
    ])

def symbolise(thing:str):
    if thing[0] == '@':
        return '<username>'
    
    if thing.replace('.','',1).isdigit():
        return '<number>'
    
    if urlparse(thing).netloc != '':
        return urlparse(thing).netloc
    
    return thing

class TrainingData():
    def __init__(self, t_data, u_data, truth):
        training_num = 0.8
        self.t_train, self.u_train, self.truth_train = np.empty(shape=(1,SEQ_LENGTH),dtype='uint32'),np.zeros(shape=(1,DETAIL_FEATURES), dtype='uint32'),np.empty(shape=(1,1))
        self.t_valid, self.u_valid, self.truth_valid = np.empty(shape=(1,SEQ_LENGTH),dtype='uint32'),np.zeros(shape=(1,DETAIL_FEATURES), dtype='uint32'),np.empty(shape=(1,1))

        #sort by category
        print('TrainingData: Sorting by category')
        t_samples = {k:[] for k in range(CATEGORIES)}
        u_samples = {k:[] for k in range(CATEGORIES)}

        for (t,u,tr) in zip(t_data,u_data,truth):
            t_samples[tr].append(t)
            u_samples[tr].append(u)

        #convert into arrays
        print('TrainingData: Converting to Arrays')
        for i in range(CATEGORIES):
            t_samples[i] = np.array(t_samples[i], dtype='uint32')
            u_samples[i] = np.array(u_samples[i], dtype='uint32')

        print('Normalising user data')
        #find max in each part of vector
        if DETAIL_MAXIMUMS is None:
            maximums = np.zeros((1,DETAIL_FEATURES), dtype='uint32')
            for i in range(CATEGORIES):
                maximums = [np.concatenate((maximums, u_samples[i])).max(axis=0)]
        else:
            maximums = DETAIL_MAXIMUMS
        #normalise
        for i in range(CATEGORIES):
            if len(u_samples[i]) > 0:
                u_samples[i] = u_samples[i] / maximums

        #split
        print('TrainingData: Splitting Train/Validate')
        for i in range(CATEGORIES):
            t_s = t_samples[i]
            u_s = u_samples[i]

            s = int(len(t_s)*training_num)
            (t_t,t_v) = np.split(t_s, [s])
            (u_t,u_v) = np.split(u_s, [s])

            if len(t_v) != 0:
                self.t_train = np.concatenate((self.t_train,t_t))
                self.u_train = np.concatenate((self.u_train,u_t))
                self.t_valid = np.concatenate((self.t_valid,t_v))
                self.u_valid = np.concatenate((self.u_valid,u_v))

                if len(t_t) > 0:
                    self.truth_train = np.concatenate((self.truth_train, np.array([[i]] * len(t_t))))
                self.truth_valid = np.concatenate((self.truth_valid, np.array([[i]] * len(t_v))))

        self.t_train = self.t_train[1:]
        self.u_train = self.u_train[1:]
        self.t_valid = self.t_valid[1:]
        self.u_valid = self.u_valid[1:]
        self.truth_train = self.truth_train[1:]
        self.truth_valid = self.truth_valid[1:]

        print("Category Train/Validation split is as follows:")
        t_cats, v_cats = {k:0 for k in range(CATEGORIES)}, {k:0 for k in range(CATEGORIES)},
        for c in self.truth_train:
            t_cats[c[0]] += 1
        for c in self.truth_valid:
            v_cats[c[0]] += 1

        print(t_cats)
        print(v_cats)

        #to categories
        print('TrainingData: Categorising truth values')
        self.truth_train = keras.utils.np_utils.to_categorical(self.truth_train, CATEGORIES)
        self.truth_valid = keras.utils.np_utils.to_categorical(self.truth_valid, CATEGORIES)

    def x_train(self):
        return [self.t_train, self.u_train]
    
    def y_train(self):
        return self.truth_train

    def x_valid(self):
        return [self.t_valid, self.u_valid]
    
    def y_valid(self):
        return self.truth_valid

    def all(self):
        return [
            np.concatenate([self.t_train, self.t_valid]),
            np.concatenate([self.u_train, self.u_valid]),
        ]

    def truth(self):
        return np.concatenate([self.truth_train,self.truth_valid])

#turns a tweet into an input. tokenizer is optional, in case data is already tokenized.
def tweet_to_training_pair(tweet):
    #get text
    text = tweet['text']

    #get user data
    posted = datetime.fromtimestamp(tweet['created_at'])
    user_data = list(map(int, [
        posted.weekday(),
        (posted - posted.replace(hour=0,minute=0,second=0)).seconds,
        (datetime.now() - datetime.strptime(tweet['author_data']['created_at'], '%Y-%m-%dT%H:%M:%S.000Z')).days / 30,
        tweet['author_data']['public_metrics']['followers_count'],
        tweet['author_data']['public_metrics']['following_count'],
        tweet['author_data']['public_metrics']['tweet_count'],
        tweet['author_data']['public_metrics']['listed_count'],
        tweet['author_data']['verified']
    ]))

    products = []
    for x in range(len(user_data)-1):
        for y in range(x+1, len(user_data)):
            products.append(user_data[x] * user_data[y])

    user_features = np.array(user_data+products, dtype='float32')

    #get ground truth for one-hot encoding
    #we'll use 7 categories for now, with the 6th being anything above 100k likes
    likes = int(tweet['final_metrics']['like_count'])
    i = 0
    while THRESHOLDS[i] < likes:
        i += 1

    return ((text, user_features), i)

def generate_training_data(data,tokenizer=None):
    i_data, t_data, u_data, truth = [], [], [], []
    splitter = TweetTokenizer(reduce_len=True, preserve_case=False)
    tweets = len(data)

    counter = 0
    #randomise order
    random.shuffle(data)
    for tweet in data:
        ((t,u),tr) = tweet_to_training_pair(tweet)

        t_data.append(t)
        u_data.append(u)
        truth.append(tr)
        counter +=1 
        print('Converted {}/{} tweets'.format(counter,tweets), end='\r')

    print()
    #tokenize
    print('Tokenizing...')
    t_data = tokenizer.texts_to_sequences(
        [
            [symbolise(s) for s in splitter.tokenize(t)]
            if len(splitter.tokenize(t)) > 0 else [0]
            for t in t_data
        ]
    )
    t_data = pad_sequences(t_data, maxlen=SEQ_LENGTH)
    print('Done.')

    return TrainingData(t_data, u_data, truth)


#Code beneath this line is depricated but kept for completeness
###############################################################################

def generate_training_data_old(data, image_directory,text_input_size,tokenizer=None):
    i_data, t_data, u_data, truth = [], [], [], []
    splitter = TweetTokenizer(reduce_len=True, preserve_case=False, strip_handles=True)
    tweets = len(data)

    counter = 0
    #randomise order
    random.shuffle(data)
    for tweet in data:
        ((_,t,u),tr) = tweet_to_training_pair(tweet,image_directory,text_input_size)

        #i_data.append(i)
        i_data.append([0])
        t_data.append(t)
        u_data.append(u)
        truth.append(tr)
        counter +=1 
        print('Converted {}/{} tweets'.format(counter,tweets), end='\r')

    print()
    #tokenize
    print('Tokenizing...')
    t_data = tokenizer.texts_to_sequences(
        [
            splitter.tokenize(t)
            for t in t_data
        ]
    )
    t_data = pad_sequences(t_data, maxlen=text_input_size)
    print('Done.')

    return TrainingData(i_data, t_data, u_data, truth)
