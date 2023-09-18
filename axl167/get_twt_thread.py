import tweepy
import json
import config
from re import search

# replace these with your own consumer key, consumer secret, access token, and access token secret
consumer_key = config.api_key
consumer_secret = config.api_key_secret
access_token = config.access_token
access_token_secret = config.access_token_secret

# authenticate with Twitter
auth = tweepy.OAuthHandler(consumer_key, consumer_secret, access_token, access_token_secret)
api = tweepy.API(auth)


# first function: get the 200 newest tweet from the user specified
def get_all_tweets(tweet):
    screen_name = tweet.user.screen_name
    last_tweet_id = tweet.id

    # create an empty list for the thread
    tweet_thread = []

    # make initial request for most recent tweets (200 is the maximum allowed count)
    new_tweets = api.user_timeline(screen_name=screen_name, count=200)
    tweet_thread.extend(new_tweets)

    # save the ID of the oldest tweet in the thread minus one
    oldest = tweet_thread[-1].id - 1
    print(tweet_thread[-1].id)

    # keep grabbing tweets until there are no tweets left to grab
    while len(new_tweets) > 0 and oldest >= last_tweet_id:
        # print(f"getting tweets before {oldest}")

        # all subsequent requests use the max_id param to prevent duplicates
        new_tweets = api.user_timeline(screen_name=screen_name, count=200, max_id=oldest)

        # save most recent tweets
        tweet_thread.extend(new_tweets)

        # update the id of the oldest tweet less one
        oldest = tweet_thread[-1].id - 1

    all_tweets_in_thread_id = [tweet.id for tweet in tweet_thread]
    return all_tweets_in_thread_id


# second function: to get all tweets after the ID specified which is part of the thread
def get_tweets_after(tweet_id):
    # create a list to store the tweets after
    tweets_after = []
    # hasReply = True

    # create response variable to get the status of each of the after tweets
    response = api.get_status(tweet_id, tweet_mode='extended')
    tweets_until_thread = get_all_tweets(response)

    # append the tweet to list of tweet after
    tweets_after.append(response)

    # check if the tweet is available to retrieve
    if tweets_until_thread[-1] > response.id:
        print("Not able to retrieve old tweets")
        return tweets_after

    # create variable start index to find the starting tweets after
    starting_index = tweets_until_thread.index(response.id)

    is_max_tweet = 0
    quietLong = 0

    while starting_index != 0 and is_max_tweet < 25:
        current_index = starting_index - 1
        current_tweet = api.get_status(tweets_until_thread[current_index], tweet_mode='extended')

        # if current tweet has the same reply ID to the tweet before (the one listed in tweets_after)
        if current_tweet.in_reply_to_status_id == tweets_after[-1].id:
            # quietLong = 0
            tweets_after.append(current_tweet)
        # BELOM NGERTI!!!!
        else:
            quietLong = quietLong + 1

        starting_index = current_index
    return tweets_after

# fourth function: get all tweets in the thread
def get_all_tweets_in_thread(tweet_id):
    # create a list to store all tweets in the thread
    tweet_thread = []
    # get the tweets after first tweet
    tweet_thread.extend(get_tweets_after(tweet_id))
    # check the length of tweet_thread
    if len(tweet_thread) >= 3:
        return tweet_thread
    else:
        print("This summarizer only works if there are 3 or more tweets in the thread!")

def get_thread(url):
    match = search('https://twitter.com/(.+)/status/([0-9]+)', url)
    username = match.group(1)
    tweet_id = match.group(2)
    if match != None:
        all_tweets_in_thread = get_all_tweets_in_thread(tweet_id)
        return all_tweets_in_thread

# fifth function: print the tweet to terminal
def get_text(tweets):
    tweet_text = []
    if len(tweets) > 0:
        # print("Thread Messages include:-")
        for tweet_id in range(len(tweets)):
            tweet_text.append(tweets[tweet_id].full_text)
            # print(tweets[tweet_id].full_text)

        return tweet_text
            # print(tweet_text)
            # print("")
    else:
        print("No Tweets to print")

if __name__ == '__main__':
    url = 'https://twitter.com/alhrkn/status/1474925007297921026'

    all = get_thread(url)
    get_text(all)