import twint
import csv
import time

def search_tweets(tweet_id, username, min_rt, sincedate, untildate):
    all_tweet = []
    # set up configuration
    c = twint.Config()
    c.Store_object = True
    c.Hide_output = True
    c.Store_object_tweets_list = all_tweet

    # search for the tweet with the matching tweet ID
    c.Search = f"(from:{username}) (to:{username}) min_retweets:{min_rt} until:{untildate} since:{sincedate}"
    twint.run.Search(c)

    thread_tweet = []
    # retrieve the original tweet
    for data in all_tweet:
        if data.id_str == tweet_id:
            thread_tweet.append(data)
        else:
            pass

    # retrieve the tweet with the matching conversation ID
    for data in all_tweet:
        if data.conversation_id == tweet_id:
            thread_tweet.append(data)
        else:
            pass

    # sort the tweets from first to last
    thread_tweet.sort(key=lambda x: x.datetime)

    thread_text = []
    for data in thread_tweet:
        thread_text.append(data.tweet)

    # add period symbol to sentences that do not end with one
    for i in range(len(thread_text)):
        if not thread_text[i].endswith(('.', ',', '-', '?', '!', '~')):
            thread_text[i] += '.'

    # concatenate the sentences into a single string in a list
    result = ' '.join(thread_text)
    return result

def to_csv(filename, tweets):
    with open(filename, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(tweets)

# set the tweet IDs, usernames, min_rt, sincedate, and untildate to search for
tweet_params = [
    {"tweet_id": "1453281825661538304", "username": "dadimdum", "min_rt": 1, "sincedate": "2021-10-27", "untildate": "2021-10-29"},
    {"tweet_id": "1451786477491273731", "username": "maskhairulanam", "min_rt": 1, "sincedate": "2021-10-23", "untildate": "2021-10-25"},
    {"tweet_id": "1451574797092081671", "username": "voidotid", "min_rt": 1, "sincedate": "2021-10-22", "untildate": "2021-10-24"},
    {"tweet_id": "1451094548512325636", "username": "praditia_rio", "min_rt": 1, "sincedate": "2021-10-21", "untildate": "2021-10-23"},
    {"tweet_id": "1450845100003639297", "username": "voidotid", "min_rt": 1, "sincedate": "2021-10-20", "untildate": "2021-10-22"},     
    {"tweet_id": "1450827452796653578", "username": "hermionyyye", "min_rt": 1, "sincedate": "2021-10-20", "untildate": "2021-10-22"},
    {"tweet_id": "1450486919301763081", "username": "voidotid", "min_rt": 1, "sincedate": "2021-10-19", "untildate": "2021-10-21"},
    {"tweet_id": "1450462723884687362", "username": "om_angger", "min_rt": 1, "sincedate": "2021-10-19", "untildate": "2021-10-21"},
    {"tweet_id": "1450281746549731329", "username": "AdjieSanPutro", "min_rt": 1, "sincedate": "2021-10-19", "untildate": "2021-10-21"},
    {"tweet_id": "1450029663724601347", "username": "SejarahUlama", "min_rt": 1, "sincedate": "2021-10-18", "untildate": "2021-10-20"},
    {"tweet_id": "1449707305411354624", "username": "hermionyyye", "min_rt": 1, "sincedate": "2021-10-17", "untildate": "2021-10-19"},
    {"tweet_id": "1449667861488672772", "username": "LBHBandung", "min_rt": 0, "sincedate": "2021-10-17", "untildate": "2021-10-19"},
    {"tweet_id": "1448523296136318981", "username": "detikcom", "min_rt": 1, "sincedate": "2021-10-14", "untildate": "2021-10-16"},
    {"tweet_id": "1447906987769954305", "username": "angga_fzn", "min_rt": 1, "sincedate": "2021-10-12", "untildate": "2021-10-14"},
    {"tweet_id": "1447905241714741252", "username": "dystinnn", "min_rt": 1, "sincedate": "2021-10-12", "untildate": "2021-10-14"},
    {"tweet_id": "1447088590765236228", "username": "noctekv", "min_rt": 1, "sincedate": "2021-10-10", "untildate": "2021-10-12"},
    {"tweet_id": "1446762470987599873", "username": "AdjieSanPutro", "min_rt": 1, "sincedate": "2021-10-09", "untildate": "2021-10-11"},
    {"tweet_id": "1446351338242129923", "username": "AJIIndonesia", "min_rt": 1, "sincedate": "2021-10-08", "untildate": "2021-10-10"},
    {"tweet_id": "1446073275193249792", "username": "theflankerID", "min_rt": 1, "sincedate": "2021-10-07", "untildate": "2021-10-09"},
    {"tweet_id": "1444984101895741448", "username": "barangshopee_", "min_rt": 1, "sincedate": "2021-10-04", "untildate": "2021-10-06"},
    {"tweet_id": "1444872138364571652", "username": "zoenightshaade", "min_rt": 1, "sincedate": "2021-10-04", "untildate": "2021-10-06"},
    {"tweet_id": "1444672802737430532", "username": "kucinggendut__", "min_rt": 1, "sincedate": "2021-10-03", "untildate": "2021-10-05"},
    {"tweet_id": "1444660982870020100", "username": "rafenditya", "min_rt": 1, "sincedate": "2021-10-03", "untildate": "2021-10-05"},
    {"tweet_id": "1443964644675112962", "username": "SejarahUlama", "min_rt": 1, "sincedate": "2021-10-01", "untildate": "2021-10-03"},
    # {"tweet_id": "1394800463699013632", "username": "angga_fzn", "min_rt": 1, "sincedate": "2021-10-18", "untildate": "2021-10-20"},
    # {"tweet_id": "1394518851212091395", "username": "angga_fzn", "min_rt": 1, "sincedate": "2021-10-18", "untildate": "2021-10-20"},
    # {"tweet_id": "1394508585384562689", "username": "faridgaban", "min_rt": 1, "sincedate": "2021-10-18", "untildate": "2021-10-20"},
    # {"tweet_id": "1394428714037907458", "username": "angga_fzn", "min_rt": 1, "sincedate": "2021-10-17", "untildate": "2021-10-19"},
    # {"tweet_id": "1394238334566760448", "username": "angga_fzn", "min_rt": 1, "sincedate": "2021-10-17", "untildate": "2021-10-19"},
    # {"tweet_id": "1394157117586632705", "username": "angga_fzn", "min_rt": 1, "sincedate": "2021-10-17", "untildate": "2021-10-19"},
    # {"tweet_id": "1393842470899748867", "username": "angga_fzn", "min_rt": 1, "sincedate": "2021-10-16", "untildate": "2021-10-18"},
    # {"tweet_id": "1393740824882544641", "username": "angga_fzn", "min_rt": 1, "sincedate": "2021-10-16", "untildate": "2021-10-18"},
    # {"tweet_id": "1393635296772714497", "username": "tirta_cipeng", "min_rt": 1, "sincedate": "2021-10-15", "untildate": "2021-10-17"},
    # {"tweet_id": "1393174538687451139", "username": "angga_fzn", "min_rt": 1, "sincedate": "2021-10-14", "untildate": "2021-10-16"},
    # {"tweet_id": "1392049125420531716", "username": "tatakhoiriyah", "min_rt": 1, "sincedate": "2021-10-14", "untildate": "2021-10-16"},

    ]

# search for tweets and store the longest sequence of sentences for each tweet
for tweet_param in tweet_params:
    longest = ""

    for i in range(7):
        result = search_tweets(tweet_param["tweet_id"], tweet_param["username"], tweet_param["min_rt"], tweet_param["sincedate"], tweet_param["untildate"])
        # longest = max(result, longest)
        if len(result) == 0 or len(result) < len(longest):
            pass
        elif len(result) >= len(longest):
            longest = result
        time.sleep(1)

    print(longest)
    to_csv('indo_twitter_thread.csv', [[longest]]) # store the longest sequence of sentences