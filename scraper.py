import snscrape.modules.twitter as sntwitter
import pandas as pd


def scrape(query):
    # Creating a list to append all tweet attributes(data)
    tweets = []
    fn = query.split(" ")[0] + "_100K.csv"
    # prepare the query
    q = sntwitter.TwitterSearchScraper(query, top=True)
    print(f"Working on {fn} with query '{query}' (top = True)")

    # Using TwitterSearchScraper to scrape data and append tweets to list
    for i, tweet in enumerate(q.get_items()):
        if i % 1000 == 0:
            print(f"{i / 1000}% done!")
        if i > 100000:
            break
        tweets.append([tweet.user.username, tweet.date, tweet.likeCount, tweet.sourceLabel, tweet.content])

    # Converting data to dataframe
    tweets_df = pd.DataFrame(tweets, columns=["User", "Date Created", "Number of Likes", "Source of Tweet", "Tweet"])
    tweets_df.to_csv(fn)


def main():
    # Creating queries
    queries = ['#WorldCup2010 lang:en since:2010-06-11 until:2010-07-11',
               '#WorldCup2014 lang:en since:2014-06-12 until:2014-07-13',
               '#WorldCup2018 lang:en since:2018-06-14 until:2018-07-15',
               '#WorldCup2022 lang:en since:2022-11-20 until:2022-12-18']
    for q in queries:
        scrape(q)


if __name__ == "__main__":
    main()
