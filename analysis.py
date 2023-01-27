from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from tqdm import tqdm
import pandas as pd
import numpy as np
import re
import ast


def _preprocess_tweet(content):
    # make all letters lowercase
    res = content.lower()
    # replace user tags with generic @user
    res = re.sub(r"@\w+", "@user", res)
    # replace links with just the word http
    res = re.sub(r"https?:\/(\/[^\/\s]+)+\/?", "http", res)
    # remove line breaks
    res = res.replace("\n", " ").replace("\r", "")
    return res


def _softmax(x):
    return np.exp(x) / sum(np.exp(x))


def _predict_label(tweet, labels, tokenizer, model):
    encoded_tweet = tokenizer(tweet, return_tensors='pt')
    output = model(**encoded_tweet)
    scores = _softmax(output[0][0].detach().numpy())
    ranking = np.argsort(scores)[::-1]
    return (labels[ranking[0]], scores[ranking[0]])


def add_sentiment(filename):
    # Use roBERTa based model trained on ~58M tweets
    # (https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment)
    # Initialize models and tokenizer
    MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    labels = ["Negative", "Neutral", "Positive"]
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    model.save_pretrained(MODEL)

    print(f"working on file: {filename}.")
    tweets_df = pd.read_csv(filename)
    # clean up tweets that are not strings
    tweets_df["Tweet"] = tweets_df["Tweet"].apply(lambda x: x if isinstance(x, str) else "EMPTY TWEET")

    # Create columns for analysis
    tweets_df["Sentiment"] = ""
    tweets_df["Confidence"] = np.nan
    tweets_df["hashtags"] = tweets_df["Tweet"].apply(lambda x: re.findall(r"#(\w+)", x))
    tweets_df["Tweet_preprocessed"] = tweets_df["Tweet"].apply(lambda x: _preprocess_tweet(x))

    # loop over rows to perform the sentiment analysis
    for index, row in tqdm(tweets_df.iterrows()):
        content = row["Tweet_preprocessed"]
        (label, score) = _predict_label(content, labels, tokenizer, model)
        tweets_df.loc[index, "Sentiment"] = label
        tweets_df.loc[index, "Confidence"] = score

    # Save the resulting Dataframe (with added columns) to csv
    tweets_df.to_csv("Analysis_" + filename)


def _hashtag_analysis(tweets_df, save_to_csv, filename="", top=20):
    ## gather hashtags and rank them on most occurring
    # First, flatten the list of lists of hashtags that occur in each tweet
    flatlist_hashtags = [item.lower() for sublist in tweets_df["hashtags"].values.tolist() for item in
                         ast.literal_eval(sublist)]
    # group hashtags together, count their occurrences
    grouped_hashtags = dict((x, flatlist_hashtags.count(x)) for x in set(flatlist_hashtags))
    # Create list of the hashtags sorted in descending order (hence the - in -item[1])
    freq_list_hashtags = [x for x in sorted(grouped_hashtags.items(), key=lambda item: -item[1])]
    print(f"{top} most frequent hashtags: {freq_list_hashtags[:top]}")

    if save_to_csv:
        # Save the full list of hashtags occurring at least 5 times in csv
        with open(f"freq_list_hashtags.csv", 'a', newline='') as file:
            file.write(f"{filename},")
            for (k, v) in freq_list_hashtags:
                if v >= 5:
                    file.write(f"{k}: {v}, ")
            file.write("\n")


def analyze_tweets(filename):
    tweets_df = pd.read_csv(filename)
    # Count sentiment labels
    labels_count = tweets_df["Sentiment"].value_counts()
    print(f"labels count:\n{labels_count}")

    # count tweets that contain protesting words: "boycott", "human right" or "corrupt"
    protest_words = r'boycott|human right|corrupt'
    protest_tweets = tweets_df[tweets_df.Tweet_preprocessed.str.contains(protest_words)]
    protest_tweets.to_csv("PROTEST_TWEETS" + filename)
    protest_labels_count = protest_tweets["Sentiment"].value_counts()
    print(f"protest labels count:\n{protest_labels_count}")

    avg_confidence = protest_tweets.groupby('Sentiment', as_index=False)["Confidence"].mean()
    print(f"protest labels sentiment average confidence:\n{avg_confidence}")

    # gather all hashtags and rank them on most occurring
    _hashtag_analysis(tweets_df, False, filename)

    # gather common hashtags of negative tweets
    negative_tweets = tweets_df[tweets_df.Sentiment.str.contains("Negative")]
    print("Negative hashtags:")
    _hashtag_analysis(negative_tweets, False, top=50)

    # gather common hashtags of positive tweets
    positive_tweets = tweets_df[tweets_df.Sentiment.str.contains("Positive")]
    print("Positive hashtags:")
    _hashtag_analysis(positive_tweets, False, top=50)


def main():
    filenames = ["#WorldCup2010_100K.csv", "#WorldCup2014_100K.csv",
                 "#WorldCup2018_100K.csv", "#WorldCup2022_100K.csv"]
    for fn in filenames:
        add_sentiment(fn)
        print(f"%%%%%%%%%%%%%%%%%%%%%%%%%%%% {fn} %%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        analyze_tweets("Analysis_" + fn)


if __name__ == "__main__":
    main()
