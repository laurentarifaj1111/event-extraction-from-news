from datasets import load_dataset

dataset = load_dataset("gopalkalpande/bbc-news-summary")

dataset["train"].to_csv("train.csv")
# dataset["test"].to_csv("test.csv")
# dataset["validation"].to_csv("val.csv")


# dataset.to_csv("bbc_news.csv")