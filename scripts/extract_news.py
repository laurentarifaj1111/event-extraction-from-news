from datetime import datetime, timedelta

import psycopg2
import requests
from langdetect import detect
from psycopg2 import sql

cnt = 0
t = 100

# Postgresql Database Connection
dbname = "mydatabase"
user = "myuser"
password = "mypassword"
host = "localhost"
port = 5432

container_name = ""
newsq = "newsq"
phaseq = "phaseq"
table_name = "news_articles"

data_directory = "data"
article_file = "article"
unwanted_file = "unwanted"
today_date = datetime.now()
previous_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

api_list = ['2e9ca2e68438477cb89a2a15745e6a35']

class CollectData:
    def __init__(self):
        pass

    def connect_to_database(self, dbname, user, password, host, port):
        conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=port)
        return conn

    def clear_file_content(self, filepath):
        # Overwrite with empty content
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('')

    def read_file_text(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()

    def append_phrase_to_file(self, filepath, existing_content, new_content):
        updated_content = existing_content + '\n' + new_content
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(updated_content)

    def read_file_into_list(self, file_content):
        return [line.strip() for line in file_content.split('\n')]

    def fetch_news(self, query, page, api_key):
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': query,
            'page': page,
            'apiKey': api_key,
            'from': previous_date,
            'to': today_date,
            'language': 'en'
        }
        response = requests.get(url, params=params)
        return response.json()

    def get_article_data(self, json_data, q):
        source_id = json_data["source"]["id"]
        source_name = json_data["source"]["name"]
        title = json_data['title']
        author = json_data['author']
        description = json_data['description']
        url = json_data['url']
        urlToImage = json_data['urlToImage']
        publishedAt = json_data['publishedAt']
        content = json_data['content']
        result = {
            "source_id": source_id,
            "source_name": source_name,
            "title": title,
            "author": author,
            "description": description,
            "content": content,
            "url": url,
            "urlToImage": urlToImage,
            "publishedAt": publishedAt,
            "category": q
        }
        return result

    def push_data(self, data, conn):
        insert_query = sql.SQL("""
                               INSERT INTO {}
                               (source_id, source_name, author, title, description, url, url_to_image, published_at,
                                content, category)
                               VALUES
                                   (%(source_id)s, %(source_name)s, %(author)s, %(title)s, %(description)s, %(url)s, %(urlToImage)s, %(publishedAt)s, %(content)s, %(category)s)
                               """).format(sql.Identifier(table_name))

        with conn.cursor() as cur:
            cur.execute(insert_query, data)
        conn.commit()

    def append_word_to_file(self, word, filename="customoutput.log"):
        with open(filename, 'a', encoding='utf-8') as file:
            file.write(str(word) + '\n')