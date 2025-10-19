from datetime import datetime, timedelta

import psycopg2
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