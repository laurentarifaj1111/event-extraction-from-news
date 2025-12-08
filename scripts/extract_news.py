import csv
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import pandas as pd
import psycopg2
import requests
from bs4 import BeautifulSoup
from langdetect import detect
from psycopg2 import sql
import json

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

    def runner(self):
        api = 0
        conn = self.connect_to_database(dbname, user, password, host, port)

        # Read newsq file from local
        newsq_path = os.path.join(data_directory, newsq)
        news_file_content = self.read_file_text(newsq_path)
        for line in news_file_content.split('\n'):
            q = line.strip()
            phaseq_path = os.path.join(data_directory, phaseq)
            existing_phrase_list = []
            if os.path.exists(phaseq_path):
                existing_phrase_list = self.read_file_into_list(self.read_file_text(phaseq_path))
            if q not in existing_phrase_list:
                try:
                    news_raw = self.fetch_news(query=q, page=1, api_key=api_list[api])
                except IndexError:
                    break

                if news_raw.get("status") != "error":
                    upper_range = min(int(news_raw["totalResults"] / 100), 5)
                    for i in range(1, upper_range + 1):
                        if i != 1:
                            news_raw = self.fetch_news(query=q, page=i, api_key=api_list[api])
                        if news_raw.get("status") != "error":
                            articles = news_raw.get("articles", [])
                            for article in articles:
                                exarticle = self.get_article_data(article, q)
                                self.push_data(exarticle, conn)
                            # append q to completed list
                            self.append_phrase_to_file(phaseq_path,
                                                       "\n".join(existing_phrase_list),
                                                       q)
                            self.append_word_to_file(f"{q} Completed")
                        else:
                            if news_raw.get("code") == "rateLimited":
                                api = (api + 1) % len(api_list)
                # end if status ok
        conn.close()
        # Clear completed list file
        phaseq_path = os.path.join(data_directory, phaseq)
        self.clear_file_content(phaseq_path)


class DumpSqlData:
    def __init__(self):
        pass

    def export_to_csv(self, csv_file_path=article_file):
        try:
            conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=port)
            cur = conn.cursor()
            select_query = sql.SQL("SELECT * FROM {}").format(sql.Identifier(table_name))
            cur.execute(select_query)
            rows = cur.fetchall()
            col_names = [desc[0] for desc in cur.description]

            out_path = os.path.join(data_directory, f"{csv_file_path}.csv")
            with open(out_path, 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(col_names)
                csv_writer.writerows(rows)

        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            cur.close()
            conn.close()

    def CurrentData(self, csv_file_path=article_file):
        target_date_str = str(previous_date)
        input_path = os.path.join(data_directory, f"{csv_file_path}.csv")
        data = pd.read_csv(input_path)
        mask = pd.notna(data['published_at'])
        filtered_df = data[mask & data['published_at'].str.startswith(target_date_str)]
        out_path = os.path.join(data_directory, f"{unwanted_file}.csv")
        filtered_df.to_csv(out_path, index=False)


class Preprocessing:
    def __init__(self):
        pass

    def filter_english(self, text):
        try:
            lang = detect(text)
            if lang == 'en':
                return text
            else:
                return None
        except:
            return None

    def remove_unwanted_row(self):
        input_path = os.path.join(data_directory, f"{unwanted_file}.csv")
        data = pd.read_csv(input_path)
        data = data.dropna(subset=['url'])
        data = data.drop_duplicates(subset=['url'], keep='first')
        data["processed_content"] = data["content"].apply(self.filter_english)
        data = data.dropna(subset=['processed_content'])
        data = data.drop(columns=['processed_content'])
        out_path = os.path.join(data_directory, f"{unwanted_file}.csv")
        data.to_csv(out_path, index=False)

    def after_prepprocessing(self):
        input_path = os.path.join(data_directory, f"{previous_date}.csv")
        data = pd.read_csv(input_path)
        data = data.dropna(subset=['author', 'url', 'title', 'full_content'])
        out_path = os.path.join(data_directory, f"{previous_date}.csv")
        data.to_csv(out_path, index=False)

    def read_csv(self):
        input_path = os.path.join(data_directory, f"{unwanted_file}.csv")
        data = pd.read_csv(input_path)
        return data

    def get_BusinessInsider(self, url):
        try:
            response = requests.get(url)
            content = response.text
            soup = BeautifulSoup(content, 'html.parser')
            script_tag = soup.find('script', {'id': '__NEXT_DATA__', 'type': 'application/json'})
            json_data = json.loads(script_tag.string)
            body = json_data['props']['pageProps']['articleShowData']['body']
            soup2 = BeautifulSoup(body, 'html.parser')
            paragraphs = soup2.find_all('p')
            text_content = ' '.join([paragraph.get_text(strip=True) for paragraph in paragraphs])
            return text_content
        except:
            return "None"

    def preprocess(self, source_name, url):

        global cnt, t
        cnt += 1
        if cnt > t:
            self.append_word_to_file(cnt)
            t += 100

        if source_name == "Business Insider":
            return self.get_BuinessInsider(url)

        if source_name == "Forbes":
            return self.get_Forbes(url)

        if source_name == "Android Central":
            return self.get_Android_Central(url)

        if source_name == "Gizmodo.com":
            return self.get_Gizmodo_com(url)

        if source_name == "BBC News":
            return self.get_bbc_news(url)

        if source_name == "Al Jazeera English":
            return self.get_al_jazeera_english(url)

        if source_name == "AllAfrica - Top Africa News":
            return self.get_allafrica(url)

        if source_name == "ABC News":
            return self.get_abc_news(url)

        if source_name == "Globalsecurity.org":
            return self.get_Globalsecurity_org(url)

        if source_name == "RT":
            return self.get_rt(url)

        if source_name == "Marketscreener.com":
            return self.get_market_screener(url)

        if source_name == "Phys.Org":
            return self.get_phys_org(url)

        if source_name == "Time":
            return self.get_time_news(url)

        if source_name == "NPR":
            return self.get_npr(url)

        if source_name == "Boing Boing":
            return self.get_boing_boing(url)

        if source_name == "CNA":
            return self.get_cna(url)

        if source_name == "The Punch":
            return self.get_punch(url)

        if source_name == "Euronews":
            return self.get_euronews(url)

        if source_name == "Deadline":
            return self.get_dedline_news(url)

        if source_name == "ReadWrite":
            return self.get_readwrite(url)

        if source_name == "International Business Times":
            return self.get_international_buiness_times(url)

        if source_name == "CNN":
            return self.get_cnn(url)

        if source_name == "The Verge":
            return self.get_The_Verge(url)

        if source_name == "The Indian Express":
            return self.get_indian_express(url)

        if source_name == "Wired":
            return self.get_wired(url)

        if source_name == "GlobeNewswire":
            return self.get_global_news_wire(url)

        if source_name == "ETF Daily News":
            return self.get_etf_daily_news(url)

        if source_name == "The Times of India":
            return self.get_times_of_india(url)

        if source_name == "Digital Trends":
            return self.get_digital_content(url)

        else:
            return "None"

    def runner(self):
        data = self.read_csv()
        data["source_name"].dropna(inplace=True)
        data["url"].dropna(inplace=True)

        def process_row(row):
            return self.preprocess(row.source_name, row.url)

        with ThreadPoolExecutor() as executor:
            data['full_content'] = list(executor.map(process_row, data.itertuples(index=False)))

        out_path = os.path.join(data_directory, f"{previous_date}.csv")
        data.to_csv(out_path, index=False)

    def get_BuinessInsider(self, url):
        try:
            response = requests.get(url)
            content = response.text

            soup = BeautifulSoup(content, 'html.parser')

            script_tag = soup.find('script', {'id': '__NEXT_DATA__', 'type': 'application/json'})
            json_data = json.loads(script_tag.string)
            body = json_data['props']['pageProps']['articleShowData']['body']

            soup = BeautifulSoup(body, 'html.parser')
            paragraphs = soup.find_all('p')
            text_content = ' '.join([paragraph.get_text(strip=True) for paragraph in paragraphs])

            return text_content

        except:
            return "None"

    def get_Forbes(self, url):
        try:
            response = requests.get(url)
            content = response.text

            soup = BeautifulSoup(content, 'html.parser')
            target_element = soup.find('div', class_='article-body fs-article fs-responsive-text current-article')

            child_tags = target_element.find_all(['h2', 'p'])
            combined_text = ' '.join([tag.text for tag in child_tags])

            return combined_text

        except:
            return "None"

    def get_Android_Central(self, url):
        try:
            response = requests.get(url)
            content = response.text

            soup = BeautifulSoup(content, 'html.parser')
            target_element = soup.find('div', id='article-body')

            child_tags = target_element.find_all(['h2', 'p'])
            combined_text = ' '.join([tag.text for tag in child_tags])

            return combined_text

        except:
            return "None"

    def get_Gizmodo_com(self, url):
        try:
            response = requests.get(url)
            content = response.text

            soup = BeautifulSoup(content, 'html.parser')
            target_element = soup.find('div', class_='sc-xs32fe-0 gKylik js_post-content')

            child_tags = target_element.find_all(['h3', 'p'])
            combined_text = ' '.join([tag.text for tag in child_tags])

            return combined_text

        except:
            return "None"

    def get_bbc_news(self, url):
        try:
            news = []
            headersList = {
                "Accept": "*/*",
                "User-Agent": "Thunder Client (https://www.thunderclient.com)"
            }

            payload = ""

            news_response = requests.request("GET", url, data=payload, headers=headersList)
            soup = BeautifulSoup(news_response.content, features="html.parser")
            soup.findAll("p", {"class": "ssrcss-1q0x1qg-Paragraph eq5iqo00"})
            soup.findAll("div", {"data-component": "text-block"})
            for para in soup.findAll("div", {"data-component": "text-block"}):
                news.append(para.find("p").getText())
            joinnews = " ".join(news)

            return joinnews
        except:
            return "None"

    def get_al_jazeera_english(self, url):
        try:
            news_response = requests.get(url)
            soup = BeautifulSoup(news_response.content, features="html.parser")

            paragraphs = soup.find_all('p')
            text_content = ' '.join([paragraph.get_text(strip=True) for paragraph in paragraphs])

            return text_content

        except:
            return "None"

    def get_allafrica(self, url):
        try:
            news_response = requests.get(url)
            soup = BeautifulSoup(news_response.content, features="html.parser")
            target_element = soup.find('div', class_="story-body")
            paragraphs = target_element.find_all('p')
            text_content = ' '.join([paragraph.get_text(strip=True) for paragraph in paragraphs])

            return text_content

        except:
            return "None"

    def get_abc_news(self, url):
        try:
            news_response = requests.get(url)
            soup = BeautifulSoup(news_response.content, features="html.parser")
            content_div = soup.find('div', {'data-testid': 'prism-article-body'})
            paragraphs = content_div.find_all('p')
            text_content = ' '.join([paragraph.get_text(strip=True) for paragraph in paragraphs])

            return text_content

        except:
            return "None"

    def get_Globalsecurity_org(self, url):
        try:
            news_response = requests.get(url)
            soup = BeautifulSoup(news_response.content, features="html.parser")
            content_div = soup.find('div', {'id': 'main'}).find('div', {'id': 'content'})
            paragraphs = content_div.find_all('p')
            text_content = ' '.join([paragraph.get_text(strip=True) for paragraph in paragraphs])

            return text_content

        except:
            return "None"

    def get_rt(self, url):
        try:
            news_response = requests.get(url)
            soup = BeautifulSoup(news_response.content, features="html.parser")
            content_div = soup.find('div', class_="article").find('div', class_="article__text text")
            paragraphs = content_div.find_all('p')
            text_content = ' '.join([paragraph.get_text(strip=True) for paragraph in paragraphs])

            return text_content

        except:
            return "None"

    def get_market_screener(self, url):
        try:
            news = []
            headersList = {
                "Accept": "*/*",
                "User-Agent": "Thunder Client (https://www.thunderclient.com)"
            }

            payload = ""

            news_response = requests.request("GET", url, data=payload, headers=headersList)
            soup = BeautifulSoup(news_response.content, features="html.parser")
            content_div = soup.find('div', class_="txt-s4 article-text article-text--clear")
            paragraphs = content_div.find_all('p')
            text_content = ' '.join([paragraph.get_text(strip=True) for paragraph in paragraphs])

            return text_content

        except:
            headersList = {
                "Accept": "*/*",
                "User-Agent": "Thunder Client (https://www.thunderclient.com)"
            }

            payload = ""

            news_response = requests.request("GET", url, data=payload, headers=headersList)
            soup = BeautifulSoup(news_response.content, features="html.parser")
            meta_tag = soup.find('meta', {'http-equiv': 'refresh'})
            pattern = re.compile(r'content="0;url=\'(.*?)\'" http-equiv="refresh"')
            match = pattern.search(str(meta_tag)).group(1)
            web_url = "https://www.marketscreener.com{}".format(match)
            headersList = {
                "Accept": "*/*",
                "User-Agent": "Thunder Client (https://www.thunderclient.com)"
            }

            payload = ""

            news_response = requests.request("GET", web_url, data=payload, headers=headersList)
            soup = BeautifulSoup(news_response.content, features="html.parser")
            content_div = soup.find('div', class_="txt-s4 article-text article-text--clear ")
            paragraphs = content_div.find_all('p')
            text_content = ' '.join([paragraph.get_text(strip=True) for paragraph in paragraphs])

            return text_content

        finally:
            return "None"

    def get_phys_org(self, url):
        try:
            news = []
            headersList = {
                "Accept": "*/*",
                "User-Agent": "Thunder Client (https://www.thunderclient.com)"
            }

            payload = ""

            news_response = requests.request("GET", url, data=payload, headers=headersList)
            soup = BeautifulSoup(news_response.content, features="html.parser")
            content_div = soup.find('div', class_="mt-4 article-main")
            paragraphs = content_div.find_all('p')
            text_content = ' '.join([paragraph.get_text(strip=True) for paragraph in paragraphs])
            splited = text_content.split("More information:")
            text_content = splited[0]

            return text_content

        except:
            return "None"

    def get_time_news(self, url):
        try:
            news_response = requests.get(url)
            soup = BeautifulSoup(news_response.content, features="html.parser")
            content_div = soup.find('div', {'id': 'article-body-main'})
            paragraphs = content_div.find_all('p')
            text_content = ' '.join([paragraph.get_text(strip=True) for paragraph in paragraphs])

            return text_content

        except:
            return "None"


    def get_npr(self, url):
        try:
            news_response = requests.get(url)
            soup = BeautifulSoup(news_response.content, features="html.parser")
            content_div = soup.find('div', {'id': 'storytext'})
            paragraphs = content_div.find_all('p')
            text_content = ' '.join([paragraph.get_text(strip=True) for paragraph in paragraphs])
            return text_content

        except:
            return "None"

    def get_boing_boing(self, url):
        try:
            news_response = requests.get(url)
            soup = BeautifulSoup(news_response.content, features="html.parser")
            content_div = soup.find('section', class_="entry-content")
            paragraphs = content_div.find_all('p')
            text_content = ' '.join([paragraph.get_text(strip=True) for paragraph in paragraphs])
            return text_content

        except:
            return "None"


    def get_cna(self, url):
        try:
            news_response = requests.get(url)
            soup = BeautifulSoup(news_response.content, features="html.parser")
            content_div = soup.find('div', class_="text").find('div', class_="text-long")
            paragraphs = content_div.find_all('p')
            text_content = ' '.join([paragraph.get_text(strip=True) for paragraph in paragraphs])
            return text_content

        except:
            return "None"

    def get_punch(self, url):
        try:
            news_response = requests.get(url)
            soup = BeautifulSoup(news_response.content, features="html.parser")
            content_div = soup.find('div', class_="post-content")
            paragraphs = content_div.find_all('p')
            text_content = ' '.join([paragraph.get_text(strip=True) for paragraph in paragraphs])
            return text_content

        except:
            return "None"


    def get_euronews(self, url):
        try:
            news_response = requests.get(url)
            soup = BeautifulSoup(news_response.content, features="html.parser")
            script_tag = soup.find('script', {'type': 'application/ld+json'})
            json_data = json.loads(script_tag.string)["@graph"][0]["articleBody"]
            return json_data

        except:
            return "None"

    def get_dedline_news(self, url):
        try:
            news_response = requests.get(url)
            soup = BeautifulSoup(news_response.content, features="html.parser")
            content_div = soup.find('div',
                                    class_="a-content pmc-u-line-height-copy pmc-u-font-family-georgia pmc-u-font-size-16 pmc-u-font-size-18@desktop")
            paragraphs = content_div.find_all('p')
            text_content = ' '.join([paragraph.get_text(strip=True) for paragraph in paragraphs])
            return text_content

        except:
            return "None"

    def get_readwrite(self, url):
        try:
            news_response = requests.get(url)
            soup = BeautifulSoup(news_response.content, features="html.parser")
            content_div = soup.find('div', class_="entry-content col-md-10")
            paragraphs = content_div.find_all('p')
            text_content = ' '.join([paragraph.get_text(strip=True) for paragraph in paragraphs[:-1]])
            return text_content

        except:
            return "None"

    def get_international_buiness_times(self, url):
        try:
            headersList = {
                "Accept": "*/*",
                "User-Agent": "Thunder Client (https://www.thunderclient.com)"
            }

            payload = ""

            news_response = requests.request("GET", url, data=payload, headers=headersList)
            soup = BeautifulSoup(news_response.content, features="html.parser")
            content_div = soup.find('div', class_="article-paywall-contents")
            paragraphs = content_div.find_all('p')
            text_content = ' '.join([paragraph.get_text(strip=True) for paragraph in paragraphs])
            return text_content

        except:
            return "None"

    def get_cnn(self, url):
        try:
            news_response = requests.get(url)
            soup = BeautifulSoup(news_response.content, features="html.parser")
            content_div = soup.find('div', class_="article__content")
            paragraphs = content_div.find_all('p')
            text_content = ' '.join([paragraph.get_text(strip=True) for paragraph in paragraphs])
            return text_content

        except:
            return "None"

    def get_The_Verge(self, url):
        try:
            news_response = requests.get(url)
            soup = BeautifulSoup(news_response.content, features="html.parser")

            target_element = soup.find('div',
                                       class_='duet--article--article-body-component-container clearfix sm:ml-auto md:ml-100 md:max-w-article-body lg:mx-100')
            text_content = target_element.get_text(separator=' ', strip=True)

            return text_content
        except:
            return "None"

    def get_indian_express(self, url):
        try:
            news_response = requests.get(url)
            soup = BeautifulSoup(news_response.content, features="html.parser")
            content_div = soup.find('div', class_="story_details")
            paragraphs = content_div.find_all('p')
            text_content = ' '.join([paragraph.get_text(strip=True) for paragraph in paragraphs])
            return text_content

        except:
            return "None"

    def get_wired(self, url):
        try:
            news_response = requests.get(url)
            soup = BeautifulSoup(news_response.content, features="html.parser")
            content_div = soup.find('div', class_="body__inner-container")
            paragraphs = content_div.find_all('p')
            text_content = ' '.join([paragraph.get_text(strip=True) for paragraph in paragraphs])
            return text_content

        except:
            return "None"

    def get_global_news_wire(self, url):
        try:
            news_response = requests.get(url)
            soup = BeautifulSoup(news_response.content, features="html.parser")
            content_div = soup.find('div', {'itemprop': 'articleBody'})
            paragraphs = content_div.find_all('p')
            text_content = ' '.join([paragraph.get_text(strip=True) for paragraph in paragraphs])

            return text_content

        except:
            return "None"

    def get_etf_daily_news(self, url):
        try:
            news_response = requests.get(url)
            soup = BeautifulSoup(news_response.content, features="html.parser")
            content_div = soup.find('div', {'itemprop': 'articleBody'})
            paragraphs = content_div.find_all('p')
            text_content = ' '.join([paragraph.get_text(strip=True) for paragraph in paragraphs[:-1]])

            return text_content

        except:
            return "None"

    def get_times_of_india(self, url):
        try:
            news_response = requests.get(url)
            soup = BeautifulSoup(news_response.content, features="html.parser")
            content_div = soup.find('div', class_="_s30J clearfix")

            all_text = content_div.text
            return all_text

        except:
            try:
                news_response = requests.get(url)
                soup = BeautifulSoup(news_response.content, features="html.parser")
                content_div = soup.select('article[class^="artData clr"]')
                all_text = '\n'.join([div.get_text(separator=' ') for div in content_div])
                return all_text
            except:
                return "None"

    def get_digital_content(self, url):
        try:
            news_response = requests.get(url)
            soup = BeautifulSoup(news_response.content, features="html.parser")
            content_div = soup.find('article', {'itemprop': 'articleBody'})
            paragraphs = content_div.find_all('p')
            text_content = ' '.join([paragraph.get_text(strip=True) for paragraph in paragraphs])

            return text_content

        except:
            return "None"

        def append_word_to_file(self, word, filename="customoutput.log"):
            with open(filename, 'a') as file:
                file.write(str(word) + '\n')

    def list_files_in_directory(directory=data_directory):
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        return files

    def append_word_to_file(word, filename="customoutput.log"):
        with open(filename, 'a') as file:
            file.write(str(word) + '\n')

    def remove_files(directory_path):
        files_to_remove = ['unwanted.csv', 'article.csv']
        try:
            for filename in files_to_remove:
                file_path = os.path.join(directory_path, filename)
                if os.path.exists(file_path):
                    os.remove(file_path)
        except Exception as e:
            print(f"An error occurred: {e}")
