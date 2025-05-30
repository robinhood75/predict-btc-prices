import alpaca_trade_api as tradeapi
import json
import numpy as np
from openai import OpenAI
import polars as pl
from data.utils import *
from data.base_loader import DataGenerator


class DataGeneratorNews(DataGenerator):
    def __init__(
            self, 
            start_date, 
            end_date, 
            data_folder, 
            page_size_daily=50,
            max_char_article=5000
        ):
        """
        Parameters:
        - page_size_daily: max number of articles loaded for each day
        - max_char_article: maximum number of characters considered per article
        """
        super().__init__(start_date, end_date, data_folder)
        self.page_size_daily = page_size_daily
        self.max_char_article = max_char_article
        self.symbols = ",".join(NEWS_PARAMETERS.get("symbols"))
        self.categories = NEWS_PARAMETERS.get("categories")
        self.client = OpenAI(api_key=OPENAI_KEY)

    def load_data(self):
        """
        Load raw news data from Alpaca.
        Return: dict, maps a date to list of articles
        """
        ret = {}

        for date_idx in range(len(self.date_range) - 1):
            date = self.date_range[date_idx]
            articles_this_day = self.get_articles_daily(self.symbols, date_idx)
            ret[date] = articles_this_day

        return ret
    
    def transform_data(self, raw_data):
        """
        Categorize and score sentiment for all articles
        Parameters:
        - raw_data: raw_data as generated by load_data
        Return: 
        - polars DataFrame with a score for each date and category
        """
        scores = {}

        # Get following dict: date -> category -> score
        for date, articles in raw_data.items():
            scores[date] = {}
            for cat in self.categories:
                scores[date][cat] = []

            for article in articles:
                text = (
                    (article.get("headline") or "") + " " +
                    (article.get("summary")  or "") + " " +
                    (article.get("content")  or "")
                ).strip()

                classification = self.classify_and_score_article_gpt(text)
                cat = classification.get("category", "Other")
                sentiment_label = classification.get("sentiment", "neutral")

                score = SENTIMENT_MAP.get(sentiment_label, 0)

                # Ignore improperly labelled categories
                if cat in self.categories:
                    scores[date][cat].append(score)

        # Build DF
        all_dates = sorted(scores.keys())
        rows = []

        for date in all_dates:
            row_values = []
            for cat in self.categories:
                scores_date_cat = scores[date].get(cat, [])
                if len(scores_date_cat) == 0:
                    avg_score = 0
                else:
                    avg_score = sum(scores_date_cat) / np.sqrt(len(scores_date_cat))
                row_values.append(avg_score)

            # Each row is [date, cat1_score, cat2_score, ...]
            rows.append([date] + row_values)

        # Name columns
        ret = pl.DataFrame(rows, schema=["date"] + self.categories)

        return ret

    def get_articles_daily(self, symbols, date_idx):
        """
        Get articles from NewsAPI with keywords <query>
        Parameters:
        - symbols: keyword or list of keywords
        - date: date of queried articles 
        Return:
        - List of dicts, each dict representing an article.
        """
        alpaca_api = tradeapi.REST(
            key_id=ALPACA_KEY,
            secret_key=ALPACA_SECRET_KEY,
            base_url=ALPACA_ENDPOINT
        )

        news_list = alpaca_api.get_news(
            symbol=symbols,
            start=self.date_range[date_idx],
            end=self.date_range[date_idx + 1],
            limit=self.page_size_daily
        )

        articles = []
        for item in news_list:
            if item.summary.strip() or item.content.strip():
                # Item is a NewsV2 object, transform to dict
                articles.append({
                    "id": item.id,
                    "headline": item.headline,
                    "author": item.author,
                    "created_at": item.created_at,
                    "summary": item.summary,
                    "content": item.content,
                    "symbols": item.symbols,
                    "url": item.url,
                })
        
        return articles
    
    def get_prompt(self, article_text):
        """
        Prompt for category & sentiment categorization to feed a given LLM
        Parameters :
        - article_text: concatenation of title, summary, and content
        Return : messages to prompt
        """
        categories_str = "\n".join([f"{i + 1}) {cat}" for i, cat in enumerate(self.categories)])

        prompt = f"""
            Classify the following article into ONE of the following categories:
            {categories_str}
            If none of these categories applies, respond "Other".

            Then, provide the sentiment of the article, which can be "positive", "negative", or "neutral".

            Article content:
            ---
            {article_text}
            ---

            Output format (JSON only):
            {{
            "category": "...",
            "sentiment": "positive|negative|neutral"
            }}
        """

        messages = [
            {"role": "system", "content": "You are a text analyst."},
            {"role": "user", "content": prompt.strip()}
        ]

        return messages
    
    def classify_and_score_article_gpt(self, article_text):
        """
        Use GPT to categorize and score sentiment (positive, negative, neutre).
        Returns following dict: {"category": "...", "sentiment": "..."}
        """
        messages = self.get_prompt(article_text)
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=200,
                temperature=0.0,
            )
            answer = response.choices[0].message.content
            
            parsed = json.loads(answer)
            category = parsed.get("category", "Other")
            sentiment = parsed.get("sentiment", "neutral")
            
            return {
                "category": category,
                "sentiment": sentiment
            }
        
        except Exception as e:
            print("[ERROR GPT]", e)
            return {
                "category": "Error",
                "sentiment": "neutral"
            }   
