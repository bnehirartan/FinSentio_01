import gradio as gr
import google.generativeai as genai
import json, requests, yfinance as yf, re, unicodedata
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from fredapi import Fred
import re
from datetime import datetime, timedelta
# ✅ API Anahtarlarını tanımla
from fredapi import Fred
from torch.nn.functional import softmax
import torch
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from google.generativeai.types import Tool, FunctionDeclaration

from dotenv import load_dotenv
import os

# .env dosyasını oku (aynı dizinde ise parametre vermenize gerek yok)
load_dotenv(override=True)

# Anahtarları çek
finnhub_key        = os.getenv("finnhub_key")
exchangerate_key   = os.getenv("exchangerate_key")
fred_key           = os.getenv("fred_key")
gemini_api_key     = os.getenv("gemini_api_key")

fred = Fred(api_key=fred_key)
genai.configure(api_key=gemini_api_key)
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

def lookup_symbol(company_name):
    url = f"https://finnhub.io/api/v1/search?q={company_name}&token={finnhub_key}"
    try:
        res = requests.get(url).json()
        if res.get("count", 0) == 0 or not res.get("result"):
            return None
        # Return the first relevant symbol
        return res["result"][0]["symbol"]
    except Exception as e:
        return None
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
vader = SentimentIntensityAnalyzer()

import google.generativeai as genai

gemini_model = genai.GenerativeModel("gemini-1.5-flash-latest")

from torch.nn.functional import softmax
import torch

def fetch_stock_data(symbol):
        url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={finnhub_key}"
        try:
            res = requests.get(url).json()
            if "c" not in res:
                return {"status": "error", "message": "Missing 'c' field. Possibly invalid API key or symbol."}
            return {
                "source": "Finnhub",
                "data": {
                    "current_price": res["c"],
                    "open": res["o"],
                    "high": res["h"],
                    "low": res["l"]
                },
                "summary": f"{symbol.upper()} is trading at ${res['c']}."
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

def fetch_forex_data(pair):
        try:
            base, target = pair.upper().strip().split("/")
            symbol_yf = f"{base}{target}=X"
            url = f"https://v6.exchangerate-api.com/v6/{exchangerate_key}/pair/{base}/{target}"
            res = requests.get(url).json()
            current_rate = res.get("conversion_rate")

            if not current_rate:
                raise ValueError("ExchangeRate API did not return a valid conversion rate.")

            forex = yf.Ticker(symbol_yf)
            hist = forex.history(period="7d")["Close"]
            if len(hist) < 2:
                raise ValueError("Insufficient historical data for trend/volatility analysis.")

            vol_result = calculate_volatility(symbol_yf, period="7d")
            trend_result = analyze_trend(symbol_yf, period="7d")

            trend = "increasing" if trend_result["data"]["end"] > trend_result["data"]["start"] else \
                    "decreasing" if trend_result["data"]["end"] < trend_result["data"]["start"] else "stable"

            return {
                "source": "ExchangeRate API + Yahoo Finance",
                "data": {
                    "current_rate": round(current_rate, 4),
                    "start_rate": round(trend_result["data"]["start"], 4),
                    "end_rate": round(trend_result["data"]["end"], 4),
                    "trend": trend,
                    "volatility": round(vol_result["data"]["volatility"], 4)
                },
                "summary": f"{pair.upper()} is currently {round(current_rate, 4)}. Trend: {trend} with volatility {round(vol_result['data']['volatility'], 4)}."
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

def fetch_historical_data(symbol, period="1mo"):
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period=period)
            last_5_days = hist.tail(5).rename_axis("Date").reset_index()
            last_5_days["Date"] = last_5_days["Date"].dt.strftime("%Y-%m-%d")
            return {
                "source": "Yahoo Finance",
                "data": last_5_days.to_dict(orient="records"),
                "summary": f"Retrieved last 5 days of historical data for {symbol}."
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

def extract_macro_topic(user_input: str):
    user_input = user_input.lower()

    topic_keywords = {
        "inflation": ["inflation", "enflasyon", "price", "cpi", "cost of living"],
        "interest rate": ["interest", "rate", "faiz", "federal funds", "benchmark rate"],
        "gdp": ["gdp", "growth", "büyüme", "gross domestic", "output"]
    }

    matched_topics = []

    for topic, keywords in topic_keywords.items():
        if any(kw in user_input for kw in keywords):
            matched_topics.append(topic)

    return matched_topics if matched_topics else None



def fetch_macro_data(topic):
        print("🧪 ENTERED fetch_macro_data with topic:", topic)
        indicator_map = {
            "inflation": "CPIAUCSL",      # CPI (Consumer Price Index)
            "interest rate": "FEDFUNDS",  # Federal Funds Rate
            "gdp": "GDP"                  # Gross Domestic Product
        }

        try:
            topic_clean = topic.strip().lower()
            # Normalize keywords
            if "interest" in topic_clean and "rate" in topic_clean:
                topic_clean = "interest rate"
            elif "inflation" in topic_clean:
                topic_clean = "inflation"
            elif "gdp" in topic_clean or "gross domestic product" in topic_clean:
                topic_clean = "gdp"

            series_id = indicator_map.get(topic_clean)
            if not series_id:
                raise ValueError(f"Unsupported macro topic: '{topic}'. Try one of: {list(indicator_map.keys())}")

            data = fred.get_series(series_id).dropna()

            if topic_clean == "inflation":
                latest = data.iloc[-1]
                previous = data.iloc[-2]
                rate = ((latest - previous) / previous) * 100
                return {
                    "source": "FRED",
                    "data": {"inflation_rate_percent": round(rate, 2)},
                    "summary": f"The current inflation rate is {round(rate, 2)}% based on the most recent CPI data."
                }

            else:  # GDP or interest rate — return latest value
                latest = data.iloc[-1]
                return {
                    "source": "FRED",
                    "data": {"latest_value": round(latest, 2)},
                    "summary": f"The latest value for {topic_clean.upper()} is {round(latest, 2)}."
                }

        except Exception as e:
            return {"status": "error", "message": str(e)}




 # === Load FinBERT (English) ===
finbert_model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
finbert_tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")

# === Initialize VADER (Turkish fallback) ===
vader = SentimentIntensityAnalyzer()

def fetch_market_sentiment(topic):
    try:
        
        to_date = datetime.today().date()
        from_date = to_date - timedelta(days=7)

        # === Helper: Sentiment for English ===
        def analyze_sentiment_en(text):
            inputs = finbert_tokenizer(text, return_tensors="pt", truncation=True)
            outputs = finbert_model(**inputs)
            probs = softmax(outputs.logits, dim=1)[0]
            labels = ["negative", "neutral", "positive"]    
            return labels[torch.argmax(probs)]

        # === Helper: Sentiment for Turkish ===
        def analyze_sentiment_tr(text):
            score = vader.polarity_scores(text)["compound"]
            if score > 0.05:
                return "positive"
            elif score < -0.05:
                return "negative"
            else:
                return "neutral"

        # === Helper: Fetch News in Given Language ===
        def fetch_news(language):
            url = (
                f"https://newsapi.org/v2/everything?"
                f"qInTitle={topic}&" 
                f"from={from_date}&"
                f"to={to_date}&"
                f"language={language}&"
                f"sortBy=publishedAt&"
                f"apiKey={NEWSAPI_KEY}"
            )
            response = requests.get(url)
            if response.status_code != 200:
                return []
            return response.json().get("articles", [])

        # === Fetch Articles ===
        articles_en = fetch_news("en")
        articles_tr = fetch_news("tr")

        if not articles_en and not articles_tr:
            return {"status": "fallback", "message": f"No news found for {topic}"}

        sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
        detailed = []

        index=0

        for article in articles_en[:3]:
            headline = article.get("title", "")
            if len(headline.split()) <= 4:
                continue  # skip low-info headlines
            sentiment = analyze_sentiment_en(headline)
            sentiment_counts[sentiment] += 1
            detailed.append({
                "index": index,
                "lang": "en",
                "headline": headline,
                "sentiment": sentiment,
                "url": article.get("url", "#")
            })
            index += 1

        for article in articles_tr[:3]:
            headline = article.get("title", "")
            if len(headline.split()) <= 4:
                continue
            sentiment = analyze_sentiment_tr(headline)
            sentiment_counts[sentiment] += 1
            detailed.append({
                "index": index,
                "lang": "tr",
                "headline": headline,
                "sentiment": sentiment,
                "url": article.get("url", "#")
            })
            index += 1

        detailed.sort(key=lambda x: x["index"])  # Sort by original order
        total = sum(sentiment_counts.values())
        dominant = max(sentiment_counts, key=sentiment_counts.get)
        confidence = round(sentiment_counts[dominant] / total, 2)
        return {
            "source": "NewsAPI (TR+EN) + FinBERT/VADER",
            "data": sentiment_counts,
            "summary": f"Sentiment around '{topic}' is mostly {dominant} ({sentiment_counts[dominant]} of {total}).",
            "confidence": confidence,
            "headlines": detailed,
            "explanation": (
                f"We analyzed {total} headlines in Turkish and English. "
                f"The majority expressed a '{dominant}' sentiment. "
                f"The confidence score of {confidence * 100:.0f}% shows how dominant that sentiment was."
            )
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

def fetch_crypto_quote(symbol):
        crypto_map = {
            "btc": "BINANCE:BTCUSDT",
            "eth": "BINANCE:ETHUSDT",
            "bnb": "BINANCE:BNBUSDT",
            "ada": "BINANCE:ADAUSDT",
        }
        full_symbol = crypto_map.get(symbol.lower())
        if not full_symbol:
            return {"status": "error", "message": "Unsupported crypto symbol."}
        try:
            url = f"https://finnhub.io/api/v1/quote?symbol={full_symbol}&token={finnhub_key}"
            res = requests.get(url).json()
            return {
                "source": "Finnhub",
                "data": {"price": res["c"], "high": res["h"], "low": res["l"]},
                "summary": f"{symbol.upper()} is trading at ${res['c']}."
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

def fetch_commodity_data(commodity_name):
        commodity_symbols = {
            "gold": "GC=F",
            "silver": "SI=F",
            "oil": "CL=F",
            "natural gas": "NG=F"
        }
        symbol = commodity_symbols.get(commodity_name.lower())
        if not symbol:
            return {"status": "error", "message": f"Unsupported commodity: {commodity_name}"}
        try:
            data = yf.Ticker(symbol).history(period="1d")
            price_usd = round(data["Close"].iloc[-1], 2)
            url = f"https://v6.exchangerate-api.com/v6/{exchangerate_key}/pair/USD/TRY"
            res = requests.get(url).json()
            rate = res.get("conversion_rate")
            price_try = round(price_usd * rate, 2)
            return {
                "source": "Yahoo + ExchangeRate",
                "data": {"price_usd": price_usd, "price_try": price_try, "rate": rate},
                "summary": f"{commodity_name.title()} = {price_usd} USD ≈ {price_try} TRY"
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

def fetch_dividend_data(topic):
        try:
            if isinstance(topic, list):
                topic = topic[0]
            if not topic.isupper() or len(topic) > 5:
                symbol = lookup_symbol(topic)
                if not symbol:
                    return {"status": "error", "message": f"Could not resolve symbol."}
            else:
                symbol = topic.upper()

            url = f"https://finnhub.io/api/v1/stock/metric?symbol={symbol}&metric=all&token={finnhub_key}"
            res = requests.get(url).json()
            m = res.get("metric", {})
            return {
                "source": "Finnhub",
                "data": {
                    "dividend_yield": m.get("dividendYieldIndicatedAnnual"),
                    "dividend_per_share": m.get("dividendPerShareAnnual")
                },
                "summary": f"{symbol} yield: {m.get('dividendYieldIndicatedAnnual')}%, DPS: ${m.get('dividendPerShareAnnual')}"
            }

        except Exception as e:
            return {"status": "error", "message": str(e)}

def calculate_volatility(symbol, period="1mo"):
        try:
            hist = yf.Ticker(symbol).history(period=period)
            vol = hist["Close"].pct_change().std()
            return {"source": "Yahoo", "data": {"volatility": vol}, "summary": f"{symbol} volatility = {round(vol, 4)}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

def analyze_trend(symbol, period="1mo"):
        try:
            hist = yf.Ticker(symbol).history(period=period)
            start, end = hist["Close"].iloc[0], hist["Close"].iloc[-1]
            trend = "upward" if end > start else "downward"
            return {
                "source": "Yahoo",
                "data": {"start": start, "end": end},
                "summary": f"{symbol} is trending {trend}."
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}



def init_chat(system_instruction, tools):
  # Configure the model
  generation_config = {
    "temperature": 0.3,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
  }
  model = genai.GenerativeModel(model_name='gemini-1.5-flash-latest',
                                generation_config=generation_config,
                                safety_settings={
                                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT
                                    : HarmBlockThreshold.BLOCK_NONE,},
                                tools=tools,
                                system_instruction=system_instruction)
  chat = model.start_chat()


  return chat
intent_to_function = {
    "fetch_stock_data": fetch_stock_data,
    "fetch_forex_data": fetch_forex_data,
    "fetch_historical_data": fetch_historical_data,
    "fetch_macro_data": fetch_macro_data,
    "fetch_market_sentiment": fetch_market_sentiment,
    "fetch_crypto_quote": fetch_crypto_quote,
    "fetch_commodity_data": fetch_commodity_data,
    "fetch_dividend_data": fetch_dividend_data,
    "calculate_volatility": calculate_volatility,
    "analyze_trend": analyze_trend

}



tools = [
    Tool(function_declarations=[

        FunctionDeclaration(
            name="fetch_stock_data",
            description="Get stock price info such as current, open, high, low.",
            parameters={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock symbol (e.g., AAPL)"}
                },
                "required": ["symbol"]
            }
        ),

        FunctionDeclaration(
            name="fetch_forex_data",
            description="Get current forex rate, trend and volatility for a currency pair.",
            parameters={
                "type": "object",
                "properties": {
                    "pair": {"type": "string", "description": "Currency pair (e.g., USD/EUR)"}
                },
                "required": ["pair"]
            }
        ),

        FunctionDeclaration(
            name="fetch_historical_data",
            description="Retrieve historical stock data.",
            parameters={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock symbol (e.g., AAPL)"},
                    "period": {"type": "string", "description": "e.g. 1mo, 7d, 6mo (default: 1mo)"}
                },
                "required": ["symbol"]
            }
        ),

        FunctionDeclaration(
            name="fetch_macro_data",
            description="Get macroeconomic data such as inflation, interest rate, or GDP.",
            parameters={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "One of: 'inflation', 'interest rate', or 'GDP'"
                    }
                },
                "required": ["topic"]
            }
        ),

        FunctionDeclaration(
            name="fetch_market_sentiment",
            description="Analyze market sentiment for a stock or company name.",
            parameters={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Stock symbol or company name (e.g., AAPL or Tesla)"
                    }
                },
                "required": ["topic"]
            }
        ),

        FunctionDeclaration(
            name="fetch_crypto_quote",
            description="Get current price of a cryptocurrency like BTC, ETH, etc.",
            parameters={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Crypto symbol (e.g., btc, eth, bnb)"
                    }
                },
                "required": ["symbol"]
            }
        ),

        FunctionDeclaration(
            name="fetch_commodity_data",
            description="Get commodity price in USD and TRY for gold, silver, oil etc.",
            parameters={
                "type": "object",
                "properties": {
                    "commodity_name": {
                        "type": "string",
                        "description": "Commodity name (e.g., gold, silver, oil, natural gas)"
                    }
                },
                "required": ["commodity_name"]
            }
        ),

        FunctionDeclaration(
            name="fetch_dividend_data",
            description="Retrieve dividend yield and dividend per share of a stock.",
            parameters={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Stock symbol or company name (e.g., AAPL or Apple)"
                    }
                },
                "required": ["topic"]
            }
        ),

        FunctionDeclaration(
            name="calculate_volatility",
            description="Calculate volatility of a stock or currency pair over a given period.",
            parameters={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Symbol (e.g., AAPL or USDEUR=X)"},
                    "period": {"type": "string", "description": "Period (e.g., 1mo, 7d, etc.)"}
                },
                "required": ["symbol"]
            }
        ),

        FunctionDeclaration(
            name="analyze_trend",
            description="Analyze trend direction of a symbol (stock or currency) over a given period.",
            parameters={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Symbol (e.g., AAPL, USDJPY=X)"},
                    "period": {"type": "string", "description": "Period (e.g., 1mo, 7d, etc.)"}
                },
                "required": ["symbol"]
            }
        ),


    ])
]
def execute_op(fn):
    try:
        func = globals()[fn.name]
        return func(**fn.args)
    except Exception as e:
        return {"status": "error", "message": str(e)}
system_instruction = """
You are a financial assistant with access to real-time financial data tools.

Your main task is to help users make informed financial decisions by retrieving and interpreting live financial data.

Language Rule:
Respond in the same language the user used. If the user asks in Turkish, reply in Turkish. If the user asks in English, reply in English.

You have access to the following tools:
- fetch_stock_data(symbol: str): Gets current stock price, open, high, and low from Finnhub.
- fetch_forex_data(pair: str): Gets current forex rate, trend, and volatility from ExchangeRate + Yahoo Finance.
- fetch_historical_data(symbol: str, period: str): Retrieves the last 5 days or custom-period historical data from Yahoo Finance.
- fetch_macro_data(topic: str): Retrieves macroeconomic indicators (inflation, interest rate, GDP) using the FRED API.
- fetch_market_sentiment(topic: str): Analyzes market sentiment using recent news headlines and FinBERT.
- fetch_crypto_quote(symbol: str): Gets current price of major cryptocurrencies from Finnhub.
- fetch_commodity_data(commodity_name: str): Gets gold, oil, etc. prices in both USD and TRY.
- fetch_dividend_data(topic: str): Gets dividend yield and dividend per share using Finnhub.
- calculate_volatility(symbol: str, period: str): Computes volatility of a financial asset over a given period.
- analyze_trend(symbol: str, period: str): Identifies if an asset is trending upward, downward, or stable over a time period.

Always prioritize using tools before generating responses. Do not guess or estimate prices, rates, or trends using your internal knowledge.

💡 Sentiment & Opinion Handling:

If the user asks how the market or investors feel about a company, event, or financial situation — use `fetch_market_sentiment(topic: str)`.
If the user asks about opinions, emotional reactions, IPOs, or market sentiment — you MUST call `fetch_market_sentiment`.
DO NOT try to answer these questions yourself or guess based on prior knowledge.
ALWAYS call the tool, even if the topic is vague or not a public company. It will work with plain text.

This includes:
- IPOs (e.g., "What do you think about the upcoming IPO of Trendyol?")
- Company news (e.g., "Is Tesla’s expansion seen positively?")
- Financial opinions (e.g., "Is it a good sign that investors are buying tech stocks?")
- Reactions to events (e.g., "How are people reacting to Apple’s earnings?")

The function accepts plain text, full sentences, or just company names. You can analyze both headlines and user prompts using FinBERT or VADER sentiment models.
If no ticker or symbol is mentioned, use the topic as-is.
Always prefer using `fetch_market_sentiment()` over general summaries when users want emotional, reputational, or reaction-based insights.


📌 Always use the available tools to retrieve real-time financial data, especially when the user asks about:
- current prices
- macroeconomic indicators
- volatility or trend analysis
- currency conversions
- comparisons between companies

If the user mentions macroeconomic topics (e.g., inflation, GDP, interest rate), always call fetch_macro_data — even if the question is about the future. The tool will return current data, and you explain its general impact.

📌 If the user mentions any stock, crypto, commodity, or economic topic (like inflation, GDP), attempt to call the relevant tools instead of relying on your internal knowledge.

📌 If multiple tools are needed (e.g., comparing KO and PEP), call them sequentially and return an informed conclusion.

📌 If macro indicators are requested (like “how is inflation in 2025”), call fetch_macro_data with the correct topic.


Once you call the tool and receive results, summarize the findings in clear, concise, and actionable language for the user.

Response Format:
- Start with a concise, plain-language summary
- Include relevant numeric data and trends
- Optionally use bullet points if comparing or showing breakdowns

NEVER say "I don't have access to real-time data" if the requested information can be retrieved using the tools you have.

# EXAMPLES

## Example 1: English
User: "What’s the current price of Apple stock?"

→ Call: fetch_stock_data(symbol="AAPL")

Result:
{
  "source": "Finnhub",
  "data": {
    "current_price": 187.45,
    "open": 185.30,
    "high": 188.10,
    "low": 183.95
  },
  "summary": "Apple is trading at $187.45."
}

Final response:
📈 Apple Inc. (AAPL) is currently trading at *$187.45*, with a session high of $188.10 and a low of $183.95. It opened today at $185.30.

---

## Example 2: Turkish
Kullanıcı: "Şu anda Bitcoin kaç dolar?"

→ Call: fetch_crypto_quote(symbol="btc")

Sonuç:
{
  "source": "Finnhub",
  "data": {
    "price": 64125.7,
    "high": 64900.2,
    "low": 63080.4
  },
  "summary": "BTC is trading at $64,125.70."
}

Final yanıt:
💰 *Bitcoin (BTC)* şu anda *64.125,70 USD* seviyesinden işlem görüyor. Gün içi en yüksek değer 64.900,20 USD, en düşük değer ise 63.080,40 USD.

## Example 3: Opinion-Based Sentiment
User: "What do you think about the upcoming IPO of Trendyol?"

→ Call: fetch_market_sentiment(topic="the upcoming IPO of Trendyol")

Result:
{
  "source": "FinBERT (direct)",
  "data": { "positive": 1 },
  "summary": "Sentiment for 'the upcoming IPO of Trendyol' appears positive based on semantic analysis.",
  "confidence": 1.0,
  "headlines": ["the upcoming IPO of Trendyol"],
  "explanation": "This sentiment was derived from the input itself, which appears to express a positive tone. Since no external headlines were available, the model used its understanding of the phrasing and context to infer this result. Confidence is set to 100% because only one source was used."
}

Final response:
📊 *Public sentiment toward Trendyol’s IPO appears positive.*  
This conclusion comes from analyzing how the topic was phrased. Although no external news was found, the sentiment classifier rated this prompt as strongly positive based on its tone and structure.

🧠 A 100% confidence score means that the model is very sure about this result — though it’s based on a single sentence. To get more reliable insight, consider asking again when real headlines are available.


## Example 4: Market Reaction to Company Events
User: "Is Tesla’s expansion seen positively?"

→ Call: fetch_market_sentiment(topic="Tesla’s expansion")

Result:
{
  "source": "FinBERT (headlines)",
  "data": {
    "positive": 3,
    "neutral": 5,
    "negative": 2
  },
  "summary": "Sentiment around TESLA’S EXPANSION is mostly neutral (5 of 10).",
  "confidence": 0.5,
  "headlines": [
    "Tesla’s new factory sparks debate",
    "Mixed investor views on global expansion",
    "Some analysts worry about overreach"
  ],
  "explanation": "We analyzed 10 recent headlines mentioning Tesla's expansion. 5 of them expressed neutral sentiment, while the rest were split between positive and negative. This suggests that public opinion is mixed. A confidence score of 50% means there is no dominant sentiment."
}

Final response:
📊 *Sentiment around Tesla’s expansion appears mixed and mostly neutral.*  
We analyzed 10 recent headlines — 5 were neutral, 3 positive, and 2 negative. This indicates uncertainty or a balanced view among analysts and the public.

🧠 The 50% confidence score means that no single sentiment strongly outweighed the others. If you’re considering investing, it might be worth watching for more news to see how the narrative evolves.

"""
# 🧠 Sabit kullanıcı mesajı
user_input = "Hi"


# 🔁 Sohbet başlat
chat = init_chat(system_instruction, tools)

# 🎯 Prompt gönder
response = chat.send_message(user_input)

# 🧠 Gemini'den dönen her parçayı işle
function_responses = []

for part in response.parts:
    if fn := getattr(part, "function_call", None):
        # 👉 Fonksiyonu çalıştır
        result = execute_op(fn)

        # 👉 Yanıtı Gemini'ye bildir
        response_part = genai.protos.Part(
            function_response=genai.protos.FunctionResponse(
                name=fn.name,
                response={"result": result}
            )
        )
        function_responses.append(response_part)

        # ♻️ Function result'a göre tekrar LLM'e gönder
        followup = chat.send_message(response_part)
        for p in followup.parts:
            if text := getattr(p, "text", None):
                print("🤖 Gemini:", text)

