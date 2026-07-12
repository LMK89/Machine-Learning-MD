## Mô hình NLP để phân tích tình cảm trong Python
Trang trình bày 1: Giới thiệu về Mô hình NLP để phân tích tình cảm

Các mô hình Xử lý ngôn ngữ tự nhiên (NLP) đã cách mạng hóa việc phân tích cảm xúc, cho phép máy móc hiểu và diễn giải cảm xúc của con người trong văn bản. Bài trình bày này khám phá năm mô hình mạnh mẽ: BERT, RoBERTa, DistilBERT, ALBERT và XLNet. Chúng ta sẽ đi sâu vào kiến ​​trúc, trường hợp sử dụng và cách triển khai của chúng bằng Python, cung cấp các ví dụ thực tế cho các nhiệm vụ phân tích cảm tính.

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

# Example usage
model_name = "bert-base-uncased"
tokenizer, model = load_model(model_name)
print(f"Loaded {model_name} model and tokenizer")
```

Trang trình bày 2: BERT (Biểu diễn bộ mã hóa hai chiều từ máy biến áp)

BERT, do Google phát triển, là một mô hình dựa trên máy biến áp, học cách nhúng từ theo ngữ cảnh bằng cách xem xét cả ngữ cảnh bên trái và bên phải. Nó sử dụng mô hình ngôn ngữ đeo mặt nạ và dự đoán câu tiếp theo để đào tạo trước. Bản chất hai chiều của BERT làm cho nó có hiệu quả cao đối với các nhiệm vụ NLP khác nhau, bao gồm cả phân tích tình cảm.

```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

text = "I love this movie! It's fantastic."
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    positive_score = predictions[0][2].item()

print(f"Positive sentiment score: {positive_score:.4f}")
```

Trang trình bày 3: RoBERTa (Phương pháp tiếp cận BERT được tối ưu hóa mạnh mẽ)

RoBERTa, được Facebook AI giới thiệu, là phiên bản tối ưu hóa của BERT. Nó loại bỏ nhiệm vụ dự đoán câu tiếp theo, sử dụng mặt nạ động và được huấn luyện trên các tập dữ liệu lớn hơn với các chuỗi dài hơn. Những cải tiến này dẫn đến hiệu suất tốt hơn trên các nhiệm vụ NLP khác nhau, bao gồm cả phân tích tình cảm.

```python
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3)

text = "This product exceeded my expectations. Highly recommended!"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    positive_score = predictions[0][2].item()

print(f"Positive sentiment score: {positive_score:.4f}")
```

Slide 4: DistilBERT (BERT chưng cất)

DistilBERT là phiên bản BERT nhẹ hơn và nhanh hơn, được phát triển bởi Hugging Face. Nó giữ lại 97% hiệu suất của BERT trong khi nhỏ hơn 40% và nhanh hơn 60%. Điều này khiến nó trở nên lý tưởng cho các môi trường có nguồn lực hạn chế hoặc các ứng dụng phân tích cảm tính theo thời gian thực.

```python
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)

text = "The customer service was terrible. I'm very disappointed."
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    negative_score = predictions[0][0].item()

print(f"Negative sentiment score: {negative_score:.4f}")
```

Slide 5: ALBERT (A Lite BERT)

ALBERT, được phát triển bởi Google Research, là một phiên bản nhẹ khác của BERT. Nó sử dụng các kỹ thuật chia sẻ tham số và tham số hóa nhúng theo hệ số để giảm kích thước mô hình trong khi vẫn duy trì hiệu suất. ALBERT đặc biệt hữu ích cho các nhiệm vụ phân tích tình cảm cần triển khai trên quy mô lớn.

```python
from transformers import AlbertTokenizer, AlbertForSequenceClassification
import torch

tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
model = AlbertForSequenceClassification.from_pretrained('albert-base-v2', num_labels=3)

text = "The restaurant was okay, but nothing special."
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    neutral_score = predictions[0][1].item()

print(f"Neutral sentiment score: {neutral_score:.4f}")
```

Trang trình bày 6: XLNet (NET học tập eXtreme)

XLNet, được phát triển bởi Đại học Carnegie Mellon và Google Brain, là một mô hình ngôn ngữ tự hồi quy khắc phục các hạn chế của BERT bằng cách sử dụng mô hình ngôn ngữ hoán vị. Cách tiếp cận này cho phép XLNet nắm bắt bối cảnh hai chiều mà không cần đầu vào bị che, có khả năng dẫn đến cải thiện hiệu suất trong các nhiệm vụ phân tích cảm tính.

```python
from transformers import XLNetTokenizer, XLNetForSequenceClassification
import torch

tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased', num_labels=3)

text = "I can't believe how amazing this experience was!"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    positive_score = predictions[0][2].item()

print(f"Positive sentiment score: {positive_score:.4f}")
```

Trang trình bày 7: Tinh chỉnh phân tích cảm xúc

Tinh chỉnh các mô hình được đào tạo trước này trên tập dữ liệu phân tích tình cảm cụ thể có thể cải thiện đáng kể hiệu suất của chúng. Dưới đây là ví dụ về tinh chỉnh BERT để phân tích cảm tính bằng cách sử dụng tập dữ liệu tùy chỉnh.

```python
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Example usage (you would need to prepare your own dataset)
texts = ["I love this!", "I hate this!", "It's okay."]
labels = [2, 0, 1]  # 2: positive, 0: negative, 1: neutral

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

dataset = SentimentDataset(texts, labels, tokenizer, max_length=128)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()
```

Trang trình bày 8: Tiền xử lý dữ liệu để phân tích cảm xúc

Việc xử lý trước dữ liệu thích hợp là rất quan trọng để phân tích tình cảm hiệu quả. Trang trình bày này trình bày các kỹ thuật tiền xử lý phổ biến bằng thư viện NLTK của Python.

```python
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return ' '.join(tokens)

# Example usage
raw_text = "I absolutely loved the movie! It was amazing and thrilling. 10/10 would recommend!"
processed_text = preprocess_text(raw_text)
print(f"Original: {raw_text}")
print(f"Processed: {processed_text}")
```

Trang trình bày 9: Các phương pháp tập hợp để phân tích cảm xúc

Việc kết hợp nhiều mô hình thường có thể dẫn đến cải thiện hiệu suất trong phân tích cảm tính. Trang trình bày này trình bày cách tạo một tập hợp các mô hình khác nhau để có những dự đoán mạnh mẽ hơn.

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class SentimentEnsemble:
    def __init__(self, model_names):
        self.models = []
        self.tokenizers = []
        for name in model_names:
            tokenizer = AutoTokenizer.from_pretrained(name)
            model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=3)
            self.models.append(model)
            self.tokenizers.append(tokenizer)

    def predict(self, text):
        predictions = []
        for model, tokenizer in zip(self.models, self.tokenizers):
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
                pred = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predictions.append(pred)

        # Average predictions
        ensemble_pred = torch.mean(torch.stack(predictions), dim=0)
        return ensemble_pred

# Example usage
ensemble = SentimentEnsemble(['bert-base-uncased', 'roberta-base', 'distilbert-base-uncased'])
text = "This product is absolutely fantastic! I couldn't be happier with my purchase."
prediction = ensemble.predict(text)
sentiment = ["Negative", "Neutral", "Positive"][prediction.argmax().item()]
confidence = prediction.max().item()

print(f"Sentiment: {sentiment}")
print(f"Confidence: {confidence:.4f}")
```

Trang trình chiếu 10: Ví dụ thực tế: Phân tích cảm xúc trên mạng xã hội

Trong ví dụ này, chúng tôi sẽ phân tích cảm tính từ dữ liệu Twitter bằng mô hình BERT. Điều này có thể hữu ích cho việc giám sát thương hiệu, phân tích phản hồi của khách hàng hoặc dự đoán xu hướng.

```python
import tweepy
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Twitter API credentials (you need to obtain these from Twitter Developer Portal)
consumer_key = "your_consumer_key"
consumer_secret = "your_consumer_secret"
access_token = "your_access_token"
access_token_secret = "your_access_token_secret"

# Authenticate with Twitter API
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# Load BERT model for sentiment analysis
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment = ["Negative", "Neutral", "Positive"][predictions.argmax().item()]
    return sentiment

# Analyze tweets for a specific topic
topic = "artificial intelligence"
tweets = api.search_tweets(q=topic, lang="en", count=100)

sentiments = []
for tweet in tweets:
    sentiment = analyze_sentiment(tweet.text)
    sentiments.append(sentiment)

# Calculate sentiment distribution
sentiment_dist = {
    "Positive": sentiments.count("Positive") / len(sentiments),
    "Neutral": sentiments.count("Neutral") / len(sentiments),
    "Negative": sentiments.count("Negative") / len(sentiments)
}

print(f"Sentiment distribution for '{topic}':")
for sentiment, percentage in sentiment_dist.items():
    print(f"{sentiment}: {percentage:.2%}")
```

Slide 11: Ví dụ thực tế: Phân tích đánh giá của khách hàng

Trong ví dụ này, chúng tôi sẽ sử dụng RoBERTa để phân tích đánh giá của khách hàng về một sản phẩm, giúp doanh nghiệp hiểu được cảm nhận của khách hàng và xác định các lĩnh vực cần cải thiện.

```python
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
import pandas as pd
import matplotlib.pyplot as plt

# Load RoBERTa model
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3)

def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment = ["Negative", "Neutral", "Positive"][predictions.argmax().item()]
    confidence = predictions.max().item()
    return sentiment, confidence

# Sample customer reviews
reviews = [
    "This product is amazing! It exceeded all my expectations.",
    "Not bad, but could be better. There's room for improvement.",
    "Terrible experience. I regret buying this product.",
    "It's okay, nothing special but gets the job done.",
    "Absolutely love it! Best purchase I've made in years."
]

# Analyze sentiments
results = [{"review": review, "sentiment": analyze_sentiment(review)[0],
            "confidence": analyze_sentiment(review)[1]} for review in reviews]

# Create a DataFrame for easy analysis
df = pd.DataFrame(results)

# Calculate and display sentiment distribution
sentiment_dist = df['sentiment'].value_counts(normalize=True)
print("Sentiment Distribution:")
print(sentiment_dist)

# Display top positive and negative reviews
print("\nTop Positive Review:")
print(df[df['sentiment'] == 'Positive'].sort_values('confidence', ascending=False)['review'].iloc[0])
print("\nTop Negative Review:")
print(df[df['sentiment'] == 'Negative'].sort_values('confidence', ascending=False)['review'].iloc[0])
```

Slide 12: Xử lý phân tích cảm xúc đa ngôn ngữ

Khi các doanh nghiệp mở rộng trên toàn cầu, khả năng phân tích tình cảm bằng nhiều ngôn ngữ trở nên quan trọng. Trang trình bày này trình bày cách sử dụng mô hình đa ngôn ngữ để phân tích cảm tính trên các ngôn ngữ khác nhau.

```python
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
import torch

# Load multilingual XLM-RoBERTa model
tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
model = XLMRobertaForSequenceClassification.from_pretrained('xlm-roberta-base', num_labels=3)

def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment = ["Negative", "Neutral", "Positive"][predictions.argmax().item()]
    confidence = predictions.max().item()
    return sentiment, confidence

# Example reviews in different languages
reviews = {
    "English": "This product is fantastic!",
    "Spanish": "Este producto es fantástico!",
    "French": "Ce produit est fantastique!",
    "German": "Dieses Produkt ist fantastisch!",
    "Chinese": "这个产品太棒了！"
}

# Analyze sentiments
for language, review in reviews.items():
    sentiment, confidence = analyze_sentiment(review)
    print(f"{language}: {review}")
    print(f"Sentiment: {sentiment}, Confidence: {confidence:.4f}\n")
```

Trang trình bày 13: Phân tích cảm xúc dựa trên khía cạnh

Phân tích tình cảm dựa trên khía cạnh cho phép chúng tôi xác định tình cảm đối với các khía cạnh cụ thể của sản phẩm hoặc dịch vụ. Trang trình bày này trình bày một cách tiếp cận đơn giản bằng cách sử dụng BERT và nhận dạng thực thể được đặt tên.

```python
from transformers import pipeline
import spacy

# Load BERT sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

# Load spaCy for named entity recognition
nlp = spacy.load("en_core_web_sm")

def aspect_based_sentiment(text):
    # Perform named entity recognition
    doc = nlp(text)
    aspects = [ent.text for ent in doc.ents if ent.label_ in ["PRODUCT", "ORG"]]

    # Analyze sentiment for each aspect
    results = {}
    for aspect in aspects:
        # Find sentences containing the aspect
        sentences = [sent.text for sent in doc.sents if aspect.lower() in sent.text.lower()]
        if sentences:
            # Analyze sentiment for these sentences
            sentiments = sentiment_analyzer(sentences)
            avg_sentiment = sum(s['score'] for s in sentiments) / len(sentiments)
            results[aspect] = "Positive" if avg_sentiment > 0.5 else "Negative"

    return results

# Example usage
review = "The new iPhone camera is amazing, but the battery life is disappointing. Apple's customer service was helpful though."
aspects_sentiment = aspect_based_sentiment(review)

print("Aspect-based sentiments:")
for aspect, sentiment in aspects_sentiment.items():
    print(f"{aspect}: {sentiment}")
```

Trang trình bày 14: Phân tích tình cảm để theo dõi phương tiện truyền thông xã hội

Giám sát phương tiện truyền thông xã hội là rất quan trọng để quản lý thương hiệu và thu hút khách hàng. Trang trình bày này trình bày cách sử dụng phân tích cảm tính để theo dõi phương tiện truyền thông xã hội theo thời gian thực.

```python
import tweepy
from transformers import pipeline
import time

# Twitter API credentials (replace with your own)
consumer_key = "your_consumer_key"
consumer_secret = "your_consumer_secret"
access_token = "your_access_token"
access_token_secret = "your_access_token_secret"

# Authenticate with Twitter API
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# Load sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

class TweetListener(tweepy.StreamListener):
    def on_status(self, status):
        if hasattr(status, 'retweeted_status'):
            return

        tweet = status.text
        sentiment = sentiment_analyzer(tweet)[0]

        print(f"Tweet: {tweet}")
        print(f"Sentiment: {sentiment['label']}, Score: {sentiment['score']:.4f}")
        print("-" * 50)

    def on_error(self, status_code):
        if status_code == 420:
            return False

# Set up stream listener
stream_listener = TweetListener()
stream = tweepy.Stream(auth=api.auth, listener=stream_listener)

# Start streaming tweets (replace with your desired keywords)
stream.filter(track=["your_brand_name", "your_product_name"], languages=["en"])
```

Trang trình bày 15: Tài nguyên bổ sung

Đối với những người quan tâm đến việc tìm hiểu sâu hơn về các mô hình NLP để phân tích tình cảm, đây là một số tài nguyên có giá trị:

1. BERT: Trình bày bộ mã hóa hai chiều từ Transformers ArXiv: [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
2. RoBERTa: Phương pháp tiếp cận đào tạo trước BERT được tối ưu hóa mạnh mẽ ArXiv: [https://arxiv.org/abs/1907.11692](https://arxiv.org/abs/1907.11692)
3. DistilBERT: phiên bản chưng cất của BERT: nhỏ hơn, nhanh hơn, rẻ hơn và nhẹ hơn ArXiv: [https://arxiv.org/abs/1910.01108](https://arxiv.org/abs/1910.01108)
4. ALBERT: BERT rút gọn để tự học cách biểu diễn ngôn ngữ ArXiv: [https://arxiv.org/abs/1909.11942](https://arxiv.org/abs/1909.11942)
5. XLNet: Đào tạo trước tự hồi quy tổng quát để hiểu ngôn ngữ ArXiv: [https://arxiv.org/abs/1906.08237](https://arxiv.org/abs/1906.08237)

Các bài viết này cung cấp những giải thích sâu sắc về các mô hình mà chúng ta đã thảo luận, bao gồm kiến ​​trúc, quy trình đào tạo và so sánh hiệu suất của chúng. Chúng đóng vai trò là điểm khởi đầu tuyệt vời để hiểu nền tảng lý thuyết của các mô hình NLP mạnh mẽ này.
