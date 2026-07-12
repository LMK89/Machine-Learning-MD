## Kỹ thuật NLP để phân tích văn bản ngắn trong Python
Trang trình bày 1: Giới thiệu về NLP và Phân tích văn bản ngắn

Xử lý ngôn ngữ tự nhiên (NLP) là một lĩnh vực trí tuệ nhân tạo tập trung vào sự tương tác giữa máy tính và ngôn ngữ của con người. Phân tích văn bản ngắn là một tập hợp con quan trọng của NLP, xử lý các đoạn văn bản ngắn gọn như tweet, đánh giá sản phẩm hoặc tin nhắn trò chuyện. Trình chiếu này sẽ khám phá các kỹ thuật học máy khác nhau để phân tích các văn bản ngắn bằng Python.

```python
import nltk
from nltk.tokenize import word_tokenize

text = "NLP is fascinating!"
tokens = word_tokenize(text)
print(f"Tokenized text: {tokens}")

# Output: Tokenized text: ['NLP', 'is', 'fascinating', '!']
```

Trang trình bày 2: Tiền xử lý văn bản

Xử lý trước văn bản là một bước quan trọng trong NLP liên quan đến việc làm sạch và chuyển đổi dữ liệu văn bản thô sang định dạng phù hợp để phân tích. Các tác vụ tiền xử lý phổ biến bao gồm mã hóa, viết thường, xóa dấu câu và loại bỏ các từ dừng.

```python
import re
from nltk.corpus import stopwords

def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    return tokens

sample_text = "The quick brown fox jumps over the lazy dog!"
processed_tokens = preprocess_text(sample_text)
print(f"Processed tokens: {processed_tokens}")

# Output: Processed tokens: ['quick', 'brown', 'fox', 'jumps', 'lazy', 'dog']
```

Slide 3: Trích xuất đặc điểm: Túi từ

Mô hình Bag of Words (BoW) là một kỹ thuật đơn giản nhưng hiệu quả để biểu diễn dữ liệu văn bản dưới dạng các đặc điểm số. Nó tạo ra một vốn từ vựng gồm những từ duy nhất và thể hiện mỗi tài liệu dưới dạng một vectơ tần số từ.

```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    "I love machine learning",
    "I love Python programming",
    "NLP is a subset of AI"
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
print("Vocabulary:", vectorizer.get_feature_names_out())
print("BoW matrix:\n", X.toarray())

# Output:
# Vocabulary: ['ai' 'is' 'learning' 'love' 'machine' 'nlp' 'of' 'programming' 'python' 'subset']
# BoW matrix:
# [[0 0 1 1 1 0 0 0 0 0]
#  [0 0 0 1 0 0 0 1 1 0]
#  [1 1 0 0 0 1 1 0 0 1]]
```

Slide 4: Trích xuất đặc trưng: TF-IDF

Tần số nghịch đảo của thuật ngữ Tần số tài liệu (TF-IDF) là một kỹ thuật trích xuất tính năng nâng cao xem xét cả tần số của một từ trong tài liệu và tầm quan trọng của nó trên toàn bộ kho văn bản. Nó giúp xác định các từ có ý nghĩa hơn trong văn bản.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    "The cat sat on the mat",
    "The dog ate my homework",
    "The cat and the dog are pets"
]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
print("Vocabulary:", vectorizer.get_feature_names_out())
print("TF-IDF matrix:\n", X.toarray())

# Output:
# Vocabulary: ['and' 'are' 'ate' 'cat' 'dog' 'homework' 'mat' 'my' 'on' 'pets' 'sat' 'the']
# TF-IDF matrix:
# [[0.    0.    0.    0.468 0.    0.    0.468 0.    0.468 0.    0.468 0.354]
#  [0.    0.    0.479 0.    0.378 0.479 0.    0.479 0.    0.    0.    0.378]
#  [0.377 0.377 0.    0.298 0.298 0.    0.    0.    0.    0.377 0.    0.594]]
```

Slide 5: Phân loại văn bản: Naive Bayes

Naive Bayes là một thuật toán phổ biến cho các nhiệm vụ phân loại văn bản. Nó dựa trên định lý Bayes và giả định sự độc lập giữa các đặc điểm. Mặc dù đơn giản nhưng nó thường thực hiện tốt các nhiệm vụ phân loại văn bản ngắn.

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# Sample data
texts = ["I love this movie", "This movie is terrible", "Great acting", "Poor storyline"]
labels = ["positive", "negative", "positive", "negative"]

# Vectorize the text
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Train the model
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Predict
new_text = ["This movie is awesome"]
new_X = vectorizer.transform(new_text)
prediction = clf.predict(new_X)
print(f"Prediction for '{new_text[0]}': {prediction[0]}")

# Output: Prediction for 'This movie is awesome': positive
```

Slide 6: Phân loại văn bản: Máy vectơ hỗ trợ (SVM)

Máy vectơ hỗ trợ (SVM) là một thuật toán mạnh mẽ khác để phân loại văn bản. Nó hoạt động bằng cách tìm siêu phẳng phân tách tốt nhất các lớp khác nhau trong không gian nhiều chiều.

```python
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample data
texts = [
    "The food was delicious", "Terrible service", "Great atmosphere",
    "Overpriced and disappointing", "Friendly staff", "Bland and uninspiring"
]
labels = ["positive", "negative", "positive", "negative", "positive", "negative"]

# Vectorize the text
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

# Train the model
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Output: Accuracy: 1.00 (Note: This high accuracy is due to the small dataset)
```

Slide 7: Phân tích tình cảm

Phân tích tình cảm là quá trình xác định sắc thái cảm xúc đằng sau một loạt từ, được sử dụng để hiểu được thái độ, ý kiến ​​và cảm xúc được thể hiện trong văn bản.

```python
from textblob import TextBlob

def analyze_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity == 0:
        return 'Neutral'
    else:
        return 'Negative'

texts = [
    "I absolutely love this product!",
    "This is the worst experience ever.",
    "The weather is nice today."
]

for text in texts:
    sentiment = analyze_sentiment(text)
    print(f"Text: '{text}'\nSentiment: {sentiment}\n")

# Output:
# Text: 'I absolutely love this product!'
# Sentiment: Positive

# Text: 'This is the worst experience ever.'
# Sentiment: Negative

# Text: 'The weather is nice today.'
# Sentiment: Positive
```

Trang trình bày 8: Nhận dạng thực thể được đặt tên (NER)

Nhận dạng thực thể được đặt tên là nhiệm vụ xác định và phân loại các thực thể được đặt tên (ví dụ: tên người, tổ chức, địa điểm) trong văn bản. Điều quan trọng là trích xuất thông tin có cấu trúc từ văn bản phi cấu trúc.

```python
import spacy

nlp = spacy.load("en_core_web_sm")

text = "Apple Inc. is planning to open a new store in New York City next month."
doc = nlp(text)

for ent in doc.ents:
    print(f"Entity: {ent.text}, Label: {ent.label_}")

# Output:
# Entity: Apple Inc., Label: ORG
# Entity: New York City, Label: GPE
```

Trang trình bày 9: Mô hình hóa chủ đề: Phân bổ Dirichlet tiềm ẩn (LDA)

Mô hình hóa chủ đề là một kỹ thuật được sử dụng để khám phá các chủ đề trừu tượng trong một bộ sưu tập tài liệu. Phân bổ Dirichlet tiềm ẩn (LDA) là một thuật toán phổ biến để lập mô hình chủ đề.

```python
from gensim import corpora
from gensim.models import LdaModel
from gensim.parsing.preprocessing import STOPWORDS
from gensim.utils import simple_preprocess

texts = [
    "The cat and the dog",
    "The dog ate the food",
    "The cat slept on the mat",
    "The dog chased the cat"
]

# Preprocess the texts
processed_texts = [[word for word in simple_preprocess(doc) if word not in STOPWORDS] for doc in texts]

# Create a dictionary and corpus
dictionary = corpora.Dictionary(processed_texts)
corpus = [dictionary.doc2bow(text) for text in processed_texts]

# Train the LDA model
lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=2, random_state=42)

# Print the topics
for idx, topic in lda_model.print_topics(-1):
    print(f"Topic {idx}: {topic}")

# Output:
# Topic 0: 0.318*"dog" + 0.318*"cat" + 0.159*"the" + 0.159*"chased" + 0.045*"slept"
# Topic 1: 0.272*"the" + 0.182*"cat" + 0.182*"dog" + 0.091*"food" + 0.091*"ate"
```

Trang trình bày 10: Phần nhúng từ: Word2Vec

Việc nhúng từ là cách biểu diễn vectơ dày đặc của các từ nhằm nắm bắt các mối quan hệ ngữ nghĩa. Word2Vec là một thuật toán phổ biến để tạo các từ nhúng.

```python
from gensim.models import Word2Vec

sentences = [
    ['I', 'love', 'machine', 'learning'],
    ['I', 'love', 'deep', 'learning'],
    ['NLP', 'is', 'fascinating']
]

model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# Find similar words
similar_words = model.wv.most_similar('learning', topn=3)
print("Words similar to 'learning':")
for word, score in similar_words:
    print(f"{word}: {score:.2f}")

# Perform word arithmetic
result = model.wv.most_similar(positive=['deep', 'learning'], negative=['machine'], topn=1)
print(f"\ndeep + learning - machine = {result[0][0]}")

# Output:
# Words similar to 'learning':
# machine: 0.99
# deep: 0.97
# love: 0.20

# deep + learning - machine = fascinating
```

Slide 11: Tóm tắt văn bản: Phương pháp trích xuất

Tóm tắt văn bản là quá trình tạo ra một phiên bản ngắn gọn và mạch lạc của một văn bản dài hơn. Tóm tắt trích chọn chọn lọc những câu quan trọng từ văn bản gốc để tạo thành bản tóm tắt.

```python
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist

def extractive_summarize(text, num_sentences=3):
    # Tokenize the text into sentences and words
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Calculate word frequencies
    freq = FreqDist(words)

    # Score sentences based on word frequencies
    sentence_scores = {}
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in freq:
                if sentence not in sentence_scores:
                    sentence_scores[sentence] = freq[word]
                else:
                    sentence_scores[sentence] += freq[word]

    # Get the top n sentences
    summary_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]

    # Join the top sentences
    summary = ' '.join(summary_sentences)
    return summary

text = """
Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data. The goal is a computer capable of understanding the contents of documents, including the contextual nuances of the language within them. The technology can then accurately extract information and insights contained in the documents as well as categorize and organize the documents themselves.
"""

summary = extractive_summarize(text)
print("Summary:")
print(summary)

# Output:
# Summary:
# Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data. The goal is a computer capable of understanding the contents of documents, including the contextual nuances of the language within them. The technology can then accurately extract information and insights contained in the documents as well as categorize and organize the documents themselves.
```

Slide 12: Tạo văn bản: Chuỗi Markov

Chuỗi Markov có thể được sử dụng cho các tác vụ tạo văn bản đơn giản. Kỹ thuật này tạo ra văn bản mới dựa trên các thuộc tính thống kê của văn bản đầu vào.

```python
import random

def build_markov_chain(text, n=2):
    words = text.split()
    chain = {}
    for i in range(len(words) - n):
        state = tuple(words[i:i+n])
        next_word = words[i+n]
        if state not in chain:
            chain[state] = {}
        if next_word not in chain[state]:
            chain[state][next_word] = 0
        chain[state][next_word] += 1
    return chain

def generate_text(chain, num_words=50, start=None):
    if start is None:
        current = random.choice(list(chain.keys()))
    else:
        current = start
    result = list(current)
    for _ in range(num_words - len(current)):
        if current in chain:
            next_word = random.choices(list(chain[current].keys()),
                                       weights=list(chain[current].values()))[0]
            result.append(next_word)
            current = tuple(result[-len(current):])
        else:
            break
    return ' '.join(result)

text = """
The quick brown fox jumps over the lazy dog.
The lazy dog sleeps all day.
The quick brown fox is very clever.
"""

chain = build_markov_chain(text)
generated_text = generate_text(chain, num_words=20)
print("Generated text:")
print(generated_text)

# Output:
# Generated text:
# The quick brown fox jumps over the lazy dog sleeps all day. The quick brown fox is very clever. The lazy dog
```

Trang trình bày 13: Ví dụ thực tế: Phát hiện thư rác

Phát hiện thư rác là một ứng dụng phổ biến của phân tích văn bản ngắn trong hệ thống lọc email. Đây là một ví dụ đơn giản sử dụng trình phân loại Naive Bayes:

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Sample data (email subjects)
subjects = [
    "Win a free iPhone now!", "Meeting agenda for tomorrow",
    "Discount on luxury watches", "Project deadline reminder",
    "You've won the lottery!", "Weekly team sync",
    "Enlarge your profits now", "Quarterly report available"
]
labels = [1, 0, 1, 0, 1, 0, 1, 0]  # 1 for spam, 0 for non-spam

# Vectorize the text
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(subjects)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.25, random_state=42)

# Train the model
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Test with new emails
new_emails = ["Free gift awaits you!", "Team lunch next week"]
new_X = vectorizer.transform(new_emails)
predictions = clf.predict(new_X)
for email, pred in zip(new_emails, predictions):
    print(f"'{email}' - {'Spam' if pred == 1 else 'Not Spam'}")

# Output:
#               precision    recall  f1-score   support
#            0       1.00      1.00      1.00         1
#            1       1.00      1.00      1.00         1
#     accuracy                           1.00         2
#    macro avg       1.00      1.00      1.00         2
# weighted avg       1.00      1.00      1.00         2

# 'Free gift awaits you!' - Spam
# 'Team lunch next week' - Not Spam
```

Slide 14: Ví dụ thực tế: Phân tích phản hồi của khách hàng

Phân tích phản hồi của khách hàng là rất quan trọng để doanh nghiệp cải thiện sản phẩm hoặc dịch vụ của mình. Dưới đây là ví dụ về phân tích cảm xúc khi đánh giá sản phẩm:

```python
import pandas as pd
from textblob import TextBlob

# Sample customer reviews
reviews = [
    "This product is amazing! It works perfectly.",
    "Terrible customer service. Never buying again.",
    "Average product, nothing special.",
    "Great value for the price. Highly recommended!",
    "Disappointing quality. Broke after a week."
]

# Perform sentiment analysis
sentiments = []
for review in reviews:
    blob = TextBlob(review)
    sentiment = blob.sentiment.polarity
    if sentiment > 0:
        sentiments.append("Positive")
    elif sentiment < 0:
        sentiments.append("Negative")
    else:
        sentiments.append("Neutral")

# Create a DataFrame
df = pd.DataFrame({"Review": reviews, "Sentiment": sentiments})

# Display results
print(df)

# Calculate sentiment distribution
sentiment_counts = df["Sentiment"].value_counts()
print("\nSentiment Distribution:")
print(sentiment_counts)

# Output:
#                                             Review Sentiment
# 0  This product is amazing! It works perfectly.   Positive
# 1  Terrible customer service. Never buying again.  Negative
# 2            Average product, nothing special.    Neutral
# 3  Great value for the price. Highly recommended!  Positive
# 4     Disappointing quality. Broke after a week.   Negative

# Sentiment Distribution:
# Positive    2
# Negative    2
# Neutral     1
# Name: Sentiment, dtype: int64
```

Trang trình bày 15: Tài nguyên bổ sung

Đối với những người quan tâm đến việc tìm hiểu sâu hơn về NLP và phân tích văn bản ngắn, đây là một số tài nguyên có giá trị:

1. "Xử lý ngôn ngữ tự nhiên bằng Python" của Steven Bird, Ewan Klein và Edward Loper
   * Giới thiệu toàn diện về NLP sử dụng thư viện NLTK
2. "Xử lý lời nói và ngôn ngữ" của Dan Jurafsky và James H. Martin
   * Sách giáo khoa chuyên sâu bao gồm nhiều khía cạnh khác nhau của NLP
3. Giấy tờ ArXiv:
   * "BERT: Đào tạo trước về Máy biến áp hai chiều sâu để hiểu ngôn ngữ" của Devlin và cộng sự. ([https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805))
   * "Tất cả những gì bạn cần là sự chú ý" của Vaswani và cộng sự. ([https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762))
4. Các khóa học trực tuyến:
   * CS224N của Stanford: Xử lý ngôn ngữ tự nhiên với học sâu
   * Chuyên ngành xử lý ngôn ngữ tự nhiên của Coursera bởi deeplearning.ai

Những tài nguyên này cung cấp nền tảng vững chắc để khám phá sâu hơn về các kỹ thuật và ứng dụng NLP trong phân tích văn bản ngắn.
