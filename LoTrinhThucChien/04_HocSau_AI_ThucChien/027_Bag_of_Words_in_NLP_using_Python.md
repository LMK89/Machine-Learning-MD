## Túi từ trong NLP sử dụng Python
Slide 1: Giới thiệu về Bag of Words (BoW) trong NLP

Bag of Words là một kỹ thuật cơ bản trong Xử lý ngôn ngữ tự nhiên, thể hiện văn bản dưới dạng tập hợp các từ, không quan tâm đến ngữ pháp và trật tự từ. Phương pháp này được sử dụng để tạo các vectơ đặc trưng cho các tác vụ phân loại văn bản, phân tích tình cảm và truy xuất thông tin.

```python
from collections import Counter

text = "The quick brown fox jumps over the lazy dog"
bow = Counter(text.lower().split())
print(bow)
```

Trang trình bày 2: Token hóa: Bước đầu tiên

Mã thông báo là quá trình chia nhỏ văn bản thành các từ hoặc mã thông báo riêng lẻ. Đây là một bước quan trọng trong việc tạo ra một bản trình bày Bag of Words. Chúng tôi sẽ sử dụng thư viện NLTK để mã thông báo nâng cao hơn.

```python
import nltk
nltk.download('punkt')

text = "The quick brown fox, jumps over the lazy dog!"
tokens = nltk.word_tokenize(text)
print(tokens)
```

Slide 3: Tạo từ vựng

Sau khi token hóa, chúng ta cần tạo ra một vốn từ vựng gồm những từ duy nhất. Từ vựng này sẽ được sử dụng để tạo các vectơ đặc trưng của chúng tôi.

```python
corpus = [
    "The quick brown fox jumps over the lazy dog",
    "The lazy dog sleeps all day",
    "The quick rabbit runs fast"
]

vocabulary = set()
for sentence in corpus:
    vocabulary.update(sentence.lower().split())

print(f"Vocabulary size: {len(vocabulary)}")
print(f"Vocabulary: {vocabulary}")
```

Slide 4: Mã hóa văn bản dưới dạng vectơ BoW

Khi đã có vốn từ vựng, chúng ta có thể mã hóa từng văn bản dưới dạng vectơ tần số từ.

```python
def bow_encoding(text, vocabulary):
    vector = {word: 0 for word in vocabulary}
    for word in text.lower().split():
        if word in vector:
            vector[word] += 1
    return vector

vocabulary = list(vocabulary)  # Convert set to list for consistent ordering
encoded_texts = [bow_encoding(text, vocabulary) for text in corpus]

for i, encoded_text in enumerate(encoded_texts):
    print(f"Text {i + 1}: {encoded_text}")
```

Slide 5: Xử lý các từ dừng

Từ dừng là những từ phổ biến thường không đóng góp nhiều vào ý nghĩa của văn bản. Loại bỏ chúng có thể cải thiện hiệu suất của các mô hình NLP.

```python
from nltk.corpus import stopwords
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

text = "The quick brown fox jumps over the lazy dog"
tokens = nltk.word_tokenize(text.lower())
filtered_tokens = [word for word in tokens if word not in stop_words]

print(f"Original: {tokens}")
print(f"Filtered: {filtered_tokens}")
```

Slide 6: Từ gốc và bổ ngữ

Bắt nguồn và từ vựng hóa làm giảm các từ về dạng cơ sở hoặc dạng gốc của chúng, điều này có thể giúp tạo ra các cách biểu diễn BoW có ý nghĩa hơn.

```python
from nltk.stem import PorterStemmer, WordNetLemmatizer
nltk.download('wordnet')

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

words = ["running", "runs", "ran", "easily", "fairly"]

print("Original | Stemmed | Lemmatized")
for word in words:
    print(f"{word:9} | {stemmer.stem(word):7} | {lemmatizer.lemmatize(word)}")
```

Slide 7: TF-IDF: Cải thiện BoW

Tần số tài liệu nghịch đảo tần số (TF-IDF) là một cải tiến so với BoW đơn giản. Nó xem xét tầm quan trọng của các từ trong toàn bộ kho ngữ liệu.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    "The quick brown fox jumps over the lazy dog",
    "The lazy dog sleeps all day",
    "The quick rabbit runs fast"
]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

print("TF-IDF matrix:")
print(X.toarray())
print("\nFeature names:")
print(vectorizer.get_feature_names_out())
```

Slide 8: Triển khai BoW để phân loại văn bản

Hãy sử dụng BoW cho tác vụ phân loại văn bản đơn giản bằng trình phân loại Naive Bayes.

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Sample data
X = [
    "I love this product",
    "This is terrible",
    "Great customer service",
    "Poor quality",
    "Excellent experience"
]
y = [1, 0, 1, 0, 1]  # 1 for positive, 0 for negative

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create BoW representation
vectorizer = CountVectorizer()
X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)

# Train and evaluate the model
clf = MultinomialNB()
clf.fit(X_train_bow, y_train)
print(f"Accuracy: {clf.score(X_test_bow, y_test)}")
```

Slide 9: Trực quan hóa BoW bằng Word Clouds

Các đám mây từ cung cấp sự trình bày trực quan về tần số từ trong kho văn bản, điều này có thể hữu ích để hiểu các thuật ngữ phổ biến nhất trong mô hình BoW.

```python
from wordcloud import WordCloud
import matplotlib.pyplot as plt

text = "The quick brown fox jumps over the lazy dog. " * 10
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud Representation of BoW')
plt.show()
```

Slide 10: Xử lý từ ngoài từ vựng

Khi áp dụng mô hình BoW cho văn bản mới, chúng ta có thể gặp phải những từ không có trong vốn từ vựng ban đầu của mình. Hãy cùng khám phá cách xử lý tình huống này.

```python
def bow_encoding(text, vocabulary):
    vector = {word: 0 for word in vocabulary}
    for word in text.lower().split():
        if word in vector:
            vector[word] += 1
        else:
            vector['<UNK>'] = vector.get('<UNK>', 0) + 1
    return vector

vocabulary = set(['quick', 'brown', 'fox', 'jumps', 'lazy', 'dog', '<UNK>'])
new_text = "The fast red fox leaps over the sleepy cat"

encoded_text = bow_encoding(new_text, vocabulary)
print(encoded_text)
```

Slide 11: N-gram: Nắm bắt thứ tự từ

N-gram mở rộng mô hình BoW bằng cách xem xét các chuỗi N từ, có thể nắm bắt một số thông tin về thứ tự từ.

```python
from nltk import ngrams

text = "The quick brown fox jumps over the lazy dog"
tokens = text.split()

print("Unigrams:", list(ngrams(tokens, 1)))
print("Bigrams:", list(ngrams(tokens, 2)))
print("Trigrams:", list(ngrams(tokens, 3)))
```

Trang trình bày 12: Ví dụ thực tế: Phát hiện thư rác

Hãy sử dụng BoW cho nhiệm vụ phát hiện thư rác thực tế bằng cách sử dụng tập dữ liệu công khai.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Load the SMS Spam Collection dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
data = pd.read_csv(url, sep='\t', names=['label', 'message'])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.2, random_state=42)

# Create BoW representation
vectorizer = CountVectorizer()
X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)

# Train and evaluate the model
clf = MultinomialNB()
clf.fit(X_train_bow, y_train)
y_pred = clf.predict(X_test_bow)

print(classification_report(y_test, y_pred))
```

Slide 13: Ví dụ thực tế: Tính tương đồng của tài liệu

BoW có thể được sử dụng để đo lường mức độ tương tự giữa các tài liệu, điều này rất hữu ích cho các nhiệm vụ như hệ thống đề xuất hoặc phát hiện đạo văn.

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

documents = [
    "The cat sits on the mat",
    "The dog jumps over the fence",
    "The cat chases the mouse",
    "The bird flies in the sky"
]

vectorizer = CountVectorizer()
bow_matrix = vectorizer.fit_transform(documents)

similarity_matrix = cosine_similarity(bow_matrix)

print("Document Similarity Matrix:")
print(similarity_matrix)

# Find the most similar pair of documents
max_similarity = 0
max_pair = None
for i in range(len(documents)):
    for j in range(i+1, len(documents)):
        if similarity_matrix[i][j] > max_similarity:
            max_similarity = similarity_matrix[i][j]
            max_pair = (i, j)

print(f"\nMost similar documents: {max_pair}")
print(f"Similarity score: {max_similarity}")
print(f"Doc 1: {documents[max_pair[0]]}")
print(f"Doc 2: {documents[max_pair[1]]}")
```

Slide 14: Hạn chế của túi từ

Mặc dù BoW đơn giản và hiệu quả nhưng nó có những hạn chế:

1. Mất thông tin trật tự từ
2. Không có khả năng nắm bắt ngữ nghĩa
3. Tính chiều cao cho vốn từ vựng lớn
4. Nhạy cảm trong việc lựa chọn từ vựng

Những hạn chế này đã dẫn đến sự phát triển của các kỹ thuật tiên tiến hơn như nhúng từ (ví dụ: Word2Vec, GloVe) và các mô hình dựa trên máy biến áp (ví dụ: BERT, GPT).

```python
# Demonstrating loss of word order
sentence1 = "The cat chases the mouse"
sentence2 = "The mouse chases the cat"

bow1 = Counter(sentence1.lower().split())
bow2 = Counter(sentence2.lower().split())

print("BoW for sentence 1:", bow1)
print("BoW for sentence 2:", bow2)
print("Are the BoW representations identical?", bow1 == bow2)
```

Trang trình bày 15: Tài nguyên bổ sung

Để khám phá thêm về Bag of Words và các kỹ thuật NLP có liên quan, hãy xem xét các tài nguyên sau:

1. "Ước tính hiệu quả các biểu diễn từ trong không gian vectơ" của Mikolov và cộng sự. (2013) ArXiv: [https://arxiv.org/abs/1301.3781](https://arxiv.org/abs/1301.3781)
2. "GloVe: Các vectơ toàn cầu để thể hiện từ" của Pennington và cộng sự. (2014) ArXiv: [https://arxiv.org/abs/1405.4053](https://arxiv.org/abs/1405.4053)
3. "BERT: Đào tạo trước về Máy biến áp hai chiều sâu để hiểu ngôn ngữ" của Devlin và cộng sự. (2018) ArXiv: [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)

Các bài viết này giới thiệu các kỹ thuật tiên tiến hơn nhằm giải quyết một số hạn chế của mô hình Bag of Words cơ bản.
