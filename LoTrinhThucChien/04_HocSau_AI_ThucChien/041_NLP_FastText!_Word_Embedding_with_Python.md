## Văn bản nhanh NLP! Nhúng từ bằng Python
Slide 1: Giới thiệu về FastText

FastText là thư viện mã nguồn mở được phát triển bởi phòng thí nghiệm Nghiên cứu AI của Facebook để học cách trình bày từ và phân loại câu một cách hiệu quả. Nó mở rộng mô hình word2vec bằng cách biểu diễn mỗi từ dưới dạng một túi ký tự n-gram, cho phép nó nắm bắt thông tin từ phụ và xử lý các từ ngoài từ vựng một cách hiệu quả.

```python
import fasttext

# Train a FastText model
model = fasttext.train_unsupervised('corpus.txt', model='skipgram')

# Get word vector
word_vector = model.get_word_vector('example')
print(word_vector)
```

Trang trình bày 2: Kiến thức cơ bản về nhúng Word

Việc nhúng từ là cách biểu diễn vectơ dày đặc của các từ trong không gian vectơ liên tục. Chúng nắm bắt các mối quan hệ ngữ nghĩa giữa các từ, cho phép các từ tương tự có cách biểu diễn vectơ tương tự. FastText xây dựng dựa trên khái niệm này bằng cách kết hợp thông tin từ phụ.

```python
import numpy as np
import matplotlib.pyplot as plt

# Simplified word embedding visualization
words = ['king', 'queen', 'man', 'woman']
embeddings = np.random.rand(4, 2)  # 2D for visualization

plt.figure(figsize=(10, 8))
plt.scatter(embeddings[:, 0], embeddings[:, 1])
for i, word in enumerate(words):
    plt.annotate(word, (embeddings[i, 0], embeddings[i, 1]))
plt.title('Word Embeddings Visualization')
plt.show()
```

Trang trình bày 3: Kiến trúc mô hình FastText

FastText sử dụng mạng thần kinh nông với lớp đầu vào, lớp ẩn và lớp đầu ra. Lớp đầu vào biểu thị các từ hoặc n-gram, lớp ẩn tìm hiểu các phần nhúng và lớp đầu ra dự đoán các từ ngữ cảnh hoặc nhãn cho các tác vụ phân loại.

```python
import torch.nn as nn

class SimplifiedFastText(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SimplifiedFastText, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        output = self.linear(embeds.mean(dim=1))
        return output

# Usage
model = SimplifiedFastText(vocab_size=10000, embedding_dim=100)
```

Slide 4: Thông tin từ phụ

FastText biểu diễn các từ dưới dạng các túi ký tự n-gram, cho phép nó nắm bắt thông tin hình thái và xử lý các từ nằm ngoài từ vựng. Cách tiếp cận này đặc biệt hữu ích cho các ngôn ngữ có hình thái phong phú hoặc để xử lý các từ hiếm.

```python
def get_ngrams(word, n_min=3, n_max=6):
    ngrams = []
    word = "<" + word + ">"
    for n in range(n_min, min(len(word), n_max) + 1):
        for i in range(len(word) - n + 1):
            ngrams.append(word[i:i+n])
    return ngrams

word = "example"
print(f"N-grams for '{word}':", get_ngrams(word))
```

Slide 5: Huấn luyện mô hình FastText

Huấn luyện mô hình FastText bao gồm việc chuẩn bị một kho dữ liệu, thiết lập siêu tham số và sử dụng chế độ học có giám sát hoặc không giám sát. Chế độ không giám sát học cách biểu diễn từ, trong khi chế độ giám sát được sử dụng để phân loại văn bản.

```python
import fasttext

# Prepare corpus (one sentence per line)
with open('corpus.txt', 'w') as f:
    f.write("This is an example sentence.\n")
    f.write("Another sentence for training.\n")

# Train unsupervised model
model = fasttext.train_unsupervised('corpus.txt',
                                    model='skipgram',
                                    dim=100,
                                    epoch=5,
                                    lr=0.1)

# Save the model
model.save_model("fasttext_model.bin")
```

Slide 6: Từ tương đồng và tương tự

Việc nhúng FastText có thể được sử dụng để tìm các từ tương tự và giải quyết các từ tương tự. Điều này hữu ích cho các nhiệm vụ và ứng dụng NLP khác nhau, chẳng hạn như hệ thống khuyến nghị hoặc hiểu ngôn ngữ.

```python
import fasttext

# Load pre-trained model
model = fasttext.load_model("fasttext_model.bin")

# Find similar words
similar_words = model.get_nearest_neighbors("computer", k=5)
print("Words similar to 'computer':", similar_words)

# Word analogy
result = model.get_analogies("king", "man", "woman")
print("king - man + woman =", result)
```

Slide 7: Phân loại văn bản bằng FastText

FastText có thể được sử dụng cho các nhiệm vụ phân loại văn bản hiệu quả. Nó đặc biệt hữu ích cho các vấn đề quy mô lớn với nhiều loại. Mô hình có thể xử lý cả phân loại một nhãn và đa nhãn.

```python
import fasttext

# Prepare labeled data (format: __label__category text)
with open('train.txt', 'w') as f:
    f.write("__label__positive This movie is great!\n")
    f.write("__label__negative I didn't like the book.\n")

# Train classifier
classifier = fasttext.train_supervised('train.txt', lr=0.5, epoch=25)

# Predict
text = "I enjoyed watching this film."
predictions = classifier.predict(text)
print(f"Text: {text}")
print(f"Predicted label: {predictions[0][0]}, Probability: {predictions[1][0]:.2f}")
```

Slide 8: Xử lý từ ngoài từ vựng

Việc sử dụng thông tin từ phụ của FastText cho phép nó tạo ra các phần nhúng cho những từ không được nhìn thấy trong quá trình đào tạo. Đây là một lợi thế đáng kể so với các phương pháp nhúng từ truyền thống.

```python
import fasttext

# Load pre-trained model
model = fasttext.load_model("fasttext_model.bin")

# Get vector for an out-of-vocabulary word
oov_word = "untrainedword"
oov_vector = model.get_word_vector(oov_word)

print(f"Vector for '{oov_word}':")
print(oov_vector[:10])  # Print first 10 elements

# Find nearest neighbors for the OOV word
nearest_neighbors = model.get_nearest_neighbors(oov_word, k=5)
print(f"Nearest neighbors for '{oov_word}':", nearest_neighbors)
```

Trang trình bày 9: FastText so với Word2Vec

FastText xây dựng dựa trên Word2Vec bằng cách kết hợp thông tin từ phụ. Sự so sánh này nêu bật những điểm khác biệt và ưu điểm chính của FastText so với các kỹ thuật nhúng từ truyền thống.

```python
import fasttext
import gensim

# FastText
fasttext_model = fasttext.train_unsupervised('corpus.txt', model='skipgram')

# Word2Vec
sentences = [line.split() for line in open('corpus.txt', 'r')]
word2vec_model = gensim.models.Word2Vec(sentences, min_count=1)

# Compare embeddings
word = "example"
print("FastText embedding:", fasttext_model.get_word_vector(word)[:5])
print("Word2Vec embedding:", word2vec_model.wv[word][:5])

# Out-of-vocabulary word
oov_word = "unseeword"
print("FastText OOV:", fasttext_model.get_word_vector(oov_word)[:5])
# Word2Vec will raise KeyError for OOV words
```

Slide 10: Tiền xử lý FastText

Việc xử lý trước văn bản phù hợp là rất quan trọng để có hiệu suất FastText tối ưu. Điều này bao gồm mã thông báo, viết thường và xử lý các ký tự đặc biệt. Cách tiếp cận từ phụ của FastText giúp giảm nhu cầu xử lý trước rộng rãi so với các mô hình khác.

```python
import re
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Join tokens
    return ' '.join(tokens)

# Example usage
raw_text = "Hello, world! This is an example."
processed_text = preprocess_text(raw_text)
print("Raw text:", raw_text)
print("Processed text:", processed_text)
```

Slide 11: FastText cho ứng dụng đa ngôn ngữ

Khả năng xử lý thông tin từ phụ của FastText khiến nó đặc biệt hữu ích cho các ứng dụng đa ngôn ngữ. Nó có thể tạo ra các phần nhúng có ý nghĩa cho các ngôn ngữ có hình thái phức tạp hoặc dữ liệu huấn luyện hạn chế.

```python
import fasttext

# Train multilingual model
model = fasttext.train_unsupervised(
    'multilingual_corpus.txt',
    model='skipgram',
    dim=100,
    minn=2,
    maxn=5
)

# Get embeddings for words in different languages
languages = ['english', 'spanish', 'french', 'german']
word = 'hello'

for lang in languages:
    vector = model.get_word_vector(f"{word}_{lang}")
    print(f"Embedding for '{word}' in {lang}:", vector[:5])
```

Trang trình chiếu 12: Ví dụ thực tế: Phân tích cảm xúc

FastText có thể được sử dụng một cách hiệu quả cho các nhiệm vụ phân tích cảm xúc, chẳng hạn như phân tích đánh giá của khách hàng hoặc bài đăng trên mạng xã hội. Ví dụ này trình bày cách đào tạo và sử dụng mô hình FastText để phân loại cảm tính.

```python
import fasttext

# Prepare labeled data
with open('reviews.txt', 'w') as f:
    f.write("__label__positive The product exceeded my expectations.\n")
    f.write("__label__negative The service was disappointing.\n")
    f.write("__label__neutral It's an average product, nothing special.\n")

# Train the model
model = fasttext.train_supervised('reviews.txt', lr=0.5, epoch=25)

# Analyze new reviews
new_reviews = [
    "I love this product!",
    "The quality is terrible.",
    "It's okay, but could be better."
]

for review in new_reviews:
    label, prob = model.predict(review)
    print(f"Review: {review}")
    print(f"Sentiment: {label[0]}, Probability: {prob[0]:.2f}\n")
```

Slide 13: Ví dụ thực tế: Nhận dạng ngôn ngữ

FastText có thể được sử dụng để nhận dạng ngôn ngữ, rất hữu ích cho việc xử lý nội dung đa ngôn ngữ. Ví dụ này cho thấy cách đào tạo và sử dụng mô hình FastText để xác định ngôn ngữ trong các đoạn văn bản ngắn.

```python
import fasttext

# Prepare training data
with open('lang_data.txt', 'w') as f:
    f.write("__label__en This is an English sentence.\n")
    f.write("__label__es Esta es una oración en español.\n")
    f.write("__label__fr Ceci est une phrase en français.\n")

# Train the model
model = fasttext.train_supervised('lang_data.txt', lr=0.5, epoch=25)

# Identify languages
texts = [
    "Hello, how are you?",
    "Bonjour, comment allez-vous?",
    "Hola, ¿cómo estás?"
]

for text in texts:
    lang, prob = model.predict(text)
    print(f"Text: {text}")
    print(f"Detected language: {lang[0]}, Probability: {prob[0]:.2f}\n")
```

Trang trình bày 14: Tối ưu hóa FastText và điều chỉnh hiệu suất

Tối ưu hóa mô hình FastText liên quan đến việc điều chỉnh siêu tham số và cân nhắc sự cân bằng giữa kích thước mô hình, tốc độ đào tạo và hiệu suất. Các thông số chính bao gồm tốc độ học, thứ nguyên nhúng và kích thước n-gram.

```python
import fasttext
import time

def train_and_evaluate(params):
    start_time = time.time()
    model = fasttext.train_supervised('train_data.txt', **params)
    train_time = time.time() - start_time

    accuracy = model.test('test_data.txt')[1]
    model_size = model.get_input_matrix().size * 4 / (1024 * 1024)  # Size in MB

    return accuracy, train_time, model_size

# Different configurations
configs = [
    {'dim': 100, 'epoch': 5, 'lr': 0.1, 'wordNgrams': 2},
    {'dim': 200, 'epoch': 10, 'lr': 0.05, 'wordNgrams': 3},
    {'dim': 300, 'epoch': 15, 'lr': 0.01, 'wordNgrams': 4}
]

for i, config in enumerate(configs):
    accuracy, train_time, model_size = train_and_evaluate(config)
    print(f"Config {i+1}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Training Time: {train_time:.2f} seconds")
    print(f"  Model Size: {model_size:.2f} MB\n")
```

Trang trình bày 15: Tài nguyên bổ sung

Để biết thêm thông tin về FastText và các ứng dụng của nó, hãy cân nhắc khám phá các tài nguyên sau:

1. Tài liệu chính thức của FastText: [https://fasttext.cc/docs/en/support.html](https://fasttext.cc/docs/en/support.html)
2. "Làm phong phú vectơ từ bằng thông tin từ phụ" của P. Bojanowski và cộng sự. (2017): [https://arxiv.org/abs/1607.04606](https://arxiv.org/abs/1607.04606)
3. "Túi thủ thuật phân loại văn bản hiệu quả" của A. Joulin và cộng sự. (2016): [https://arxiv.org/abs/1607.01759](https://arxiv.org/abs/1607.01759)
4. "FastText.zip: Nén các mô hình phân loại văn bản" của A. Joulin và cộng sự. (2016): [https://arxiv.org/abs/1612.03651](https://arxiv.org/abs/1612.03651)

Các tài nguyên này cung cấp thông tin chuyên sâu về thuật toán FastText, cách triển khai nó và các ứng dụng khác nhau trong các tác vụ xử lý ngôn ngữ tự nhiên.
