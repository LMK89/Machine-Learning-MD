## Khám phá nhúng từ trong NLP bằng Python
Slide 1: Giới thiệu về Word Embeddings

Việc nhúng từ là cách biểu diễn vectơ dày đặc của các từ nhằm nắm bắt các ý nghĩa và mối quan hệ ngữ nghĩa. Chúng là nền tảng cho nhiều nhiệm vụ NLP.

```python
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Example word embedding
word_embedding = {
    "king": np.array([0.1, 0.2, 0.3]),
    "queen": np.array([0.15, 0.25, 0.35]),
    "man": np.array([0.05, 0.1, 0.15]),
    "woman": np.array([0.08, 0.13, 0.18])
}

# Visualize embeddings using t-SNE
embeddings = np.array(list(word_embedding.values()))
tsne = TSNE(n_components=2, random_state=42)
embedded = tsne.fit_transform(embeddings)

plt.figure(figsize=(10, 8))
for i, word in enumerate(word_embedding.keys()):
    plt.scatter(embedded[i, 0], embedded[i, 1])
    plt.annotate(word, (embedded[i, 0], embedded[i, 1]))
plt.title("2D Visualization of Word Embeddings")
plt.show()
```

Slide 2: Kỹ thuật nhúng từ

Có nhiều kỹ thuật khác nhau để tạo phần nhúng từ, bao gồm Word2Vec, GloVe và FastText. Chúng tôi sẽ tập trung vào Word2Vec trong bài trình bày này.

```python
from gensim.models import Word2Vec

# Sample corpus
corpus = [
    ["I", "love", "natural", "language", "processing"],
    ["Word", "embeddings", "are", "powerful", "for", "NLP", "tasks"],
    ["Python", "is", "great", "for", "implementing", "NLP", "models"]
]

# Train Word2Vec model
model = Word2Vec(sentences=corpus, vector_size=100, window=5, min_count=1, workers=4)

# Get vector for a word
vector = model.wv["language"]
print(f"Vector for 'language': {vector[:5]}...")  # Showing first 5 dimensions
```

Trang trình bày 3: Word2Vec: Mô hình Skip-gram

Mô hình Skip-gram dự đoán các từ ngữ cảnh cho một từ mục tiêu. Nó có hiệu quả cho việc học cách biểu diễn từ.

```python
import torch
import torch.nn as nn

class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        output = self.linear(embeds)
        return output

# Example usage
vocab_size = 5000
embedding_dim = 300
model = SkipGramModel(vocab_size, embedding_dim)

# Dummy input
input_word = torch.tensor([42])
output = model(input_word)
print(f"Output shape: {output.shape}")
```

Slide 4: Word2Vec: Mô hình túi từ liên tục (CBOW)

CBOW dự đoán một từ mục tiêu dựa trên các từ ngữ cảnh của nó. Nó được đào tạo nhanh hơn so với Skip-gram nhưng có thể kém chính xác hơn đối với những từ không thường xuyên.

```python
class CBOWModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOWModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        hidden = torch.mean(embeds, dim=1)
        output = self.linear(hidden)
        return output

# Example usage
vocab_size = 5000
embedding_dim = 300
model = CBOWModel(vocab_size, embedding_dim)

# Dummy input (context words)
context_words = torch.tensor([[10, 20, 30, 40]])
output = model(context_words)
print(f"Output shape: {output.shape}")
```

Slide 5: Luyện tập cách nhúng từ

Việc đào tạo nhúng từ liên quan đến việc tối ưu hóa mô hình để dự đoán các từ dựa trên ngữ cảnh của chúng hoặc ngược lại.

```python
import torch.optim as optim
import torch.nn.functional as F

# Assuming we have a SkipGramModel instance called 'model'
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop (simplified)
for epoch in range(num_epochs):
    for batch in data_loader:
        target_word, context_words = batch

        optimizer.zero_grad()
        output = model(target_word)
        loss = F.cross_entropy(output, context_words)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
```

Trang trình bày 6: Sử dụng tính năng nhúng từ được đào tạo trước

Các phần nhúng được đào tạo trước như GloVe có thể được sử dụng để khởi động các tác vụ NLP mà không cần đào tạo lại từ đầu.

```python
import numpy as np
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

# Convert GloVe format to Word2Vec format
glove_input_file = 'glove.6B.100d.txt'
word2vec_output_file = 'glove.6B.100d.word2vec.txt'
glove2word2vec(glove_input_file, word2vec_output_file)

# Load the pre-trained embeddings
embeddings = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)

# Get vector for a word
vector = embeddings['python']
print(f"Vector for 'python': {vector[:5]}...")  # Showing first 5 dimensions

# Find similar words
similar_words = embeddings.most_similar('python', topn=5)
print("Words similar to 'python':", similar_words)
```

Slide 7: Từ tương đồng và tương đồng

Việc nhúng từ nắm bắt các mối quan hệ ngữ nghĩa, cho phép thực hiện các tác vụ tương tự và tương tự từ.

```python
from gensim.models import KeyedVectors

# Load pre-trained Word2Vec embeddings
embeddings = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

# Word similarity
similarity = embeddings.similarity('cat', 'dog')
print(f"Similarity between 'cat' and 'dog': {similarity:.4f}")

# Word analogy
result = embeddings.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
print(f"king - man + woman = {result[0][0]}")

# Visualize word relationships
words = ['king', 'queen', 'man', 'woman', 'prince', 'princess']
vectors = [embeddings[word] for word in words]

tsne = TSNE(n_components=2, random_state=42)
vectors_2d = tsne.fit_transform(vectors)

plt.figure(figsize=(10, 8))
for i, word in enumerate(words):
    plt.scatter(vectors_2d[i, 0], vectors_2d[i, 1])
    plt.annotate(word, (vectors_2d[i, 0], vectors_2d[i, 1]))
plt.title("Word Relationships Visualization")
plt.show()
```

Slide 8: Xử lý từ ngoài từ vựng

Xử lý các từ không có trong từ vựng là điều quan trọng đối với các hệ thống NLP mạnh mẽ.

```python
import numpy as np

class SimpleEmbedding:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.embedding_dim = len(next(iter(embeddings.values())))
        self.unk_vector = np.zeros(self.embedding_dim)

    def get_vector(self, word):
        return self.embeddings.get(word.lower(), self.unk_vector)

    def handle_oov(self, word):
        if word.lower() not in self.embeddings:
            # Simple method: Use character n-grams
            n = 3
            char_ngrams = [word[i:i+n] for i in range(len(word)-n+1)]
            ngram_vectors = [self.get_vector(ng) for ng in char_ngrams]
            return np.mean(ngram_vectors, axis=0)
        return self.get_vector(word)

# Example usage
embeddings = {"cat": np.array([0.1, 0.2, 0.3]), "dog": np.array([0.2, 0.3, 0.4])}
simple_embed = SimpleEmbedding(embeddings)

print(simple_embed.handle_oov("cat"))  # Known word
print(simple_embed.handle_oov("catdog"))  # OOV word
```

Slide 9: Nhúng từ để phân loại văn bản

Việc nhúng từ có thể cải thiện đáng kể các nhiệm vụ phân loại văn bản.

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Assuming we have a SimpleEmbedding instance called 'embeddings'

def text_to_vector(text, embeddings):
    words = text.split()
    word_vectors = [embeddings.handle_oov(word) for word in words]
    return np.mean(word_vectors, axis=0)

# Example dataset
texts = [
    "I love this movie",
    "This film is terrible",
    "Great acting and plot",
    "Worst movie ever"
]
labels = [1, 0, 1, 0]  # 1 for positive, 0 for negative

# Convert texts to vectors
X = np.array([text_to_vector(text, embeddings) for text in texts])
y = np.array(labels)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple classifier
clf = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
```

Slide 10: Tinh chỉnh cách nhúng từ

Tinh chỉnh các phần nhúng được đào tạo trước có thể điều chỉnh chúng cho phù hợp với các nhiệm vụ hoặc miền cụ thể.

```python
import torch
import torch.nn as nn
import torch.optim as optim

class EmbeddingClassifier(nn.Module):
    def __init__(self, pretrained_embeddings, num_classes):
        super(EmbeddingClassifier, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
        self.fc = nn.Linear(pretrained_embeddings.shape[1], num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        pooled = torch.mean(embedded, dim=1)
        return self.fc(pooled)

# Assuming we have pretrained embeddings and a vocabulary
pretrained_embeddings = torch.FloatTensor(...)  # Load your pretrained embeddings
vocab = {...}  # Your vocabulary mapping words to indices

# Create model and optimizer
model = EmbeddingClassifier(pretrained_embeddings, num_classes=2)
optimizer = optim.Adam(model.parameters())

# Training loop (simplified)
for epoch in range(num_epochs):
    for batch_texts, batch_labels in data_loader:
        optimizer.zero_grad()

        # Convert texts to indices
        indices = torch.LongTensor([[vocab.get(word, vocab['<UNK>']) for word in text.split()] for text in batch_texts])

        outputs = model(indices)
        loss = nn.CrossEntropyLoss()(outputs, batch_labels)

        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
```

Slide 11: Đánh giá việc nhúng từ

Đánh giá các phần nhúng từ là rất quan trọng để đảm bảo chất lượng và sự phù hợp của chúng cho các nhiệm vụ tiếp theo.

```python
import numpy as np
from scipy.stats import spearmanr

def evaluate_similarity(embeddings, similarity_dataset):
    human_scores = []
    model_scores = []

    for word1, word2, human_score in similarity_dataset:
        if word1 in embeddings and word2 in embeddings:
            vec1 = embeddings[word1]
            vec2 = embeddings[word2]
            model_score = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

            human_scores.append(float(human_score))
            model_scores.append(model_score)

    correlation, _ = spearmanr(human_scores, model_scores)
    return correlation

# Example usage
similarity_dataset = [
    ("cat", "dog", 0.8),
    ("happy", "sad", 0.1),
    ("king", "queen", 0.9),
    # ... more word pairs and human-annotated similarity scores
]

correlation = evaluate_similarity(embeddings, similarity_dataset)
print(f"Spearman correlation: {correlation:.4f}")
```

Trang trình bày 12: Nhúng từ trong mạng thần kinh

Việc kết hợp các từ nhúng vào mạng thần kinh có thể nâng cao hiệu suất của chúng trên các nhiệm vụ NLP khác nhau.

```python
import torch
import torch.nn as nn

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, filter_sizes, num_classes):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, embedding_dim)) for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        return self.fc(x)

# Example usage
vocab_size = 10000
embedding_dim = 300
num_filters = 100
filter_sizes = [3, 4, 5]
num_classes = 2

model = TextCNN(vocab_size, embedding_dim, num_filters, filter_sizes, num_classes)

# Dummy input
input_text = torch.LongTensor([[1, 2, 3, 4, 5]])
output = model(input_text)
print(f"Output shape: {output.shape}")
```

Trang trình bày 13: Nhúng từ để nhận dạng thực thể được đặt tên (NER)

Việc nhúng từ nâng cao khả năng nhận dạng thực thể được đặt tên bằng cách nắm bắt thông tin ngữ nghĩa.

```python
import torch
import torch.nn as nn

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_tags):
        super(BiLSTM_CRF, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim, num_tags)
        self.crf = nn.Linear(num_tags, num_tags)  # Simplified CRF layer

    def forward(self, x):
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)
        tag_space = self.hidden2tag(lstm_out)
        tag_scores = self.crf(tag_space)
        return tag_scores

# Example usage
vocab_size, embedding_dim, hidden_dim, num_tags = 10000, 100, 128, 9
model = BiLSTM_CRF(vocab_size, embedding_dim, hidden_dim, num_tags)
dummy_input = torch.LongTensor([[1, 2, 3, 4, 5]])
output = model(dummy_input)
print(f"Output shape: {output.shape}")
```

Trang trình bày 14: Nhúng từ theo ngữ cảnh

Những tiến bộ gần đây đã dẫn đến việc nhúng từ theo ngữ cảnh, giúp nắm bắt nghĩa của từ dựa trên ngữ cảnh.

```python
from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Example sentences
sentences = ["The bank is by the river.", "I need to bank my check."]

# Process sentences
for sentence in sentences:
    inputs = tokenizer(sentence, return_tensors="pt")
    outputs = model(**inputs)

    # Get embeddings for the word "bank"
    bank_index = torch.where(inputs['input_ids'][0] == tokenizer.convert_tokens_to_ids('bank'))[0]
    bank_embedding = outputs.last_hidden_state[0, bank_index, :]

    print(f"Embedding for 'bank' in '{sentence}':")
    print(bank_embedding.shape)
    print(bank_embedding[0, :10])  # Print first 10 dimensions
    print()
```

Trang trình bày 15: Nhúng từ trong NLP đa ngôn ngữ

Việc nhúng từ có thể được sử dụng cho các tác vụ NLP đa ngôn ngữ, cho phép hiểu nhiều ngôn ngữ.

```python
from gensim.models import KeyedVectors

# Load pre-trained multilingual embeddings (example using MUSE)
en_embeddings = KeyedVectors.load_word2vec_format('wiki.multi.en.vec')
fr_embeddings = KeyedVectors.load_word2vec_format('wiki.multi.fr.vec')

def translate(word, source_embed, target_embed):
    if word not in source_embed.key_to_index:
        return "Word not found in source embeddings"

    source_vector = source_embed[word]
    target_word, similarity = target_embed.similar_by_vector(source_vector, topn=1)[0]
    return f"'{word}' translates to '{target_word}' (similarity: {similarity:.2f})"

# Example usage
print(translate('dog', en_embeddings, fr_embeddings))
print(translate('chat', fr_embeddings, en_embeddings))
```

Trang trình bày 16: Tài nguyên bổ sung

Để khám phá thêm về cách nhúng từ trong NLP:

1. "Ước tính hiệu quả các biểu diễn từ trong không gian vectơ" của Mikolov và cộng sự. (2013) ArXiv: [https://arxiv.org/abs/1301.3781](https://arxiv.org/abs/1301.3781)
2. "GloVe: Các vectơ toàn cầu để thể hiện từ" của Pennington và cộng sự. (2014) ArXiv: [https://arxiv.org/abs/1405.3531](https://arxiv.org/abs/1405.3531)
3. "BERT: Đào tạo trước về Máy biến áp hai chiều sâu để hiểu ngôn ngữ" của Devlin và cộng sự. (2018) ArXiv: [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
