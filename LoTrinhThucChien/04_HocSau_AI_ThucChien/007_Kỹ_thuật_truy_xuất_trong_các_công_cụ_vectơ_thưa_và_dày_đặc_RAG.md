## Kỹ thuật truy xuất trong RAG Công cụ vectơ thưa và dày đặc
Trang trình bày 1: Kỹ thuật truy xuất trong RAG: Công cụ vectơ thưa thớt và dày đặc

Thế hệ tăng cường truy xuất (RAG) kết hợp việc truy xuất thông tin với tạo văn bản. Bài trình bày này tập trung vào hai kỹ thuật truy xuất chính: các công cụ vectơ thưa thớt và dày đặc, được triển khai bằng Python.

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

# Example corpus
corpus = [
    "Sparse vectors in RAG",
    "Dense vectors for retrieval",
    "Combining sparse and dense techniques"
]

# Sparse vectorization (TF-IDF)
tfidf = TfidfVectorizer()
sparse_vectors = tfidf.fit_transform(corpus)

# Dense vectorization
model = SentenceTransformer('all-MiniLM-L6-v2')
dense_vectors = model.encode(corpus)

print("Sparse vectors shape:", sparse_vectors.shape)
print("Dense vectors shape:", dense_vectors.shape)
```

Trang trình bày 2: Biểu diễn vectơ thưa thớt: TF-IDF

TF-IDF (Tần số tài liệu nghịch đảo tần số thuật ngữ) là một biểu diễn vectơ thưa thớt phổ biến. Nó nắm bắt được tầm quan trọng của các từ trong tài liệu so với tập hợp tài liệu.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "TF-IDF captures word importance",
    "Sparse vectors are efficient for large vocabularies",
    "TF-IDF is widely used in information retrieval"
]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

print("TF-IDF Matrix shape:", tfidf_matrix.shape)
print("Vocabulary size:", len(vectorizer.vocabulary_))

# Get feature names (words)
feature_names = vectorizer.get_feature_names_out()
print("First 5 features:", feature_names[:5])
```

Trang trình bày 3: Tính toán TF-IDF

TF-IDF được tính bằng cách nhân tần số thuật ngữ (TF) với tần số tài liệu nghịch đảo (IDF). Hãy chia nhỏ phép tính cho một thuật ngữ.

```python
import numpy as np

def tf(term, doc):
    return doc.count(term) / len(doc.split())

def idf(term, docs):
    n_docs_with_term = sum(1 for doc in docs if term in doc)
    return np.log((len(docs) + 1) / (n_docs_with_term + 1)) + 1

def tfidf(term, doc, docs):
    return tf(term, doc) * idf(term, docs)

documents = [
    "This is a sample document",
    "Another example document",
    "Third document for demonstration"
]

term = "document"
doc = documents[0]

tf_value = tf(term, doc)
idf_value = idf(term, documents)
tfidf_value = tfidf(term, doc, documents)

print(f"TF: {tf_value:.4f}")
print(f"IDF: {idf_value:.4f}")
print(f"TF-IDF: {tfidf_value:.4f}")
```

Trang trình bày 4: Biểu diễn vectơ dày đặc: Nhúng từ

Các biểu diễn vectơ dày đặc, như nhúng từ, nắm bắt các mối quan hệ ngữ nghĩa giữa các từ trong không gian có chiều thấp. Các mô hình phổ biến bao gồm Word2Vec, GloVe và FastText.

```python
from gensim.models import Word2Vec

# Sample corpus
corpus = [
    ["dense", "vectors", "capture", "semantic", "relationships"],
    ["word", "embeddings", "are", "useful", "for", "many", "nlp", "tasks"],
    ["vector", "representations", "in", "low", "dimensional", "space"]
]

# Train Word2Vec model
model = Word2Vec(sentences=corpus, vector_size=100, window=5, min_count=1, workers=4)

# Get vector for a word
word_vector = model.wv['dense']

print("Vector shape:", word_vector.shape)
print("First 5 dimensions:", word_vector[:5])

# Find similar words
similar_words = model.wv.most_similar('vector', topn=3)
print("Words similar to 'vector':", similar_words)
```

Slide 5: Nhúng câu bằng Transformers

Để truy xuất tài liệu, chúng ta thường cần biểu diễn toàn bộ câu hoặc đoạn văn. Các mô hình dựa trên biến áp như BERT có thể tạo ra các vectơ dày đặc cho văn bản dài hơn.

```python
from sentence_transformers import SentenceTransformer

# Load pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Sample sentences
sentences = [
    "Sentence embeddings represent whole sentences.",
    "They capture context and meaning effectively.",
    "Transformer models excel at generating these embeddings."
]

# Generate embeddings
embeddings = model.encode(sentences)

print("Embeddings shape:", embeddings.shape)
print("First sentence embedding (first 5 dimensions):", embeddings[0][:5])

# Calculate similarity between sentences
from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
print(f"Similarity between first two sentences: {similarity:.4f}")
```

Trang trình bày 6: Truy xuất thưa thớt: Chỉ mục đảo ngược

Chỉ mục đảo ngược là cấu trúc dữ liệu được sử dụng để truy xuất hiệu quả trong không gian vectơ thưa thớt. Nó ánh xạ các thuật ngữ tới các tài liệu chứa chúng.

```python
from collections import defaultdict

def build_inverted_index(documents):
    inverted_index = defaultdict(list)
    for doc_id, doc in enumerate(documents):
        for term in doc.split():
            inverted_index[term].append(doc_id)
    return inverted_index

documents = [
    "sparse vectors in information retrieval",
    "efficient search using inverted index",
    "vector space model for document ranking"
]

index = build_inverted_index(documents)

# Query the index
query = "vectors in retrieval"
matching_docs = set()
for term in query.split():
    matching_docs.update(index.get(term, []))

print("Matching document IDs:", matching_docs)
```

Trang trình bày 7: Truy xuất dày đặc: Hàng xóm gần nhất gần đúng

Để truy xuất vectơ dày đặc, chúng tôi thường sử dụng các thuật toán lân cận gần nhất (ANN) gần đúng như đồ thị Thế giới nhỏ có thể điều hướng phân cấp (HNSW).

```python
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Generate random dense vectors
np.random.seed(42)
num_vectors = 1000
vector_dim = 128
vectors = np.random.rand(num_vectors, vector_dim)

# Build ANN index
ann_index = NearestNeighbors(n_neighbors=5, algorithm='ball_tree')
ann_index.fit(vectors)

# Query vector
query_vector = np.random.rand(1, vector_dim)

# Find nearest neighbors
distances, indices = ann_index.kneighbors(query_vector)

print("Nearest neighbor indices:", indices[0])
print("Distances:", distances[0])
```

Slide 8: Truy xuất lai: Kết hợp thưa thớt và dày đặc

Truy xuất kết hợp kết hợp các điểm mạnh của cả kỹ thuật thưa thớt và dày đặc. Chúng ta có thể sử dụng các phương pháp tập hợp hoặc học cách xếp hạng các phương pháp tiếp cận.

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

documents = [
    "Hybrid retrieval combines sparse and dense techniques",
    "Ensemble methods improve search results",
    "Learning to rank optimizes retrieval performance"
]

# Sparse retrieval (TF-IDF)
tfidf = TfidfVectorizer()
sparse_vectors = tfidf.fit_transform(documents)

# Dense retrieval (Sentence Transformers)
model = SentenceTransformer('all-MiniLM-L6-v2')
dense_vectors = model.encode(documents)

# Hybrid scoring function
def hybrid_score(query, doc_id, alpha=0.5):
    sparse_score = sparse_vectors[doc_id].dot(tfidf.transform([query]).T).toarray()[0][0]
    dense_score = np.dot(dense_vectors[doc_id], model.encode([query])[0])
    return alpha * sparse_score + (1 - alpha) * dense_score

query = "combining retrieval techniques"
scores = [hybrid_score(query, i) for i in range(len(documents))]

print("Hybrid retrieval scores:", scores)
print("Best matching document:", documents[np.argmax(scores)])
```

Slide 9: Ví dụ thực tế: Công cụ tìm kiếm tài liệu

Triển khai một công cụ tìm kiếm tài liệu đơn giản bằng cách sử dụng cả kỹ thuật truy xuất thưa thớt và dày đặc.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import numpy as np

class DocumentSearchEngine:
    def __init__(self, documents):
        self.documents = documents
        self.tfidf = TfidfVectorizer()
        self.sparse_vectors = self.tfidf.fit_transform(documents)
        self.dense_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dense_vectors = self.dense_model.encode(documents)

    def search(self, query, k=3, alpha=0.5):
        sparse_query = self.tfidf.transform([query])
        dense_query = self.dense_model.encode([query])

        sparse_scores = self.sparse_vectors.dot(sparse_query.T).toarray().flatten()
        dense_scores = np.dot(self.dense_vectors, dense_query.T).flatten()

        hybrid_scores = alpha * sparse_scores + (1 - alpha) * dense_scores
        top_k = np.argsort(hybrid_scores)[::-1][:k]

        return [(self.documents[i], hybrid_scores[i]) for i in top_k]

# Example usage
documents = [
    "Python is a popular programming language",
    "Machine learning models require large datasets",
    "Natural language processing analyzes text data",
    "Deep learning architectures include neural networks",
    "Data scientists use various statistical techniques"
]

search_engine = DocumentSearchEngine(documents)
results = search_engine.search("programming languages for data science")

for doc, score in results:
    print(f"Score: {score:.4f} - {doc}")
```

Trang trình chiếu 10: Ví dụ thực tế: Hệ thống khuyến nghị

Sử dụng các biểu diễn vectơ dày đặc để xây dựng hệ thống đề xuất dựa trên nội dung đơn giản cho các bài viết.

```python
from sentence_transformers import SentenceTransformer
import numpy as np

class ArticleRecommender:
    def __init__(self, articles):
        self.articles = articles
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = self.model.encode(articles)

    def recommend(self, user_history, n=3):
        user_embedding = self.model.encode(user_history)
        similarities = np.dot(self.embeddings, user_embedding.T).flatten()
        top_n = np.argsort(similarities)[::-1][:n]
        return [(self.articles[i], similarities[i]) for i in top_n]

# Example usage
articles = [
    "The impact of artificial intelligence on modern society",
    "Exploring the wonders of the deep ocean",
    "Advancements in renewable energy technologies",
    "The role of genetics in personalized medicine",
    "Space exploration: Past achievements and future goals"
]

recommender = ArticleRecommender(articles)
user_history = "I'm interested in technology and its effects on our world"

recommendations = recommender.recommend(user_history)

print("Recommended articles:")
for article, similarity in recommendations:
    print(f"Similarity: {similarity:.4f} - {article}")
```

Slide 11: Những thách thức trong việc truy xuất RAG

Việc truy xuất trong hệ thống RAG phải đối mặt với một số thách thức, bao gồm xử lý các truy vấn ngoài phân phối, xử lý các bộ dữ liệu quy mô lớn và duy trì thông tin cập nhật.

```python
import numpy as np
from sklearn.preprocessing import normalize

class RAGRetriever:
    def __init__(self, documents, embeddings):
        self.documents = documents
        self.embeddings = normalize(embeddings)  # Normalize for cosine similarity

    def retrieve(self, query_embedding, k=3, threshold=0.5):
        query_embedding = normalize(query_embedding.reshape(1, -1))
        similarities = np.dot(self.embeddings, query_embedding.T).flatten()
        top_k = np.argsort(similarities)[::-1][:k]

        results = []
        for idx in top_k:
            if similarities[idx] >= threshold:
                results.append((self.documents[idx], similarities[idx]))
            else:
                break  # Stop if similarity is below threshold

        return results if results else [("No relevant documents found", 0)]

# Example usage
documents = [
    "Artificial intelligence and machine learning",
    "Climate change and global warming",
    "Quantum computing and cryptography",
    "Renewable energy sources and sustainability"
]
embeddings = np.random.rand(len(documents), 128)  # Simulated embeddings

retriever = RAGRetriever(documents, embeddings)
query_embedding = np.random.rand(128)  # Simulated query embedding

results = retriever.retrieve(query_embedding, threshold=0.7)

for doc, score in results:
    print(f"Score: {score:.4f} - {doc}")
```

Slide 12: Đánh giá hiệu suất truy xuất

Đánh giá chất lượng truy xuất là rất quan trọng đối với hệ thống RAG. Các số liệu phổ biến bao gồm độ chính xác, thu hồi và độ chính xác trung bình trung bình (MAP).

```python
import numpy as np
from sklearn.metrics import precision_score, recall_score

def mean_average_precision(relevant_docs, retrieved_docs, k=10):
    if not relevant_docs:
        return 0.0

    score = 0.0
    num_hits = 0

    for i, doc in enumerate(retrieved_docs[:k]):
        if doc in relevant_docs:
            num_hits += 1
            score += num_hits / (i + 1)

    return score / len(relevant_docs)

# Example evaluation
relevant_docs = set([1, 3, 5, 7])
retrieved_docs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

precision = precision_score(
    [1 if doc in relevant_docs else 0 for doc in retrieved_docs],
    [1 if doc in relevant_docs else 0 for doc in range(1, 11)]
)

recall = recall_score(
    [1 if doc in relevant_docs else 0 for doc in retrieved_docs],
    [1 if doc in relevant_docs else 0 for doc in range(1, 11)]
)

map_score = mean_average_precision(relevant_docs, retrieved_docs)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Mean Average Precision: {map_score:.4f}")
```

Trang trình bày 13: Định hướng tương lai trong truy xuất RAG

Việc truy xuất RAG tiếp tục phát triển với những tiến bộ trong việc truy xuất thông tin thần kinh, học ít lần và kỹ thuật truy xuất đa phương thức. Những đổi mới này nhằm mục đích cải thiện độ chính xác và hiệu quả truy xuất trong các tình huống khác nhau.

```python
import numpy as np
from scipy.special import softmax

class AdvancedRetriever:
    def __init__(self, documents, text_embeddings, image_embeddings):
        self.documents = documents
        self.text_embeddings = text_embeddings
        self.image_embeddings = image_embeddings

    def multi_modal_retrieve(self, text_query, image_query, k=3):
        text_sim = np.dot(self.text_embeddings, text_query)
        image_sim = np.dot(self.image_embeddings, image_query)

        combined_sim = 0.7 * text_sim + 0.3 * image_sim
        top_k = np.argsort(combined_sim)[::-1][:k]

        return [(self.documents[i], combined_sim[i]) for i in top_k]

# Simulated data
documents = ["Doc1", "Doc2", "Doc3", "Doc4"]
text_embeddings = np.random.rand(4, 128)
image_embeddings = np.random.rand(4, 256)

retriever = AdvancedRetriever(documents, text_embeddings, image_embeddings)
text_query = np.random.rand(128)
image_query = np.random.rand(256)

results = retriever.multi_modal_retrieve(text_query, image_query)
for doc, score in results:
    print(f"Score: {score:.4f} - {doc}")
```

Trang trình bày 14: Những cân nhắc về mặt đạo đức trong việc truy xuất RAG

Khi các hệ thống RAG trở nên phổ biến hơn, điều quan trọng là phải giải quyết các mối lo ngại về đạo đức như sai lệch khi truy xuất, bảo vệ quyền riêng tư và khả năng khuếch đại thông tin sai lệch.

```python
import numpy as np
from sklearn.preprocessing import StandardScaler

class EthicalRetriever:
    def __init__(self, documents, embeddings, sensitive_attributes):
        self.documents = documents
        self.embeddings = embeddings
        self.sensitive_attributes = sensitive_attributes
        self.scaler = StandardScaler()
        self.normalized_embeddings = self.scaler.fit_transform(embeddings)

    def fair_retrieve(self, query_embedding, k=3):
        query_embedding = self.scaler.transform(query_embedding.reshape(1, -1))
        similarities = np.dot(self.normalized_embeddings, query_embedding.T).flatten()

        # Apply fairness constraint
        fairness_scores = np.mean(self.sensitive_attributes, axis=1)
        adjusted_similarities = similarities * (1 - fairness_scores)

        top_k = np.argsort(adjusted_similarities)[::-1][:k]
        return [(self.documents[i], adjusted_similarities[i]) for i in top_k]

# Simulated data
documents = ["Doc1", "Doc2", "Doc3", "Doc4"]
embeddings = np.random.rand(4, 128)
sensitive_attributes = np.random.randint(0, 2, size=(4, 3))  # Binary attributes

retriever = EthicalRetriever(documents, embeddings, sensitive_attributes)
query_embedding = np.random.rand(1, 128)

results = retriever.fair_retrieve(query_embedding)
for doc, score in results:
    print(f"Score: {score:.4f} - {doc}")
```

Trang trình bày 15: Tài nguyên bổ sung

Để khám phá thêm về kỹ thuật truy xuất vectơ và RAG, hãy xem xét các tài nguyên sau:

1. "Truy xuất đoạn đường dày đặc để trả lời câu hỏi trên miền mở" (Karpukhin và cộng sự, 2020) ArXiv: [https://arxiv.org/abs/2004.04906](https://arxiv.org/abs/2004.04906)
2. "Thế hệ tăng cường truy xuất cho các nhiệm vụ NLP chuyên sâu về tri thức" (Lewis và cộng sự, 2020) ArXiv: [https://arxiv.org/abs/2005.11401](https://arxiv.org/abs/2005.11401)
3. "REALM: Đào tạo trước về mô hình ngôn ngữ tăng cường truy xuất" (Guu và cộng sự, 2020) ArXiv: [https://arxiv.org/abs/2002.08909](https://arxiv.org/abs/2002.08909)
4. "Cải thiện khả năng hiểu ngôn ngữ bằng cách đào tạo trước mang tính sáng tạo" (Radford và cộng sự, 2018) Có sẵn tại: [https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/lingu-unsupervised/lingu\_knowing\_paper.pdf](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/lingu-unsupervised/lingu_under Hiểu_paper.pdf)

Các bài viết này cung cấp các cuộc thảo luận chuyên sâu về các khía cạnh khác nhau của kỹ thuật truy xuất trong bối cảnh mô hình ngôn ngữ và hệ thống trả lời câu hỏi.
