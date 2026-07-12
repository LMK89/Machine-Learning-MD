## Tích hợp phân đoạn muộn vào quy trình làm việc RAG dựa trên Python
Trang trình bày 1: Giới thiệu về Phân đoạn muộn trong quy trình làm việc RAG

Chunking muộn là một kỹ thuật được sử dụng để cải thiện hiệu quả và độ chính xác của hệ thống Thế hệ tăng cường truy xuất (RAG). Nó liên quan đến việc chia tài liệu thành các phần nhỏ hơn tại thời điểm truy vấn, thay vì trong giai đoạn lập chỉ mục ban đầu. Cách tiếp cận này cho phép truy xuất văn bản linh hoạt và nhận biết ngữ cảnh hơn, dẫn đến hiệu suất tốt hơn trong các tác vụ xử lý ngôn ngữ tự nhiên khác nhau.

```python
import nltk
from nltk.tokenize import sent_tokenize

def late_chunking(document, max_chunk_size=200):
    sentences = sent_tokenize(document)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chunk_size:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

# Example usage
document = "This is a sample document. It contains multiple sentences. We will split it into chunks using late chunking."
chunks = late_chunking(document)
print(chunks)
```

Slide 2: Lợi ích của việc Chunking muộn

Phân đoạn muộn mang lại một số lợi thế so với các phương pháp phân đoạn trước truyền thống. Nó cho phép điều chỉnh động các kích thước khối dựa trên truy vấn hoặc nhiệm vụ cụ thể hiện có. Tính linh hoạt này cho phép hệ thống nắm bắt được bối cảnh phù hợp hơn và duy trì sự mạch lạc về mặt ngữ nghĩa trong từng đoạn. Ngoài ra, Phân đoạn muộn có thể giúp giảm yêu cầu lưu trữ và cải thiện tốc độ xử lý bằng cách loại bỏ nhu cầu lưu trữ nhiều phiên bản được phân đoạn trước của cùng một tài liệu.

```python
import time

def measure_chunking_performance(document, chunk_function, iterations=1000):
    start_time = time.time()
    for _ in range(iterations):
        chunks = chunk_function(document)
    end_time = time.time()
    return (end_time - start_time) / iterations

# Compare performance of late chunking vs pre-chunking
document = "This is a sample document. " * 1000  # Large document

def pre_chunking(document, chunk_size=200):
    return [document[i:i+chunk_size] for i in range(0, len(document), chunk_size)]

late_chunking_time = measure_chunking_performance(document, late_chunking)
pre_chunking_time = measure_chunking_performance(document, pre_chunking)

print(f"Late Chunking average time: {late_chunking_time:.6f} seconds")
print(f"Pre-Chunking average time: {pre_chunking_time:.6f} seconds")
```

Trang trình bày 3: Triển khai phân đoạn muộn trong đường ống RAG

Để tích hợp Late Chunking vào quy trình làm việc RAG, chúng ta cần sửa đổi bước truy xuất. Thay vì truy xuất văn bản được phân đoạn trước, chúng tôi sẽ truy xuất toàn bộ tài liệu và áp dụng Phân đoạn muộn trước khi chuyển các đoạn sang mô hình ngôn ngữ. Cách tiếp cận này đảm bảo rằng các khối được tạo dựa trên ngữ cảnh truy vấn hiện tại.

```python
from transformers import AutoTokenizer, AutoModel
import torch

class RAGWithLateChunking:
    def __init__(self, model_name="bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def retrieve_and_chunk(self, query, documents, max_chunk_size=200):
        # Retrieve relevant documents (simplified for this example)
        relevant_docs = self.simple_retrieval(query, documents)

        # Apply late chunking to retrieved documents
        chunks = []
        for doc in relevant_docs:
            chunks.extend(late_chunking(doc, max_chunk_size))

        return chunks

    def simple_retrieval(self, query, documents):
        # Simplified retrieval based on keyword matching
        return [doc for doc in documents if query.lower() in doc.lower()]

    def generate_response(self, query, documents):
        chunks = self.retrieve_and_chunk(query, documents)
        # Process chunks with the language model (simplified)
        inputs = self.tokenizer(chunks, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model(**inputs)
        # Further processing would be required here to generate the final response
        return outputs

# Example usage
rag_system = RAGWithLateChunking()
query = "What is the capital of France?"
documents = ["Paris is the capital of France.", "France is a country in Europe.", "The Eiffel Tower is in Paris."]
response = rag_system.generate_response(query, documents)
print("Response generated successfully.")
```

Trang trình bày 4: Phân đoạn phần mềm nhận biết truy vấn

Chunking muộn cho phép phân tách tài liệu nhận biết truy vấn. Bằng cách xem xét truy vấn khi tạo các khối, chúng tôi có thể đảm bảo rằng thông tin liên quan được lưu giữ cùng nhau, cải thiện chất lượng ngữ cảnh được truy xuất cho mô hình ngôn ngữ.

```python
import re

def query_aware_chunking(document, query, max_chunk_size=200):
    # Find all occurrences of query terms in the document
    query_terms = re.findall(r'\w+', query.lower())
    term_positions = {}
    for term in query_terms:
        term_positions[term] = [m.start() for m in re.finditer(r'\b' + re.escape(term.lower()) + r'\b', document.lower())]

    # Create chunks around query term occurrences
    chunks = []
    last_end = 0
    for term, positions in term_positions.items():
        for pos in positions:
            start = max(0, pos - max_chunk_size // 2)
            end = min(len(document), pos + max_chunk_size // 2)
            if start < last_end:
                chunks[-1] = document[last_end:end]
            else:
                chunks.append(document[start:end])
            last_end = end

    # Add any remaining text as a final chunk
    if last_end < len(document):
        chunks.append(document[last_end:])

    return chunks

# Example usage
document = "Paris is the capital of France. It is known for its beautiful architecture and rich history. The Eiffel Tower, located in Paris, is a famous landmark."
query = "What is the capital of France?"
chunks = query_aware_chunking(document, query)
print("Query-aware chunks:", chunks)
```

Slide 5: Xử lý sự chồng chéo khi Chunking muộn

Để đảm bảo tính liên tục của ngữ cảnh giữa các khối, chúng tôi có thể triển khai chiến lược chồng chéo trong thuật toán Phân đoạn muộn. Cách tiếp cận này giúp duy trì sự mạch lạc và giảm nguy cơ chia tách thông tin quan trọng qua các ranh giới khối.

```python
def overlapping_late_chunking(document, max_chunk_size=200, overlap=50):
    words = document.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + max_chunk_size
        if end > len(words):
            end = len(words)

        chunk = ' '.join(words[start:end])
        chunks.append(chunk)

        start += max_chunk_size - overlap

    return chunks

# Example usage
document = "This is a sample document. " * 20  # Create a longer document
chunks = overlapping_late_chunking(document)
print(f"Number of chunks: {len(chunks)}")
print("First chunk:", chunks[0])
print("Second chunk:", chunks[1])
```

Trang trình bày 6: Kích thước khối thích ứng

Định cỡ đoạn thích ứng là một kỹ thuật điều chỉnh kích thước đoạn dựa trên nội dung và cấu trúc của tài liệu. Cách tiếp cận này có thể cải thiện chất lượng của các đoạn bằng cách tôn trọng các ranh giới tự nhiên trong văn bản, chẳng hạn như ngắt đoạn hoặc thay đổi chủ đề.

```python
import re

def adaptive_chunk_sizing(document, min_chunk_size=100, max_chunk_size=300):
    paragraphs = re.split(r'\n\s*\n', document)
    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) <= max_chunk_size:
            current_chunk += paragraph + "\n\n"
        else:
            if len(current_chunk) >= min_chunk_size:
                chunks.append(current_chunk.strip())
                current_chunk = paragraph + "\n\n"
            else:
                # If the current chunk is too small, we'll exceed max_chunk_size
                chunks.append((current_chunk + paragraph).strip())
                current_chunk = ""

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

# Example usage
document = """
Paragraph 1: This is the first paragraph. It contains multiple sentences.

Paragraph 2: This is a short paragraph.

Paragraph 3: This paragraph is longer and contains more information. It might exceed the minimum chunk size on its own.

Paragraph 4: Another short paragraph.

Paragraph 5: The final paragraph of our example document.
"""

chunks = adaptive_chunk_sizing(document)
for i, chunk in enumerate(chunks, 1):
    print(f"Chunk {i} (length: {len(chunk)}):")
    print(chunk)
    print()
```

Trang trình bày 7: Phân đoạn muộn về nhận thức ngữ nghĩa

Phân đoạn muộn nhận biết ngữ nghĩa sử dụng các kỹ thuật xử lý ngôn ngữ tự nhiên để tạo ra các khối dựa trên sự mạch lạc về ngữ nghĩa. Phương pháp này nhằm mục đích gắn kết các khái niệm liên quan lại với nhau, nâng cao chất lượng ngữ cảnh được cung cấp cho mô hình ngôn ngữ.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def semantic_aware_chunking(document, max_chunk_size=200, similarity_threshold=0.3):
    sentences = sent_tokenize(document)
    vectorizer = TfidfVectorizer()
    sentence_vectors = vectorizer.fit_transform(sentences)

    chunks = []
    current_chunk = []
    current_length = 0

    for i, sentence in enumerate(sentences):
        if current_length + len(sentence) <= max_chunk_size:
            current_chunk.append(sentence)
            current_length += len(sentence)
        else:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = len(sentence)

        # Check semantic similarity with the next sentence
        if i < len(sentences) - 1:
            similarity = cosine_similarity(sentence_vectors[i], sentence_vectors[i+1])[0][0]
            if similarity < similarity_threshold and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# Example usage
document = """
Natural language processing (NLP) is a field of artificial intelligence. It focuses on the interaction between computers and human language. NLP techniques are used in many applications.

Machine learning is a subset of AI. It involves algorithms that improve through experience. Deep learning is a type of machine learning based on artificial neural networks.

Python is a popular programming language. It is widely used in data science and AI development. Many NLP and machine learning libraries are available in Python.
"""

chunks = semantic_aware_chunking(document)
for i, chunk in enumerate(chunks, 1):
    print(f"Chunk {i}:")
    print(chunk)
    print()
```

Slide 8: Xử lý tài liệu đa ngôn ngữ

Khi xử lý các tài liệu đa ngôn ngữ, Phân đoạn muộn cần được điều chỉnh để tôn trọng ranh giới ngôn ngữ và đảm bảo mã thông báo phù hợp cho từng ngôn ngữ. Dưới đây là ví dụ về cách triển khai Phân đoạn muộn nhận biết ngôn ngữ:

```python
from langdetect import detect
import nltk
from nltk.tokenize import sent_tokenize

def multi_lingual_late_chunking(document, max_chunk_size=200):
    # Detect the primary language of the document
    primary_language = detect(document)

    # Use language-specific sentence tokenizer if available
    try:
        sentences = sent_tokenize(document, language=primary_language)
    except:
        sentences = sent_tokenize(document)  # Fallback to default tokenizer

    chunks = []
    current_chunk = ""
    current_lang = primary_language

    for sentence in sentences:
        sentence_lang = detect(sentence)

        if len(current_chunk) + len(sentence) <= max_chunk_size and sentence_lang == current_lang:
            current_chunk += sentence + " "
        else:
            if current_chunk:
                chunks.append((current_lang, current_chunk.strip()))
            current_chunk = sentence + " "
            current_lang = sentence_lang

    if current_chunk:
        chunks.append((current_lang, current_chunk.strip()))

    return chunks

# Example usage
multi_lingual_document = """
This is an English sentence. Este es una oración en español.
これは日本語の文章です。 This is another English sentence.
Это предложение на русском языке. And back to English.
"""

chunks = multi_lingual_late_chunking(multi_lingual_document)
for i, (lang, chunk) in enumerate(chunks, 1):
    print(f"Chunk {i} (Language: {lang}):")
    print(chunk)
    print()
```

Slide 9: Phân đoạn muộn để tóm tắt tài liệu dài

Phân đoạn muộn có thể đặc biệt hữu ích cho việc tóm tắt các tài liệu dài. Bằng cách tạo các phần có ý nghĩa về mặt ngữ nghĩa, chúng ta có thể tạo các bản tóm tắt cho từng phần và sau đó kết hợp chúng thành một bản tóm tắt tổng thể mạch lạc.

```python
from transformers import pipeline

def summarize_with_late_chunking(document, max_chunk_size=500):
    chunks = late_chunking(document, max_chunk_size)
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    chunk_summaries = []
    for chunk in chunks:
        summary = summarizer(chunk, max_length=50, min_length=10, do_sample=False)[0]['summary_text']
        chunk_summaries.append(summary)

    final_summary = " ".join(chunk_summaries)
    return final_summary

# Example usage
long_document = """
Artificial Intelligence (AI) is revolutionizing various industries. It encompasses machine learning, deep learning, and neural networks. AI systems can process vast amounts of data and make decisions based on patterns and insights.

Natural Language Processing (NLP) is a branch of AI that focuses on the interaction between computers and human language. NLP applications include machine translation, sentiment analysis, and chatbots. These technologies are improving communication between humans and machines.

Computer Vision is another important field within AI. It deals with how computers gain high-level understanding from digital images or videos. Applications of computer vision include facial recognition, autonomous vehicles, and medical image analysis.

Robotics is an interdisciplinary field that integrates AI, mechanical engineering, and electronics. Robots are being used in manufacturing, healthcare, and exploration. As AI advances, robots are becoming more autonomous and capable of complex tasks.

Ethics in AI is a growing concern. As AI systems become more powerful, questions arise about privacy, bias, and the impact on employment. Researchers and policymakers are working to develop guidelines for responsible AI development and deployment.
"""

summary = summarize_with_late_chunking(long_document)
print("Document Summary:")
print(summary)
```

Trang trình bày 10: Ví dụ thực tế: Trả lời câu hỏi muộn

Trong ví dụ này, chúng tôi sẽ sử dụng Phân đoạn muộn để cải thiện hệ thống trả lời câu hỏi. Bằng cách phân chia bối cảnh một cách linh hoạt dựa trên câu hỏi, chúng tôi có thể cung cấp thông tin phù hợp hơn cho mô hình.

```python
from transformers import pipeline

def qa_with_late_chunking(question, context, max_chunk_size=300):
    chunks = query_aware_chunking(context, question, max_chunk_size)
    qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

    best_answer = None
    best_score = 0

    for chunk in chunks:
        result = qa_pipeline(question=question, context=chunk)
        if result['score'] > best_score:
            best_answer = result['answer']
            best_score = result['score']

    return best_answer, best_score

# Example usage
context = """
The Python programming language was created by Guido van Rossum and first released in 1991.
It is known for its simplicity and readability, making it popular among beginners and experienced programmers alike.
Python is widely used in various fields, including web development, data science, artificial intelligence, and scientific computing.
Python's design philosophy emphasizes code readability with its notable use of significant whitespace.
"""

question = "Who created Python?"
answer, score = qa_with_late_chunking(question, context)
print(f"Question: {question}")
print(f"Answer: {answer}")
print(f"Confidence score: {score:.2f}")
```

Trang trình chiếu 11: Ví dụ thực tế: Phân đoạn muộn để phân loại văn bản

Trong ví dụ này, chúng tôi sẽ sử dụng Phân đoạn muộn để cải thiện hệ thống phân loại văn bản. Bằng cách chia nhỏ các tài liệu dài thành các phần nhỏ hơn, dễ quản lý hơn, chúng ta có thể phân loại từng phần và sau đó tổng hợp các kết quả để phân loại cuối cùng.

```python
from transformers import pipeline
from collections import Counter

def classify_with_late_chunking(document, max_chunk_size=200):
    chunks = late_chunking(document, max_chunk_size)
    classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

    chunk_classifications = []
    for chunk in chunks:
        result = classifier(chunk)[0]
        chunk_classifications.append(result['label'])

    # Aggregate classifications
    final_classification = Counter(chunk_classifications).most_common(1)[0][0]
    return final_classification

# Example usage
document = """
Renewable energy sources are becoming increasingly important in the fight against climate change.
Solar power harnesses the energy of the sun to generate electricity, while wind turbines convert the kinetic energy of wind into electrical power.
These clean energy sources reduce our reliance on fossil fuels and help to decrease greenhouse gas emissions.
However, challenges remain in terms of energy storage and grid integration for these intermittent sources.
Despite these challenges, many countries are setting ambitious targets for renewable energy adoption to create a more sustainable future.
"""

classification = classify_with_late_chunking(document)
print(f"Document classification: {classification}")
```

Trang trình bày 12: Tối ưu hóa hiệu suất phân đoạn muộn

Để nâng cao hiệu quả của Late Chunking trong môi trường sản xuất, chúng tôi có thể triển khai bộ nhớ đệm và xử lý song song. Cách tiếp cận này làm giảm các tính toán dư thừa và tận dụng các hệ thống đa lõi.

```python
import multiprocessing
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_late_chunking(document, max_chunk_size=200):
    return late_chunking(document, max_chunk_size)

def parallel_late_chunking(documents, max_chunk_size=200):
    with multiprocessing.Pool() as pool:
        chunks_list = pool.starmap(cached_late_chunking,
                                   [(doc, max_chunk_size) for doc in documents])
    return chunks_list

# Example usage
documents = [
    "This is the first document. " * 10,
    "Here's another document with different content. " * 10,
    "A third document for demonstration purposes. " * 10
]

all_chunks = parallel_late_chunking(documents)
for i, doc_chunks in enumerate(all_chunks, 1):
    print(f"Document {i} chunks: {len(doc_chunks)}")
    print(f"First chunk: {doc_chunks[0][:50]}...")
    print()
```

Slide 13: Đánh giá chất lượng phân đoạn muộn

Để đảm bảo tính hiệu quả của Phân đoạn muộn, điều quan trọng là phải đánh giá chất lượng của các khối được tạo ra. Chúng ta có thể sử dụng các số liệu như tính mạch lạc về mặt ngữ nghĩa và khả năng lưu giữ thông tin để đánh giá chất lượng khối.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def evaluate_chunk_quality(original_document, chunks):
    # Semantic coherence
    vectorizer = TfidfVectorizer()
    chunk_vectors = vectorizer.fit_transform(chunks)
    coherence_scores = cosine_similarity(chunk_vectors)
    avg_coherence = np.mean(coherence_scores)

    # Information retention
    original_vector = vectorizer.transform([original_document])
    combined_chunks_vector = vectorizer.transform([" ".join(chunks)])
    retention_score = cosine_similarity(original_vector, combined_chunks_vector)[0][0]

    return {
        "average_coherence": avg_coherence,
        "information_retention": retention_score
    }

# Example usage
document = "Late Chunking is a technique used in natural language processing. " * 10
chunks = late_chunking(document)

quality_metrics = evaluate_chunk_quality(document, chunks)
print("Chunk Quality Metrics:")
print(f"Average Coherence: {quality_metrics['average_coherence']:.2f}")
print(f"Information Retention: {quality_metrics['information_retention']:.2f}")
```

Trang trình bày 14: Tài nguyên bổ sung

Để khám phá thêm về Phân đoạn muộn và các ứng dụng của nó trong quy trình công việc RAG, hãy xem xét các tài nguyên sau:

1. Bài báo ArXiv: "Thế hệ tăng cường truy xuất cho các nhiệm vụ NLP chuyên sâu về tri thức" của Lewis và cộng sự. (2020) URL: [https://arxiv.org/abs/2005.11401](https://arxiv.org/abs/2005.11401)
2. Bài báo ArXiv: "Sự chú ý là tất cả những gì bạn cần" của Vaswani et al. (2017), giới thiệu kiến trúc Transformer được sử dụng trong nhiều mô hình NLP hiện đại. URL: [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
3. Tài liệu thư viện Hugging Face Transformers, cung cấp cách triển khai các mô hình và kỹ thuật NLP khác nhau. URL: [https://huggingface.co/transformers/](https://huggingface.co/transformers/)
4. Tài liệu Python chính thức về đa xử lý, hữu ích để tối ưu hóa URL hiệu suất Phân đoạn muộn: [https://docs.python.org/3/library/multiprocessing.html](https://docs.python.org/3/library/multiprocessing.html)

Những tài nguyên này cung cấp những hiểu biết sâu sắc có giá trị về nền tảng lý thuyết và cách triển khai thực tế các kỹ thuật liên quan đến quy trình làm việc Chunking muộn và RAG.
