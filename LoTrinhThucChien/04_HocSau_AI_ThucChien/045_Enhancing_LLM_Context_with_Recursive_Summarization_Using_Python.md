## Nâng cao bối cảnh LLM bằng tóm tắt đệ quy bằng Python
Trang trình bày 1: Giới thiệu về Nâng cao bối cảnh LLM

Mô hình ngôn ngữ lớn (LLM) có cửa sổ ngữ cảnh hạn chế. Tóm tắt đệ quy là một kỹ thuật để mở rộng bối cảnh này bằng cách cô đọng thông tin lặp đi lặp lại. Cách tiếp cận này cho phép LLM xử lý các tài liệu lớn hơn trong khi vẫn giữ được thông tin quan trọng.

```python
import transformers

def load_llm():
    model = transformers.AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    return model, tokenizer

model, tokenizer = load_llm()
```

Slide 2: Tìm hiểu ngữ cảnh của Windows

Cửa sổ ngữ cảnh xác định số lượng văn bản tối đa mà LLM có thể xử lý cùng một lúc. Ví dụ: GPT-3 có cửa sổ ngữ cảnh gồm 4096 mã thông báo. Tóm tắt đệ quy giúp khắc phục hạn chế này bằng cách cô đọng các văn bản dài thành các bản tóm tắt ngắn hơn, giàu thông tin hơn.

```python
def get_context_window(model):
    return model.config.max_position_embeddings

context_window = get_context_window(model)
print(f"Model context window: {context_window} tokens")
```

Trang trình bày 3: Phân đoạn văn bản

Bước đầu tiên trong quá trình tóm tắt đệ quy là chia văn bản đầu vào thành các phần có thể quản lý được, vừa với cửa sổ ngữ cảnh của LLM. Điều này đảm bảo rằng mỗi đoạn có thể được xử lý độc lập.

```python
def chunk_text(text, max_chunk_size):
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        if len(" ".join(current_chunk + [word])) <= max_chunk_size:
            current_chunk.append(word)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
```

Slide 4: Tóm tắt từng phần riêng lẻ

Sau khi phân đoạn, mỗi đoạn văn bản được tóm tắt độc lập. Điều này làm giảm nội dung trong khi vẫn giữ được thông tin quan trọng. Quá trình tóm tắt có thể được tùy chỉnh dựa trên các yêu cầu cụ thể của ứng dụng của bạn.

```python
def summarize_chunk(chunk, model, tokenizer):
    inputs = tokenizer(chunk, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs.input_ids, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
```

Slide 5: Recursive Summarization Process

Recursive summarization involves repeatedly summarizing the summaries until the entire text fits within the context window. This process creates a hierarchical structure of summaries, with each level containing more condensed information.

```python
def recursive_summarize(text, model, tokenizer, max_chunk_size, target_size):
    if len(text) <= target_size:
        return text

    chunks = chunk_text(text, max_chunk_size)
    summaries = [summarize_chunk(chunk, model, tokenizer) for chunk in chunks]
    combined_summary = " ".join(summaries)

    return recursive_summarize(combined_summary, model, tokenizer, max_chunk_size, target_size)
```

Slide 6: Xử lý tài liệu dài

Đối với những tài liệu cực kỳ dài, quá trình tóm tắt đệ quy có thể cần phải được áp dụng nhiều lần. Điều này đảm bảo rằng bản tóm tắt cuối cùng vừa với cửa sổ ngữ cảnh của LLM trong khi vẫn nắm bắt được bản chất của toàn bộ tài liệu.

```python
def process_long_document(document, model, tokenizer, max_chunk_size, target_size):
    sections = document.split("\n\n")  # Assuming sections are separated by double newlines
    section_summaries = []

    for section in sections:
        summary = recursive_summarize(section, model, tokenizer, max_chunk_size, target_size // len(sections))
        section_summaries.append(summary)

    return " ".join(section_summaries)
```

Trang trình bày 7: Bảo toàn hệ thống phân cấp bối cảnh

Để duy trì cấu trúc của tài liệu, điều quan trọng là phải duy trì thứ bậc thông tin trong quá trình tóm tắt. Điều này có thể đạt được bằng cách tóm tắt ở các cấp độ khác nhau (ví dụ: đoạn văn, phần, chương) và kết hợp các kết quả.

```python
def hierarchical_summarization(document, model, tokenizer, max_chunk_size, target_size):
    chapters = document.split("Chapter")
    chapter_summaries = []

    for chapter in chapters[1:]:  # Skip the first empty split
        sections = chapter.split("Section")
        section_summaries = []

        for section in sections[1:]:  # Skip the first empty split
            summary = recursive_summarize(section, model, tokenizer, max_chunk_size, target_size // (len(chapters) * len(sections)))
            section_summaries.append(summary)

        chapter_summary = " ".join(section_summaries)
        chapter_summaries.append(chapter_summary)

    return " ".join(chapter_summaries)
```

Slide 8: Cân bằng nén và lưu giữ thông tin

Tìm sự cân bằng phù hợp giữa nén và lưu giữ thông tin là rất quan trọng. Thử nghiệm các tỷ lệ và kỹ thuật tóm tắt khác nhau để đạt được kết quả tối ưu cho trường hợp sử dụng cụ thể của bạn.

```python
def adaptive_summarization(text, model, tokenizer, max_chunk_size, target_size, compression_ratio=0.5):
    if len(text) <= target_size:
        return text

    chunks = chunk_text(text, max_chunk_size)
    summaries = []

    for chunk in chunks:
        chunk_target_size = int(len(chunk) * compression_ratio)
        summary = summarize_chunk(chunk, model, tokenizer)

        if len(summary) > chunk_target_size:
            summary = summary[:chunk_target_size]

        summaries.append(summary)

    combined_summary = " ".join(summaries)

    if len(combined_summary) <= target_size:
        return combined_summary
    else:
        return adaptive_summarization(combined_summary, model, tokenizer, max_chunk_size, target_size, compression_ratio * 0.9)
```

Trang trình bày 9: Triển khai mã thông báo tùy chỉnh

Để kiểm soát nhiều hơn quá trình tóm tắt, hãy triển khai mã thông báo tùy chỉnh phù hợp với miền hoặc ngôn ngữ cụ thể của bạn. Điều này có thể cải thiện chất lượng tóm tắt cho các văn bản chuyên ngành.

```python
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace

def train_custom_tokenizer(texts):
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordPieceTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

    tokenizer.train_from_iterator(texts, trainer)
    return tokenizer

custom_tokenizer = train_custom_tokenizer(your_text_corpus)
```

Slide 10: Tăng cường tóm tắt bằng cách trích xuất thông tin chính

Cải thiện chất lượng của bản tóm tắt bằng cách trích xuất và ưu tiên thông tin chính như thực thể được đặt tên, ngày tháng hoặc thuật ngữ dành riêng cho tên miền. Điều này đảm bảo rằng các chi tiết quan trọng được giữ nguyên trong bản tóm tắt cuối cùng.

```python
import spacy

nlp = spacy.load("en_core_web_sm")

def extract_key_info(text):
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents]
    key_phrases = [chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) > 1]
    return list(set(entities + key_phrases))

def enhanced_summarization(chunk, model, tokenizer, key_info):
    summary = summarize_chunk(chunk, model, tokenizer)
    key_info_text = ", ".join(key_info)
    enhanced_summary = f"{summary}\n\nKey information: {key_info_text}"
    return enhanced_summary
```

Slide 11: Xử lý đầu vào đa phương thức

Mở rộng kỹ thuật tóm tắt đệ quy để xử lý đầu vào đa phương thức, chẳng hạn như văn bản có hình ảnh hoặc bảng. Điều này đòi hỏi phải điều chỉnh quy trình tóm tắt để kết hợp thông tin từ các phương thức khác nhau.

```python
from PIL import Image
import pytesseract

def extract_text_from_image(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text

def summarize_multimodal_content(text, image_paths, model, tokenizer, max_chunk_size, target_size):
    image_texts = [extract_text_from_image(img_path) for img_path in image_paths]
    combined_text = text + "\n" + "\n".join(image_texts)
    return recursive_summarize(combined_text, model, tokenizer, max_chunk_size, target_size)
```

Slide 12: Đánh giá chất lượng tóm tắt

Đánh giá chất lượng của các bản tóm tắt được tạo bằng cách sử dụng các số liệu như điểm ROUGE hoặc độ tương tự về ngữ nghĩa. Điều này giúp tinh chỉnh quá trình tóm tắt và đảm bảo rằng phương pháp đệ quy duy trì độ chính xác của nội dung.

```python
from rouge import Rouge
from sentence_transformers import SentenceTransformer, util

def evaluate_summary(original_text, summary):
    rouge = Rouge()
    scores = rouge.get_scores(summary, original_text)

    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    original_embedding = model.encode(original_text, convert_to_tensor=True)
    summary_embedding = model.encode(summary, convert_to_tensor=True)
    semantic_similarity = util.pytorch_cos_sim(original_embedding, summary_embedding).item()

    return {
        'rouge': scores[0],
        'semantic_similarity': semantic_similarity
    }
```

Trang trình bày 13: Tối ưu hóa cho các ứng dụng thời gian thực

Đối với các ứng dụng thời gian thực, hãy tối ưu hóa quy trình tóm tắt đệ quy để giảm độ trễ. Triển khai cơ chế bộ nhớ đệm và xử lý song song để cải thiện hiệu suất khi xử lý khối lượng văn bản lớn.

```python
import concurrent.futures
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_summarize_chunk(chunk, model_name, tokenizer_name):
    model, tokenizer = load_llm(model_name, tokenizer_name)
    return summarize_chunk(chunk, model, tokenizer)

def parallel_summarize(chunks, model_name, tokenizer_name, max_workers=4):
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(cached_summarize_chunk, chunk, model_name, tokenizer_name) for chunk in chunks]
        summaries = [future.result() for future in concurrent.futures.as_completed(futures)]
    return summaries
```

Slide 14: Tích hợp với hệ thống truy xuất tài liệu

Kết hợp tóm tắt đệ quy với hệ thống truy xuất tài liệu để nâng cao khả năng tìm kiếm. Sử dụng các bản tóm tắt được tạo để tạo thêm chỉ mục tìm kiếm nhiều thông tin hơn và cải thiện khả năng kết hợp truy vấn.

```python
from elasticsearch import Elasticsearch

def index_document_with_summary(es, doc_id, original_text, summary):
    es.index(index="documents", id=doc_id, body={
        "original_text": original_text,
        "summary": summary
    })

def search_documents(es, query, size=10):
    result = es.search(index="documents", body={
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["original_text", "summary^2"]
            }
        },
        "size": size
    })
    return result['hits']['hits']

es = Elasticsearch()
index_document_with_summary(es, "doc1", original_text, summary)
search_results = search_documents(es, "your search query")
```

Trang trình bày 15: Tài nguyên bổ sung

1. "Tóm tắt đệ quy để hiểu tài liệu dài" của Balachandran et al. (2023) arXiv:2301.13703 \[cs.CL\] [https://arxiv.org/abs/2301.13703](https://arxiv.org/abs/2301.13703)
2. "Longformer: Máy biến áp tài liệu dài" của Beltagy et al. (2020) arXiv:2004.05150 \[cs.CL\] [https://arxiv.org/abs/2004.05150](https://arxiv.org/abs/2004.05150)
3. "BART: Đào tạo trước về khử nhiễu từ trình tự này sang trình tự khác để tạo, dịch và hiểu ngôn ngữ tự nhiên" của Lewis và cộng sự. (2019) arXiv:1910.13461 \[cs.CL\] [https://arxiv.org/abs/1910.13461](https://arxiv.org/abs/1910.13461)
