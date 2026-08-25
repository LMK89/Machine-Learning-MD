## Xây dựng các mô hình NLP mạnh mẽ với khả năng tăng cường dữ liệu
Trang trình bày 1: Giới thiệu về Tăng cường dữ liệu trong NLP

Tăng cường dữ liệu là một kỹ thuật được sử dụng để tăng tính đa dạng và quy mô của dữ liệu huấn luyện bằng cách tạo các phiên bản sửa đổi của dữ liệu hiện có. Trong Xử lý ngôn ngữ tự nhiên (NLP), điều này giúp xây dựng các mô hình mạnh mẽ hơn có thể khái quát hóa tốt hơn dữ liệu không nhìn thấy được.

```python
import nlpaug.augmenter.word as naw

# Example of a simple augmentation
augmenter = naw.SynonymAug(aug_src='wordnet')
text = "The quick brown fox jumps over the lazy dog"
augmented_text = augmenter.augment(text)
print(augmented_text)
```

Trang trình bày 2: Tại sao phải tăng cường dữ liệu?

Tăng cường dữ liệu giúp vượt qua những thách thức chung trong NLP:

1. Dữ liệu được dán nhãn hạn chế
2. Bộ dữ liệu mất cân bằng
3. Trang bị quá mức
4. Cải thiện khả năng khái quát hóa mô hình

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Load a sample dataset
df = pd.read_csv('sentiment_data.csv')
X = df['text']
y = df['sentiment']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
```

Trang trình bày 3: Các loại tăng cường dữ liệu trong NLP

1. Thay thế từ vựng
2. Dịch ngược
3. Tạo văn bản
4. Tạo tiếng ồn
5. Hoán vị câu

```python
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas

text = "I love this movie, it's amazing!"

# Lexical Substitution
aug_synonym = naw.SynonymAug(aug_src='wordnet')
print("Synonym:", aug_synonym.augment(text))

# Back-Translation
aug_back_translation = naw.BackTranslationAug(from_model_name='facebook/wmt19-en-de', to_model_name='facebook/wmt19-de-en')
print("Back-Translation:", aug_back_translation.augment(text))

# Sentence Permutation
aug_sentence = nas.ContextualWordEmbsForSentenceAug(model_path='distilbert-base-uncased')
print("Sentence Augmentation:", aug_sentence.augment(text))
```

Slide 4: Thay thế từ vựng

Thay thế từ vựng liên quan đến việc thay thế các từ bằng từ đồng nghĩa, từ trái nghĩa hoặc các từ liên quan. Kỹ thuật này giúp mô hình tìm hiểu các mối quan hệ ngữ nghĩa và cải thiện phạm vi từ vựng.

```python
import nlpaug.augmenter.word as naw

text = "The cat is sleeping on the couch"

# Synonym replacement
aug_synonym = naw.SynonymAug(aug_src='wordnet')
print("Synonym:", aug_synonym.augment(text))

# Antonym replacement
aug_antonym = naw.AntonymAug()
print("Antonym:", aug_antonym.augment(text))

# Word embedding replacement
aug_w2v = naw.WordEmbsAug(model_type='word2vec', model_path='./word2vec.bin')
print("Word Embedding:", aug_w2v.augment(text))
```

Slide 5: Dịch ngược

Dịch ngược bao gồm việc dịch văn bản sang ngôn ngữ khác và sau đó quay lại ngôn ngữ gốc. Kỹ thuật này giới thiệu các cụm từ và cấu trúc câu đa dạng.

```python
from transformers import MarianMTModel, MarianTokenizer

def back_translate(text, source_lang="en", target_lang="fr"):
    # Load models
    model_name = f'Helsinki-NLP/opus-mt-{source_lang}-{target_lang}'
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)

    # Translate to target language
    translated = model.generate(**tokenizer(text, return_tensors="pt", padding=True))
    tgt_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated][0]

    # Translate back to source language
    model_name = f'Helsinki-NLP/opus-mt-{target_lang}-{source_lang}'
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)

    back_translated = model.generate(**tokenizer(tgt_text, return_tensors="pt", padding=True))
    back_text = [tokenizer.decode(t, skip_special_tokens=True) for t in back_translated][0]

    return back_text

original_text = "The weather is beautiful today"
augmented_text = back_translate(original_text)
print(f"Original: {original_text}")
print(f"Augmented: {augmented_text}")
```

Slide 6: Tạo văn bản

Tạo văn bản liên quan đến việc tạo văn bản mới dựa trên dữ liệu hiện có. Điều này có thể được thực hiện bằng cách sử dụng các mô hình ngôn ngữ hoặc hệ thống dựa trên quy tắc để mở rộng tập dữ liệu bằng các ví dụ tổng hợp.

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_text(prompt, max_length=50):
    model_name = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2)

    return tokenizer.decode(output[0], skip_special_tokens=True)

prompt = "The restaurant was"
generated_text = generate_text(prompt)
print(f"Generated text: {generated_text}")
```

Trang trình bày 7: Tạo tiếng ồn

Việc chèn tiếng ồn liên quan đến việc thêm các nhiễu loạn ngẫu nhiên vào văn bản, chẳng hạn như lỗi chính tả, hoán đổi ký tự hoặc xóa từ. Kỹ thuật này giúp tạo ra một mô hình mạnh mẽ hơn có thể xử lý đầu vào không hoàn hảo.

```python
import random
import string

def add_noise(text, p=0.1):
    words = text.split()
    noisy_words = []

    for word in words:
        if random.random() < p:
            noise_type = random.choice(['swap', 'delete', 'insert'])
            if noise_type == 'swap' and len(word) > 1:
                i, j = random.sample(range(len(word)), 2)
                word = list(word)
                word[i], word[j] = word[j], word[i]
                word = ''.join(word)
            elif noise_type == 'delete' and len(word) > 1:
                i = random.randint(0, len(word) - 1)
                word = word[:i] + word[i+1:]
            elif noise_type == 'insert':
                i = random.randint(0, len(word))
                char = random.choice(string.ascii_lowercase)
                word = word[:i] + char + word[i:]
        noisy_words.append(word)

    return ' '.join(noisy_words)

original_text = "The quick brown fox jumps over the lazy dog"
noisy_text = add_noise(original_text)
print(f"Original: {original_text}")
print(f"Noisy: {noisy_text}")
```

Slide 8: Hoán vị câu

Hoán vị câu liên quan đến việc thay đổi thứ tự các câu trong tài liệu hoặc tạo ra các tổ hợp câu mới. Kỹ thuật này giúp người mẫu tìm hiểu các cấu trúc diễn ngôn khác nhau và cải thiện sự hiểu biết mạch lạc.

```python
import random

def permute_sentences(text):
    sentences = text.split('.')
    sentences = [s.strip() for s in sentences if s.strip()]
    random.shuffle(sentences)
    return '. '.join(sentences) + '.'

original_text = "I went to the store. It was a sunny day. I bought some groceries. The cashier was friendly."
permuted_text = permute_sentences(original_text)
print(f"Original: {original_text}")
print(f"Permuted: {permuted_text}")
```

Trang trình bày 9: Triển khai tăng cường dữ liệu trong đường ống

Việc tích hợp tăng cường dữ liệu vào quy trình NLP của bạn liên quan đến việc áp dụng các kỹ thuật tăng cường cho dữ liệu đào tạo của bạn trước khi đào tạo mô hình.

```python
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
import torch

def augment_data(texts, labels, aug_technique, aug_factor=2):
    augmented_texts, augmented_labels = [], []
    for text, label in zip(texts, labels):
        augmented_texts.append(text)
        augmented_labels.append(label)
        for _ in range(aug_factor - 1):
            aug_text = aug_technique(text)
            augmented_texts.append(aug_text)
            augmented_labels.append(label)
    return augmented_texts, augmented_labels

# Assume we have texts and labels
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2)

# Apply augmentation
X_train_aug, y_train_aug = augment_data(X_train, y_train, aug_technique=add_noise)

# Tokenize and create dataset
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenizer(X_train_aug, truncation=True, padding=True)
train_dataset = torch.utils.data.TensorDataset(
    torch.tensor(train_encodings['input_ids']),
    torch.tensor(train_encodings['attention_mask']),
    torch.tensor(y_train_aug)
)

# Train model (simplified)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
# ... (training loop)
```

Trang trình bày 10: Đánh giá tác động tăng cường

Điều quan trọng là đánh giá tác động của việc tăng cường dữ liệu đến hiệu suất mô hình của bạn. So sánh hiệu suất của mô hình có và không có tăng cường.

```python
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

def evaluate_model(model, X_test, y_test):
    with torch.no_grad():
        inputs = tokenizer(X_test, return_tensors="pt", padding=True, truncation=True)
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)

    return accuracy, report

# Assume we have trained two models: model_no_aug and model_with_aug

accuracy_no_aug, report_no_aug = evaluate_model(model_no_aug, X_test, y_test)
accuracy_with_aug, report_with_aug = evaluate_model(model_with_aug, X_test, y_test)

print(f"Accuracy without augmentation: {accuracy_no_aug}")
print(f"Accuracy with augmentation: {accuracy_with_aug}")
print("\nClassification Report (No Augmentation):")
print(report_no_aug)
print("\nClassification Report (With Augmentation):")
print(report_with_aug)
```

Slide 11: Ví dụ thực tế: Phân tích cảm xúc

Hãy áp dụng tính năng tăng cường dữ liệu cho tác vụ phân tích cảm tính bằng cách sử dụng các bài đánh giá phim.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import nlpaug.augmenter.word as naw
import torch

# Load data (assume we have a CSV with 'review' and 'sentiment' columns)
df = pd.read_csv('movie_reviews.csv')
X = df['review'].tolist()
y = df['sentiment'].tolist()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Augmentation function
aug_synonym = naw.SynonymAug(aug_src='wordnet')

def augment_data(texts, labels, aug_factor=2):
    augmented_texts, augmented_labels = [], []
    for text, label in zip(texts, labels):
        augmented_texts.append(text)
        augmented_labels.append(label)
        for _ in range(aug_factor - 1):
            aug_text = aug_synonym.augment(text)
            augmented_texts.append(aug_text)
            augmented_labels.append(label)
    return augmented_texts, augmented_labels

# Apply augmentation
X_train_aug, y_train_aug = augment_data(X_train, y_train)

# Tokenize and create datasets
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenizer(X_train_aug, truncation=True, padding=True)
test_encodings = tokenizer(X_test, truncation=True, padding=True)

train_dataset = torch.utils.data.TensorDataset(
    torch.tensor(train_encodings['input_ids']),
    torch.tensor(train_encodings['attention_mask']),
    torch.tensor(y_train_aug)
)

test_dataset = torch.utils.data.TensorDataset(
    torch.tensor(test_encodings['input_ids']),
    torch.tensor(test_encodings['attention_mask']),
    torch.tensor(y_test)
)

# Train model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

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
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()

# Evaluate
results = trainer.evaluate()
print(results)
```

Trang trình bày 12: Ví dụ thực tế: Nhận dạng thực thể được đặt tên (NER)

Hãy áp dụng tính năng tăng cường dữ liệu cho tác vụ Nhận dạng thực thể được đặt tên bằng cách sử dụng các bài báo.

```python
import spacy
import random

nlp = spacy.load("en_core_web_sm")

train_data = [
    ("Apple Inc. is planning to open a new store in New York City.", {"entities": [(0, 9, "ORG"), (41, 54, "GPE")]}),
    ("Microsoft announced a partnership with OpenAI.", {"entities": [(0, 9, "ORG"), (37, 42, "ORG")]})
]

def augment_ner_data(text, entities):
    doc = nlp(text)
    augmented_text = []
    augmented_entities = []

    for token in doc:
        if random.random() < 0.1 and token.pos_ in ["NOUN", "VERB", "ADJ"]:
            synonyms = [syn.lower_ for syn in token._.synonyms]
            if synonyms:
                replacement = random.choice(synonyms)
                augmented_text.append(replacement)
            else:
                augmented_text.append(token.text)
        else:
            augmented_text.append(token.text)

    augmented_text = " ".join(augmented_text)

    for start, end, label in entities["entities"]:
        new_start = len(" ".join(augmented_text.split()[:start]))
        new_end = new_start + len(" ".join(augmented_text.split()[start:end]))
        augmented_entities.append((new_start, new_end, label))

    return augmented_text, {"entities": augmented_entities}

augmented_train_data = []
for text, annotations in train_data:
    augmented_train_data.append((text, annotations))
    for _ in range(2):  # Create 2 augmented examples for each original
        aug_text, aug_annotations = augment_ner_data(text, annotations)
        augmented_train_data.append((aug_text, aug_annotations))

print(f"Original dataset size: {len(train_data)}")
print(f"Augmented dataset size: {len(augmented_train_data)}")
print("\nSample augmented data:")
print(augmented_train_data[2])
```

Slide 13: Những thách thức và cân nhắc

Khi triển khai tăng cường dữ liệu cho NLP:

1. Giữ nguyên ý nghĩa ngữ nghĩa
2. Duy trì tính nhất quán của nhãn
3. Kỹ thuật tăng thăng bằng
4. Tránh đưa ra thành kiến

```python
def check_augmentation_quality(original, augmented):
    original_doc = nlp(original)
    augmented_doc = nlp(augmented)

    # Check semantic similarity
    similarity = original_doc.similarity(augmented_doc)

    # Check label consistency (example for sentiment analysis)
    original_sentiment = original_doc.sentiment
    augmented_sentiment = augmented_doc.sentiment

    print(f"Semantic similarity: {similarity}")
    print(f"Original sentiment: {original_sentiment}")
    print(f"Augmented sentiment: {augmented_sentiment}")

    if similarity < 0.7 or abs(original_sentiment - augmented_sentiment) > 0.3:
        print("Warning: Augmentation may have altered meaning or label.")

original_text = "The movie was fantastic and I enjoyed every minute of it."
augmented_text = "The film was terrific and I relished each moment of it."

check_augmentation_quality(original_text, augmented_text)
```

Trang trình bày 14: Các phương pháp hay nhất để tăng cường dữ liệu NLP

1. Thử nghiệm nhiều kỹ thuật
2. Sử dụng tính năng bổ sung dành riêng cho tên miền khi có thể
3. Giám sát tác động đến hiệu suất của mô hình
4. Thường xuyên cập nhật các chiến lược gia tăng

```python
def augmentation_pipeline(text, techniques):
    augmented_texts = [text]
    for technique in techniques:
        new_text = technique(text)
        augmented_texts.append(new_text)
    return augmented_texts

# Example usage
techniques = [
    lambda x: add_noise(x, p=0.1),
    lambda x: back_translate(x, source_lang="en", target_lang="fr"),
    lambda x: aug_synonym.augment(x)
]

sample_text = "The weather is beautiful today"
augmented_samples = augmentation_pipeline(sample_text, techniques)

for i, sample in enumerate(augmented_samples):
    print(f"Sample {i}: {sample}")
```

Trang trình bày 15: Tài nguyên bổ sung

Để khám phá thêm về tăng cường dữ liệu trong NLP:

1. URL "Khảo sát về các phương pháp tăng cường dữ liệu cho NLP" (ArXiv:2105.03075): [https://arxiv.org/abs/2105.03075](https://arxiv.org/abs/2105.03075)
2. URL "EDA: Kỹ thuật tăng cường dữ liệu dễ dàng để tăng hiệu suất cho các nhiệm vụ phân loại văn bản" (ArXiv:1901.11196) URL: [https://arxiv.org/abs/1901.11196](https://arxiv.org/abs/1901.11196)
3. URL "Tăng cường dữ liệu bằng cách sử dụng các mô hình máy biến áp được đào tạo trước" (ArXiv:2003.02245): [https://arxiv.org/abs/2003.02245](https://arxiv.org/abs/2003.02245)

Các tài nguyên này cung cấp các cuộc thảo luận chuyên sâu về các kỹ thuật tăng cường dữ liệu khác nhau và ứng dụng của chúng trong các nhiệm vụ NLP.
