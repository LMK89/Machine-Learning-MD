## Tinh chỉnh LLM với Điểm PEFT và ROUGE
Trang trình bày 1: Giới thiệu về Tinh chỉnh các mô hình ngôn ngữ lớn

Tinh chỉnh các mô hình ngôn ngữ lớn (LLM) đã trở thành một kỹ thuật quan trọng trong xử lý ngôn ngữ tự nhiên. Quá trình này bao gồm việc điều chỉnh các mô hình được đào tạo trước cho phù hợp với các nhiệm vụ hoặc lĩnh vực cụ thể, cải thiện hiệu suất và hiệu quả của chúng. Trong phần trình bày này, chúng ta sẽ khám phá cách tinh chỉnh bằng cách sử dụng các kỹ thuật Tinh chỉnh tham số hiệu quả (PEFT), đặc biệt tập trung vào Thích ứng xếp hạng thấp (LoRA) và đánh giá kết quả bằng cách sử dụng điểm Rouge.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load a pre-trained model
model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(f"Loaded model: {model_name}")
print(f"Model parameters: {model.num_parameters():,}")
```

Trang trình bày 2: Tìm hiểu về Tinh chỉnh hiệu quả tham số (PEFT)

Các kỹ thuật PEFT nhằm mục đích tinh chỉnh các mô hình ngôn ngữ lớn trong khi chỉ cập nhật một tập hợp con nhỏ các tham số của mô hình. Cách tiếp cận này làm giảm đáng kể tài nguyên tính toán và yêu cầu lưu trữ so với tinh chỉnh đầy đủ. Các phương pháp PEFT duy trì kiến ​​thức chung của mô hình trong khi điều chỉnh nó cho phù hợp với các nhiệm vụ cụ thể, khiến chúng trở nên lý tưởng cho các môi trường hạn chế về tài nguyên.

```python
from peft import get_peft_config, PeftModel, PeftConfig, LoraConfig

# Define a LoRA configuration
peft_config = LoraConfig(
    task_type="CAUSAL_LM",
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)

# Apply PEFT to the model
peft_model = PeftModel.from_pretrained(model, peft_config)

print(f"Trainable parameters: {peft_model.num_parameters(train=True):,}")
print(f"Total parameters: {peft_model.num_parameters():,}")
```

Trang trình bày 3: Thích ứng cấp thấp (LoRA): Đi sâu

LoRA là một kỹ thuật PEFT bổ sung các ma trận cấp thấp có thể huấn luyện được vào các lớp của mô hình được huấn luyện trước. Các ma trận này nắm bắt thông tin cụ thể của nhiệm vụ trong khi vẫn giữ nguyên hầu hết mô hình ban đầu. Hiệu quả của LoRA đến từ khả năng tìm hiểu các điều chỉnh có ý nghĩa với một số lượng nhỏ tham số, thường nhỏ hơn 1% kích thước của mô hình ban đầu.

```python
import torch.nn as nn

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4):
        super().__init__()
        self.A = nn.Parameter(torch.randn(in_features, rank))
        self.B = nn.Parameter(torch.zeros(rank, out_features))

    def forward(self, x):
        return x @ (self.A @ self.B)

# Example usage
lora_layer = LoRALayer(768, 768, rank=8)
input_tensor = torch.randn(1, 768)
output = lora_layer(input_tensor)

print(f"Input shape: {input_tensor.shape}")
print(f"Output shape: {output.shape}")
print(f"LoRA parameters: {sum(p.numel() for p in lora_layer.parameters()):,}")
```

Slide 4: Chuẩn bị dữ liệu để tinh chỉnh

Trước khi tinh chỉnh, chúng ta cần chuẩn bị tập dữ liệu của mình. Điều này liên quan đến việc mã hóa văn bản, tạo mặt nạ chú ý và định dạng dữ liệu cho mô hình của chúng tôi. Hãy sử dụng một ví dụ đơn giản về việc chuẩn bị tập dữ liệu để phân tích cảm tính.

```python
from datasets import load_dataset
from torch.utils.data import DataLoader

# Load a sample dataset
dataset = load_dataset("imdb", split="train[:1000]")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(["text"])
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
tokenized_dataset.set_format("torch")

dataloader = DataLoader(tokenized_dataset, shuffle=True, batch_size=8)

print(f"Number of samples: {len(tokenized_dataset)}")
print(f"Sample features: {next(iter(dataloader)).keys()}")
```

Trang trình bày 5: Triển khai Tinh chỉnh LoRA

Bây giờ chúng ta đã chuẩn bị xong dữ liệu, hãy triển khai tinh chỉnh LoRA. Chúng tôi sẽ sử dụng thư viện Hugging Face Transformers cùng với thư viện PEFT để tinh chỉnh mô hình của chúng tôi về nhiệm vụ phân tích cảm xúc.

```python
from transformers import TrainingArguments, Trainer

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
)

# Create a Trainer instance
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Start fine-tuning
trainer.train()

print("Fine-tuning completed!")
print(f"Trained model saved to: {training_args.output_dir}")
```

Trang trình bày 6: Đánh giá các mô hình tinh chỉnh: Giới thiệu về Điểm Rouge

Điểm Rouge (Nghiên cứu định hướng thu hồi để đánh giá Gisting) là một tập hợp các số liệu dùng để đánh giá chất lượng của văn bản được tạo ra, đặc biệt là trong các nhiệm vụ như tóm tắt. Nó so sánh văn bản được tạo với một hoặc nhiều văn bản tham chiếu, đo lường sự chồng chéo của n-gram, chuỗi từ và cặp từ.

```python
from rouge_score import rouge_scorer

def calculate_rouge(prediction, reference):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(prediction, reference)
    return scores

# Example usage
prediction = "The cat sat on the mat."
reference = "A cat is sitting on the mat."
scores = calculate_rouge(prediction, reference)

for metric, score in scores.items():
    print(f"{metric}: {score.fmeasure:.4f}")
```

Trang trình bày 7: Tìm hiểu về số liệu Rouge

Rouge đưa ra một số số liệu, mỗi số liệu nắm bắt các khía cạnh khác nhau của độ tương tự văn bản:

* Rouge-N: Đo sự chồng chéo của n-gram giữa văn bản được tạo và văn bản tham chiếu.
* Rouge-L: Tính dãy con chung dài nhất giữa các văn bản.
* Rouge-W: Một phiên bản có trọng số của Rouge-L thiên về các trận đấu liên tiếp.

Hãy triển khai một hàm để tính các số liệu này:

```python
def detailed_rouge_scores(prediction, reference):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(prediction, reference)

    results = {}
    for metric, score in scores.items():
        results[f"{metric}_precision"] = score.precision
        results[f"{metric}_recall"] = score.recall
        results[f"{metric}_fmeasure"] = score.fmeasure

    return results

# Example usage
prediction = "The quick brown fox jumps over the lazy dog."
reference = "A fast brown fox leaps above a sleepy canine."
detailed_scores = detailed_rouge_scores(prediction, reference)

for metric, value in detailed_scores.items():
    print(f"{metric}: {value:.4f}")
```

Trang trình bày 8: Giải thích điểm Rouge

Điểm Rouge nằm trong khoảng từ 0 đến 1, trong đó điểm cao hơn biểu thị mức độ tương đồng cao hơn giữa văn bản được tạo và văn bản tham chiếu. Tuy nhiên, việc giải thích những điểm số này đòi hỏi bối cảnh và sự hiểu biết về nhiệm vụ cụ thể. Hãy tạo một hàm để cung cấp cách diễn giải định tính về điểm Rouge:

```python
def interpret_rouge_score(score):
    if score < 0.2:
        return "Poor similarity"
    elif score < 0.4:
        return "Fair similarity"
    elif score < 0.6:
        return "Moderate similarity"
    elif score < 0.8:
        return "Good similarity"
    else:
        return "Excellent similarity"

# Example usage
rouge_l_score = 0.65
interpretation = interpret_rouge_score(rouge_l_score)
print(f"Rouge-L score: {rouge_l_score:.2f}")
print(f"Interpretation: {interpretation}")
```

Trang trình bày 9: Ví dụ thực tế: Tinh chỉnh tóm tắt văn bản

Hãy áp dụng kiến ​​thức của chúng ta vào một tình huống thực tế: tinh chỉnh mô hình tóm tắt văn bản. Chúng tôi sẽ sử dụng một tập dữ liệu nhỏ gồm các bài báo và phần tóm tắt của chúng để minh họa quy trình.

```python
from datasets import load_dataset

# Load a summarization dataset
dataset = load_dataset("cnn_dailymail", "3.0.0", split="train[:100]")

# Prepare the data
def preprocess_function(examples):
    inputs = ["summarize: " + doc for doc in examples["article"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

    labels = tokenizer(examples["highlights"], max_length=128, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Fine-tune the model
trainer = Trainer(
    model=peft_model,
    args=TrainingArguments(output_dir="./summarization_model", num_train_epochs=3),
    train_dataset=tokenized_dataset,
)

trainer.train()
print("Fine-tuning for summarization completed!")
```

Slide 10: Đánh giá mô hình tóm tắt

Bây giờ chúng ta đã tinh chỉnh mô hình của mình để tóm tắt, hãy đánh giá hiệu suất của nó bằng cách sử dụng điểm Rouge. Chúng tôi sẽ tạo bản tóm tắt cho một số bài viết thử nghiệm và so sánh chúng với bản tóm tắt tham khảo.

```python
from transformers import pipeline

# Load the fine-tuned model
summarizer = pipeline("summarization", model=peft_model, tokenizer=tokenizer)

# Test data
test_article = """
The United Nations has warned that the world is facing its largest humanitarian crisis since 1945.
More than 20 million people in four countries are at risk of starvation and famine.
The UN humanitarian chief Stephen O'Brien has called for an urgent mobilization of funds to
prevent a catastrophe. The countries most at risk are Yemen, South Sudan, Somalia and Nigeria.
Ongoing conflicts in these regions have exacerbated food shortages and economic crises.
"""

# Generate summary
generated_summary = summarizer(test_article, max_length=50, min_length=10, do_sample=False)[0]['summary_text']

# Reference summary (human-written)
reference_summary = "UN warns of largest humanitarian crisis since 1945 with over 20 million at risk of starvation in four countries due to conflicts and economic issues."

# Calculate Rouge scores
scores = calculate_rouge(generated_summary, reference_summary)

print("Generated Summary:", generated_summary)
print("\nReference Summary:", reference_summary)
print("\nRouge Scores:")
for metric, score in scores.items():
    print(f"{metric}: {score.fmeasure:.4f}")
```

Trang trình bày 11: Ví dụ thực tế: Tinh chỉnh phân tích cảm xúc

Hãy cùng khám phá một ứng dụng thực tế khác: tinh chỉnh mô hình phân tích cảm tính khi đánh giá sản phẩm. Chúng tôi sẽ sử dụng tập dữ liệu về các đánh giá sản phẩm của Amazon để minh họa quy trình.

```python
from datasets import load_dataset

# Load a sentiment analysis dataset
dataset = load_dataset("amazon_reviews_multi", "en", split="train[:1000]")

# Prepare the data
def preprocess_function(examples):
    return tokenizer(examples["review_body"], truncation=True, padding="max_length")

tokenized_dataset = dataset.map(preprocess_function, batched=True)
tokenized_dataset = tokenized_dataset.rename_column("stars", "labels")
tokenized_dataset = tokenized_dataset.remove_columns(["review_id", "product_id", "reviewer_id", "review_title", "product_category", "review_body", "language"])

# Fine-tune the model
trainer = Trainer(
    model=peft_model,
    args=TrainingArguments(output_dir="./sentiment_model", num_train_epochs=3),
    train_dataset=tokenized_dataset,
)

trainer.train()
print("Fine-tuning for sentiment analysis completed!")
```

Slide 12: Đánh giá mô hình phân tích cảm xúc

Bây giờ chúng ta đã tinh chỉnh mô hình để phân tích cảm tính, hãy đánh giá hiệu suất của nó qua một số bài đánh giá thử nghiệm. Chúng tôi sẽ sử dụng mô hình đã tinh chỉnh để dự đoán cảm tính và so sánh chúng với xếp hạng thực tế.

```python
from transformers import pipeline

# Load the fine-tuned model
sentiment_analyzer = pipeline("sentiment-analysis", model=peft_model, tokenizer=tokenizer)

# Test data
test_reviews = [
    "This product exceeded my expectations. It's durable and works perfectly!",
    "I'm disappointed with the quality. It broke after just a week of use.",
    "Average product. Does the job, but nothing special."
]

# Predict sentiments
for review in test_reviews:
    result = sentiment_analyzer(review)[0]
    sentiment = "Positive" if result['label'] == "LABEL_1" else "Negative"
    confidence = result['score']
    print(f"Review: {review}")
    print(f"Predicted Sentiment: {sentiment} (Confidence: {confidence:.2f})")
    print()

# Note: In a real scenario, you would compare these predictions with actual ratings
# and calculate metrics like accuracy, precision, recall, and F1-score.
```

Slide 13: Những thách thức và cân nhắc trong việc tinh chỉnh

Mặc dù việc tinh chỉnh LLM bằng các kỹ thuật PEFT như LoRA có thể hiệu quả nhưng điều quan trọng là bạn phải nhận thức được những thách thức tiềm ẩn:

1. Trang bị quá mức: Các mô hình được tinh chỉnh có thể hoạt động tốt trên dữ liệu huấn luyện nhưng không thể khái quát hóa thành dữ liệu mới, chưa được nhìn thấy.
2. Sự quên lãng nghiêm trọng: Mô hình có thể mất đi một số kiến ​​thức chung trong khi thích ứng với một nhiệm vụ cụ thể.
3. Khuếch đại sai lệch: Việc tinh chỉnh các bộ dữ liệu sai lệch có thể làm trầm trọng thêm các sai lệch hiện có trong mô hình.

Để giải quyết những thách thức này, hãy xem xét những điều sau:

```python
from transformers import EarlyStoppingCallback

# Example: Using early stopping to prevent overfitting
early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=3)

trainer = Trainer(
    model=peft_model,
    args=TrainingArguments(
        output_dir="./robust_model",
        num_train_epochs=10,
        evaluation_strategy="steps",
        eval_steps=100,
        load_best_model_at_end=True,
    ),
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset.select(range(100)),  # Small validation set
    callbacks=[early_stopping_callback],
)

trainer.train()

print("Robust fine-tuning completed!")
```

Slide 14: Kết luận và định hướng tương lai

Tinh chỉnh các mô hình ngôn ngữ lớn bằng cách sử dụng các kỹ thuật PEFT như LoRA mang lại một cách mạnh mẽ để điều chỉnh các mô hình được đào tạo trước cho phù hợp với các nhiệm vụ cụ thể trong khi vẫn duy trì hiệu quả. Điểm Rouge cung cấp thước đo có giá trị để đánh giá chất lượng văn bản được tạo ra, đặc biệt trong các tác vụ như tóm tắt.

Khi lĩnh vực NLP tiếp tục phát triển, chúng ta có thể mong đợi thấy:

1. Các kỹ thuật PEFT tiên tiến hơn giúp giảm hơn nữa các yêu cầu tính toán khi tinh chỉnh.
2. Các số liệu đánh giá được cải thiện nhằm nắm bắt nhiều khía cạnh sắc thái hơn về chất lượng và mức độ liên quan của văn bản.
3. Các kỹ thuật để giải quyết những thách thức như thành kiến ​​và đảm bảo việc sử dụng các mô hình đã được tinh chỉnh một cách có đạo đức.

Bằng cách kết hợp những tiến bộ này với việc xem xét cẩn thận các thách thức và phương pháp hay nhất, chúng tôi có thể tiếp tục vượt qua ranh giới về những gì có thể làm được với các mô hình ngôn ngữ lớn.

Trang trình bày 15: Tài nguyên bổ sung

Đối với những người muốn tìm hiểu sâu hơn về các chủ đề được trình bày trong bài trình bày này, đây là một số tài nguyên có giá trị:

1. LoRA: Thích ứng cấp thấp của các mô hình ngôn ngữ lớn ArXiv: [https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)
2. ROUGE: Gói đánh giá tự động các bản tóm tắt ArXiv: [https://arxiv.org/abs/1803.01937](https://arxiv.org/abs/1803.01937)
3. Học chuyển giao tham số hiệu quả cho NLP ArXiv: [https://arxiv.org/abs/1902.00751](https://arxiv.org/abs/1902.00751)
4. Thu nhỏ để mở rộng quy mô: Hướng dẫn về ArXiv tinh chỉnh tham số hiệu quả: [https://arxiv](https://arxiv).
