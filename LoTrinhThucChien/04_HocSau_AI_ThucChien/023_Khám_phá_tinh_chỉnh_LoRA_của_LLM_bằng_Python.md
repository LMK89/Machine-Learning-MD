## Khám phá Tinh chỉnh LoRA của LLM bằng Python
Slide 1: Giới thiệu về LoRA Finetuning

LoRA (Thích ứng cấp thấp) là một kỹ thuật để tinh chỉnh hiệu quả các mô hình ngôn ngữ lớn (LLM) với tài nguyên tính toán tối thiểu. Nó hoạt động bằng cách thêm các ma trận nhỏ, có thể huấn luyện vào các lớp chú ý của mô hình, cho phép điều chỉnh theo nhiệm vụ cụ thể mà không cần sửa đổi toàn bộ mô hình. Cách tiếp cận này làm giảm đáng kể số lượng tham số có thể huấn luyện và yêu cầu bộ nhớ.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

model = AutoModelForCausalLM.from_pretrained("gpt2")
lora_config = LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, lora_config)
```

Trang trình bày 2: Tìm hiểu kiến ​​trúc LoRA

LoRA giới thiệu các ma trận phân rã cấp thấp (A và B) cho các lớp chú ý của mô hình được đào tạo trước. Các ma trận này được khởi tạo ngẫu nhiên và được huấn luyện theo nhiệm vụ cụ thể. Ma trận trọng số ban đầu W bị cố định và việc điều chỉnh được thực hiện thông qua các ma trận cấp thấp: W + AB^T. Cách tiếp cận này cho phép tinh chỉnh hiệu quả với việc cập nhật tham số tối thiểu.

```python
class LoRALayer(torch.nn.Module):
    def __init__(self, in_features, out_features, rank=4):
        super().__init__()
        self.A = torch.nn.Parameter(torch.randn(in_features, rank))
        self.B = torch.nn.Parameter(torch.randn(rank, out_features))
        self.W = torch.nn.Linear(in_features, out_features)
        self.W.weight.requires_grad = False  # Freeze original weights

    def forward(self, x):
        return self.W(x) + torch.matmul(torch.matmul(x, self.A), self.B)
```

Slide 3: Thiết lập môi trường

Để bắt đầu tinh chỉnh LoRA, chúng ta cần thiết lập môi trường Python với các thư viện cần thiết. Chúng tôi sẽ sử dụng thư viện Transformers của Hugging Face cùng với thư viện PEFT (Tinh chỉnh hiệu quả tham số), triển khai LoRA.

```python
!pip install transformers peft datasets torch
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
```

Slide 4: Preparing the Dataset

For this example, we'll use a simple text classification dataset. We'll load it using the Hugging Face datasets library and preprocess it for our model.

```python
dataset = load_dataset("imdb", split="train[:1000]")

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenized_dataset = dataset.map(preprocess_function, batched=True)
```

Slide 5: Khởi tạo Model với LoRA

Chúng tôi sẽ bắt đầu với mô hình GPT-2 được đào tạo trước và áp dụng LoRA cho mô hình đó. Điều này liên quan đến việc định cấu hình các tham số LoRA và gói mô hình của chúng tôi bằng thư viện PEFT.

```python
model = AutoModelForCausalLM.from_pretrained("gpt2")

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["c_attn"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

Slide 6: Xác định đối số đào tạo

Trước khi bắt đầu đào tạo, chúng ta cần thiết lập các đối số đào tạo của mình. Các tham số này kiểm soát các khía cạnh khác nhau của quá trình đào tạo, chẳng hạn như tốc độ học tập, kích thước lô và số lượng kỷ nguyên.

```python
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)
```

Slide 7: Đào tạo người mẫu

Bây giờ chúng ta đã thiết lập xong mô hình, tập dữ liệu và đối số đào tạo, chúng ta có thể bắt đầu quá trình tinh chỉnh. Lớp Trainer xử lý vòng lặp đào tạo cho chúng ta.

```python
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./lora_finetuned_model")
```

Slide 8: Suy luận với mô hình tinh chỉnh

Sau khi đào tạo, chúng ta có thể sử dụng mô hình đã tinh chỉnh của mình để suy luận. Đây là cách tải mô hình và tạo văn bản dựa trên lời nhắc.

```python
from peft import PeftModel, PeftConfig

config = PeftConfig.from_pretrained("./lora_finetuned_model")
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(model, "./lora_finetuned_model")

tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

prompt = "This movie was"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

outputs = model.generate(input_ids=input_ids, max_length=50, num_return_sequences=1)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

Trang trình chiếu 9: Ví dụ thực tế: Phân tích cảm xúc

Hãy áp dụng mô hình được tinh chỉnh LoRA của chúng tôi cho một nhiệm vụ trong thế giới thực: phân tích tình cảm. Chúng tôi sẽ sử dụng thông tin này để phân loại các bài đánh giá phim là tích cực hay tiêu cực.

```python
def classify_sentiment(review):
    prompt = f"Classify the sentiment of this movie review: '{review}'\nSentiment:"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    outputs = model.generate(input_ids=input_ids, max_length=len(prompt) + 10)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).split("Sentiment:")[-1].strip()

reviews = [
    "This movie was absolutely fantastic! I loved every minute of it.",
    "I was disappointed by this film. The plot was confusing and the acting was subpar."
]

for review in reviews:
    sentiment = classify_sentiment(review)
    print(f"Review: {review}\nSentiment: {sentiment}\n")
```

Slide 10: Ví dụ thực tế: Tóm tắt văn bản

Một ứng dụng thực tế khác của mô hình được tinh chỉnh LoRA của chúng tôi là tóm tắt văn bản. Chúng ta có thể sử dụng nó để tạo ra những bản tóm tắt ngắn gọn cho những văn bản dài hơn.

```python
def summarize_text(text):
    prompt = f"Summarize the following text:\n{text}\n\nSummary:"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    outputs = model.generate(input_ids=input_ids, max_length=len(prompt) + 100, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).split("Summary:")[-1].strip()

long_text = """
The Internet of Things (IoT) is transforming the way we live and work.
It refers to the interconnected network of physical devices, vehicles,
home appliances, and other items embedded with electronics, software,
sensors, and network connectivity, which enables these objects to collect
and exchange data. The IoT has applications in various fields, including
smart homes, healthcare, agriculture, and industrial automation.
"""

summary = summarize_text(long_text)
print(f"Original text:\n{long_text}\n\nSummary:\n{summary}")
```

Trang trình bày 11: Điều chỉnh siêu tham số cho LoRA

Tối ưu hóa siêu tham số LoRA có thể tác động đáng kể đến hiệu suất của mô hình. Các tham số chính bao gồm các mô-đun xếp hạng (r), alpha và đích. Đây là ví dụ về cách thực hiện tìm kiếm lưới đơn giản cho các tham số này.

```python
import itertools

def train_and_evaluate(r, alpha, target_modules):
    lora_config = LoraConfig(r=r, lora_alpha=alpha, target_modules=target_modules)
    model = get_peft_model(AutoModelForCausalLM.from_pretrained("gpt2"), lora_config)

    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset)
    trainer.train()

    # Evaluate the model (you need to implement your own evaluation metric)
    return evaluate_model(model)

r_values = [4, 8, 16]
alpha_values = [16, 32, 64]
target_modules = [["c_attn"], ["c_attn", "c_proj"]]

best_score = float('-inf')
best_params = None

for r, alpha, modules in itertools.product(r_values, alpha_values, target_modules):
    score = train_and_evaluate(r, alpha, modules)
    if score > best_score:
        best_score = score
        best_params = (r, alpha, modules)

print(f"Best parameters: r={best_params[0]}, alpha={best_params[1]}, target_modules={best_params[2]}")
```

Trang trình bày 12: Hình dung tác động của LoRA

Để hiểu rõ hơn về cách LoRA ảnh hưởng đến mô hình, chúng ta có thể hình dung các mẫu chú ý trước và sau khi tinh chỉnh. Điều này có thể cung cấp thông tin chi tiết về cách thay đổi trọng tâm của mô hình đối với các nhiệm vụ cụ thể.

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_attention(model, text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs, output_attentions=True)
    attention = outputs.attentions[-1].squeeze().detach().numpy()

    plt.figure(figsize=(10, 8))
    sns.heatmap(attention, cmap="YlOrRd")
    plt.title("Attention Pattern")
    plt.xlabel("Token Position (Key)")
    plt.ylabel("Token Position (Query)")
    plt.show()

text = "The quick brown fox jumps over the lazy dog."

print("Attention pattern before LoRA finetuning:")
plot_attention(AutoModelForCausalLM.from_pretrained("gpt2"), text)

print("\nAttention pattern after LoRA finetuning:")
plot_attention(model, text)
```

Trang trình bày 13: Hợp nhất các trọng số LoRA

Sau khi tinh chỉnh, chúng ta có thể hợp nhất các trọng số LoRA với mô hình cơ sở để suy luận hiệu quả. Bước này kết hợp các trọng số mô hình ban đầu với các điều chỉnh LoRA đã học.

```python
from peft import PeftModel

# Load the base model and LoRA weights
base_model = AutoModelForCausalLM.from_pretrained("gpt2")
peft_model = PeftModel.from_pretrained(base_model, "./lora_finetuned_model")

# Merge weights
merged_model = peft_model.merge_and_unload()

# Save the merged model
merged_model.save_pretrained("./merged_model")

# Now you can use the merged model for inference without LoRA overhead
merged_model = AutoModelForCausalLM.from_pretrained("./merged_model")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

prompt = "The future of AI is"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
outputs = merged_model.generate(input_ids=input_ids, max_length=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

Trang trình bày 14: Tài nguyên bổ sung

Đối với những người quan tâm đến việc tìm hiểu sâu hơn về LoRA và các kỹ thuật tinh chỉnh hiệu quả, đây là một số tài nguyên có giá trị:

1. Bài viết gốc của LoRA: "LoRA: Sự thích ứng ở cấp độ thấp của các mô hình ngôn ngữ lớn" (arXiv:2106.09685)
2. Tài liệu thư viện PEFT ôm mặt: [https://huggingface.co/docs/peft/index](https://huggingface.co/docs/peft/index)
3. "Học chuyển giao tham số hiệu quả cho NLP" (arXiv:1902.00751)
4. "Thu nhỏ để tăng quy mô: Hướng dẫn tinh chỉnh tham số hiệu quả" (arXiv:2303.15647)

Các tài nguyên này cung cấp những giải thích sâu sắc về LoRA và các kỹ thuật liên quan, cũng như các ứng dụng của chúng trong các tác vụ xử lý ngôn ngữ tự nhiên khác nhau.
