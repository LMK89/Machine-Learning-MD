## Trực quan hóa dữ liệu nhiều chiều bằng UMAP trong Python
Trang trình bày 1: Giới thiệu về Mô hình ngôn ngữ lớn (LLM)

Mô hình ngôn ngữ lớn là hệ thống AI được đào tạo trên lượng lớn dữ liệu văn bản để hiểu và tạo ra văn bản giống con người. Chúng tạo thành nền tảng cho nhiều nhiệm vụ xử lý ngôn ngữ tự nhiên hiện đại.

```python
import transformers

# Load a pre-trained LLM
model_name = "gpt2"
model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

# Generate text
input_text = "The future of AI is"
input_ids = tokenizer.encode(input_text, return_tensors="pt")
output = model.generate(input_ids, max_length=50, num_return_sequences=1)

print(tokenizer.decode(output[0], skip_special_tokens=True))
```

Slide 2: Đào tạo các mô hình ngôn ngữ lớn

LLM thường được đào tạo bằng cách sử dụng phương pháp học tập không giám sát trên một lượng lớn dữ liệu văn bản. Quá trình đào tạo bao gồm việc dự đoán từ tiếp theo trong một chuỗi, cho phép mô hình tìm hiểu các mẫu và mối quan hệ trong ngôn ngữ.

```python
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

# Load pre-trained model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Prepare dataset
dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="path/to/text/file.txt",
    block_size=128
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

# Start training
trainer.train()
```

Trang trình bày 3: Hạn chế của LLM truyền thống

Mặc dù LLM truyền thống mạnh mẽ nhưng có những hạn chế như kiến ​​thức lỗi thời, không có khả năng truy cập thông tin bên ngoài và tiềm ẩn ảo giác hoặc thông tin không chính xác.

```python
import openai

openai.api_key = 'your-api-key'

def query_llm(prompt):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=100
    )
    return response.choices[0].text.strip()

# Example of a limitation: outdated information
prompt = "What is the current population of New York City?"
result = query_llm(prompt)
print(f"LLM Response: {result}")
print("Note: This information might be outdated or inaccurate.")
```

Trang trình bày 4: Giới thiệu về thế hệ tăng cường truy xuất (RAG)

RAG là một kỹ thuật kết hợp sức mạnh của LLM với khả năng truy xuất thông tin liên quan từ các nguồn bên ngoài, giải quyết một số hạn chế của LLM truyền thống.

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset

# Load model and tokenizer
model_name = "facebook/rag-token-nq"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Load dataset
dataset = load_dataset("nq_open", split="train[:100]")

# Function to generate answer
def generate_answer(question):
    inputs = tokenizer(question, return_tensors="pt")
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
question = dataset[0]["question"]
answer = generate_answer(question)
print(f"Question: {question}")
print(f"Generated Answer: {answer}")
```

Slide 5: Các thành phần của hệ thống RAG

Các hệ thống RAG thường bao gồm ba thành phần chính: bộ truy xuất, bộ tạo (LLM) và cơ chế tổng hợp kết hợp thông tin được truy xuất với đầu ra của LLM.

```python
import faiss
import numpy as np
from transformers import DPRQuestionEncoder, DPRContextEncoder

# Simplified RAG components

class Retriever:
    def __init__(self, context_encoder, passages):
        self.context_encoder = context_encoder
        self.passages = passages
        self.index = self._build_index()

    def _build_index(self):
        embeddings = self.context_encoder(self.passages)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        return index

    def retrieve(self, query, k=5):
        query_embedding = self.context_encoder([query])
        _, indices = self.index.search(query_embedding, k)
        return [self.passages[i] for i in indices[0]]

class Generator:
    def __init__(self, model):
        self.model = model

    def generate(self, query, retrieved_passages):
        context = " ".join(retrieved_passages)
        input_text = f"Query: {query}\nContext: {context}\nAnswer:"
        return self.model(input_text)

# Usage example (pseudo-code)
# retriever = Retriever(context_encoder, passages)
# generator = Generator(llm_model)
# query = "What is the capital of France?"
# retrieved_passages = retriever.retrieve(query)
# answer = generator.generate(query, retrieved_passages)
```

Slide 6: Triển khai RAG với Transformers ôm mặt

Ôm Mặt cung cấp các công cụ và mô hình để triển khai hệ thống RAG một cách dễ dàng. Đây là một ví dụ sử dụng triển khai RAG của họ.

```python
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
from datasets import load_dataset

# Load RAG components
model_name = "facebook/rag-token-nq"
tokenizer = RagTokenizer.from_pretrained(model_name)
retriever = RagRetriever.from_pretrained(model_name, index_name="exact", use_dummy_dataset=True)
model = RagSequenceForGeneration.from_pretrained(model_name, retriever=retriever)

# Load a sample dataset
dataset = load_dataset("nq_open", split="train[:5]")

# Function to generate answer using RAG
def generate_rag_answer(question):
    input_dict = tokenizer(question, return_tensors="pt")
    generated = model.generate(**input_dict)
    return tokenizer.decode(generated[0], skip_special_tokens=True)

# Example usage
for sample in dataset:
    question = sample["question"]
    answer = generate_rag_answer(question)
    print(f"Question: {question}")
    print(f"RAG Answer: {answer}\n")
```

Trang trình bày 7: Ưu điểm của RAG so với LLM truyền thống

Hệ thống RAG cung cấp một số lợi thế, bao gồm thông tin cập nhật, giảm ảo giác và khả năng trích dẫn nguồn thông tin được tạo ra.

```python
import random
from datetime import datetime

class TraditionalLLM:
    def generate(self, prompt):
        return "Generated response based on training data up to 2022."

class RAGSystem:
    def __init__(self):
        self.knowledge_base = {
            "AI advancements": "Latest AI models achieve human-level performance in various tasks.",
            "Climate change": "Global temperature rise of 1.1°C observed since pre-industrial times.",
            "COVID-19": "New variants continue to emerge, highlighting the importance of vaccination."
        }

    def retrieve(self, query):
        return random.choice(list(self.knowledge_base.values()))

    def generate(self, prompt):
        retrieved_info = self.retrieve(prompt)
        current_date = datetime.now().strftime("%Y-%m-%d")
        return f"As of {current_date}, {retrieved_info}"

# Compare traditional LLM and RAG system
llm = TraditionalLLM()
rag = RAGSystem()

prompt = "Tell me about recent developments in AI."
print(f"Traditional LLM: {llm.generate(prompt)}")
print(f"RAG System: {rag.generate(prompt)}")
```

Slide 8: Tinh chỉnh mô hình RAG

Tinh chỉnh cho phép các mô hình RAG thích ứng với các miền hoặc tác vụ cụ thể, cải thiện hiệu suất của chúng trên các ứng dụng được nhắm mục tiêu.

```python
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

# Load pre-trained RAG model
model_name = "facebook/rag-token-nq"
tokenizer = RagTokenizer.from_pretrained(model_name)
retriever = RagRetriever.from_pretrained(model_name, index_name="exact", use_dummy_dataset=True)
model = RagSequenceForGeneration.from_pretrained(model_name, retriever=retriever)

# Prepare dataset (example using a QA dataset)
dataset = load_dataset("squad", split="train[:1000]")

def preprocess_function(examples):
    inputs = tokenizer(examples["question"], truncation=True, padding="max_length")
    outputs = tokenizer(examples["answers"]["text"][0], truncation=True, padding="max_length")
    return {
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
        "labels": outputs.input_ids,
    }

processed_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./rag_finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=1000,
    save_total_limit=2,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset,
)

# Start fine-tuning
trainer.train()
```

Slide 9: Đánh giá hệ thống RAG

Đánh giá hệ thống RAG bao gồm việc đánh giá cả thành phần truy xuất và chất lượng tạo tổng thể. Các số liệu như ROUGE, BLEU và đánh giá con người thường được sử dụng.

```python
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
import nltk
nltk.download('punkt')

def evaluate_rag(rag_model, test_data):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = []
    bleu_scores = []

    for sample in test_data:
        question = sample['question']
        reference = sample['answer']

        # Generate answer using RAG model
        generated = rag_model.generate(question)

        # Calculate ROUGE scores
        rouge_score = scorer.score(reference, generated)
        rouge_scores.append(rouge_score)

        # Calculate BLEU score
        reference_tokens = nltk.word_tokenize(reference)
        generated_tokens = nltk.word_tokenize(generated)
        bleu_score = sentence_bleu([reference_tokens], generated_tokens)
        bleu_scores.append(bleu_score)

    # Calculate average scores
    avg_rouge = {key: sum(score[key].fmeasure for score in rouge_scores) / len(rouge_scores) for key in rouge_scores[0]}
    avg_bleu = sum(bleu_scores) / len(bleu_scores)

    return {
        "ROUGE": avg_rouge,
        "BLEU": avg_bleu
    }

# Example usage (pseudo-code)
# test_data = load_test_data()
# rag_model = load_rag_model()
# evaluation_results = evaluate_rag(rag_model, test_data)
# print(evaluation_results)
```

Slide 10: Giới thiệu về Agentic RAG

Agentic RAG mở rộng khái niệm RAG bằng cách kết hợp khả năng ra quyết định và lập kế hoạch nhiệm vụ tự động, cho phép hệ thống thực hiện các nhiệm vụ nhiều bước, phức tạp hơn.

```python
import random

class AgenticRAG:
    def __init__(self):
        self.knowledge_base = {
            "weather": "It's sunny today with a high of 25°C.",
            "schedule": "You have a meeting at 2 PM.",
            "email": "You have 3 unread emails."
        }

    def retrieve(self, query):
        return self.knowledge_base.get(query.lower(), "No information found.")

    def decide_action(self, user_input):
        if "weather" in user_input.lower():
            return "check_weather"
        elif "schedule" in user_input.lower():
            return "check_schedule"
        elif "email" in user_input.lower():
            return "check_email"
        else:
            return "ask_for_clarification"

    def execute_action(self, action):
        if action == "check_weather":
            return self.retrieve("weather")
        elif action == "check_schedule":
            return self.retrieve("schedule")
        elif action == "check_email":
            return self.retrieve("email")
        else:
            return "I'm not sure what you're asking. Can you please clarify?"

    def interact(self, user_input):
        action = self.decide_action(user_input)
        return self.execute_action(action)

# Example usage
agent = AgenticRAG()
user_queries = [
    "What's the weather like?",
    "Do I have any meetings today?",
    "Check my emails",
    "What's for lunch?"
]

for query in user_queries:
    response = agent.interact(query)
    print(f"User: {query}")
    print(f"Agent: {response}\n")
```

Slide 11: Các thành phần của hệ thống Agentic RAG

Các hệ thống RAG tác nhân thường bao gồm các thành phần lập kế hoạch, ra quyết định và thực hiện nhiệm vụ, bên cạnh các thành phần truy xuất và tạo của RAG truyền thống.

```python
import random

class Planner:
    def create_plan(self, goal):
        # Simplified planning logic
        steps = ["research", "analyze", "summarize"]
        return steps

class Retriever:
    def retrieve(self, query):
        # Simulated retrieval
        documents = [
            "Document about AI advancements.",
            "Paper on machine learning algorithms.",
            "Article on natural language processing."
        ]
        return random.choice(documents)

class Generator:
    def generate(self, context, query):
        # Simulated text generation
        return f"Generated response based on {context} and query: {query}"

class AgenticRAG:
    def __init__(self):
        self.planner = Planner()
        self.retriever = Retriever()
        self.generator = Generator()

    def execute_task(self, goal):
        plan = self.planner.create_plan(goal)
        result = ""
        for step in plan:
            retrieved_info = self.retriever.retrieve(step)
            result += self.generator.generate(retrieved_info, step) + " "
        return result.strip()

# Example usage
agentic_rag = AgenticRAG()
task_goal = "Explain recent advancements in AI"
result = agentic_rag.execute_task(task_goal)
print(f"Task: {task_goal}")
print(f"Result: {result}")
```

Slide 12: Ví dụ thực tế: Trợ lý cá nhân

Hệ thống Agentic RAG có thể được sử dụng để tạo ra một trợ lý cá nhân tiên tiến hơn có khả năng xử lý các nhiệm vụ phức tạp, gồm nhiều bước.

```python
import random
from datetime import datetime, timedelta

class PersonalAssistantRAG:
    def __init__(self):
        self.knowledge_base = {
            "weather": {"condition": "sunny", "temperature": 25},
            "calendar": [
                {"event": "Team Meeting", "time": "14:00"},
                {"event": "Dentist Appointment", "time": "10:00"}
            ],
            "tasks": ["Buy groceries", "Finish report", "Call mom"]
        }

    def retrieve(self, query):
        return self.knowledge_base.get(query, "No information found.")

    def plan_day(self):
        weather = self.retrieve("weather")
        calendar = self.retrieve("calendar")
        tasks = self.retrieve("tasks")

        plan = f"Today's weather: {weather['condition']}, {weather['temperature']}°C\n\n"
        plan += "Schedule:\n"
        for event in calendar:
            plan += f"- {event['time']}: {event['event']}\n"
        plan += "\nTasks:\n"
        for task in tasks:
            plan += f"- {task}\n"

        return plan

assistant = PersonalAssistantRAG()
daily_plan = assistant.plan_day()
print(daily_plan)
```

Trang trình chiếu 13: Ví dụ thực tế: Trợ lý nghiên cứu tự động

Hệ thống Agentic RAG có thể hỗ trợ các nhà nghiên cứu bằng cách tự động hóa việc đánh giá tài liệu và tóm tắt các phát hiện trên nhiều nguồn.

```python
class ResearchAssistantRAG:
    def __init__(self):
        self.knowledge_base = {
            "AI": ["Recent advancements in neural networks",
                   "Applications of machine learning in healthcare",
                   "Ethical considerations in AI development"],
            "Climate": ["Impact of greenhouse gases on global warming",
                        "Renewable energy technologies",
                        "Climate change mitigation strategies"]
        }

    def retrieve(self, topic):
        return self.knowledge_base.get(topic, [])

    def summarize(self, texts):
        # Simulated summarization
        return "Summary of key findings from multiple sources."

    def conduct_research(self, topic):
        relevant_texts = self.retrieve(topic)
        summary = self.summarize(relevant_texts)
        return f"Research on {topic}:\n{summary}"

assistant = ResearchAssistantRAG()
research_topic = "AI"
research_report = assistant.conduct_research(research_topic)
print(research_report)
```

Slide 14: Thách thức và định hướng tương lai

Các hệ thống RAG tác nhân phải đối mặt với những thách thức như duy trì sự gắn kết qua nhiều bước, xử lý sự mơ hồ và đảm bảo việc ra quyết định có đạo đức. Các hướng nghiên cứu trong tương lai bao gồm cải thiện các thuật toán lập kế hoạch, nâng cao độ chính xác khi truy xuất và phát triển các số liệu đánh giá mạnh mẽ hơn.

```python
import random

class FutureAgenticRAG:
    def __init__(self):
        self.knowledge_base = {"AI Ethics": "Principles for responsible AI development"}

    def retrieve(self, query):
        return self.knowledge_base.get(query, "No information found.")

    def generate(self, context):
        return f"Generated response based on: {context}"

    def ethical_check(self, action):
        ethics_guidelines = self.retrieve("AI Ethics")
        # Simulated ethical decision-making
        return random.choice([True, False])

    def execute_task(self, task):
        retrieved_info = self.retrieve(task)
        proposed_action = self.generate(retrieved_info)

        if self.ethical_check(proposed_action):
            return f"Executing: {proposed_action}"
        else:
            return "Action not taken due to ethical concerns."

future_rag = FutureAgenticRAG()
task = "Develop a new AI model"
result = future_rag.execute_task(task)
print(result)
```

Trang trình bày 15: Tài nguyên bổ sung

Để biết thêm thông tin về LLM, RAG và RAG đại lý, hãy xem xét khám phá các tài nguyên sau:

1. "Thế hệ tăng cường truy xuất cho các nhiệm vụ NLP chuyên sâu về tri thức" (Lewis và cộng sự, 2020) ArXiv: [https://arxiv.org/abs/2005.11401](https://arxiv.org/abs/2005.11401)
2. "Mô hình ngôn ngữ là những người học ít cơ hội" (Brown và cộng sự, 2020) ArXiv: [https://arxiv.org/abs/2005.14165](https://arxiv.org/abs/2005.14165)
3. "Chuỗi tư duy thúc đẩy suy luận trong các mô hình ngôn ngữ lớn" (Wei và cộng sự, 2022) ArXiv: [https://arxiv.org/abs/2201.11903](https://arxiv.org/abs/2201.11903)

Những bài viết này cung cấp những hiểu biết sâu sắc về sự phát triển và ứng dụng của các mô hình ngôn ngữ tiên tiến và các hệ thống tăng cường truy xuất.
