##So sánh các mô hình CNN 2 lớp trên CIFAR-10
Slide 1: So sánh mô hình: Kiến trúc CNN

Hai mô hình CNN 2 lớp được huấn luyện trên bộ dữ liệu CIFAR-10 cho kết quả có độ chính xác khác nhau. Mô hình A đạt độ chính xác 70%, trong khi Mô hình B đạt 74%. Sự khác biệt này không phải do điều chỉnh siêu tham số, cho thấy rằng có các yếu tố khác đang tác động. Hãy cùng khám phá những lý do có thể dẫn đến khoảng cách hiệu suất này.

```python
import tensorflow as tf
from tensorflow.keras import layers, models

def create_cnn_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

model_a = create_cnn_model()
model_b = create_cnn_model()

# Train and evaluate models
# ...

print(f"Model A accuracy: {model_a_accuracy:.2f}")
print(f"Model B accuracy: {model_b_accuracy:.2f}")
```

Slide 2: Các yếu tố ảnh hưởng đến hiệu suất của mô hình

Một số yếu tố có thể góp phần tạo ra sự khác biệt về hiệu suất giữa hai mô hình CNN có vẻ giống hệt nhau. Chúng bao gồm việc khởi tạo trọng số, xáo trộn dữ liệu và các biến thể nhỏ trong quá trình huấn luyện. Ngay cả với cùng một kiến ​​trúc, những yếu tố này có thể dẫn đến sự tối ưu cục bộ khác nhau trong quá trình đào tạo.

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_loss_curves(model_a_history, model_b_history):
    plt.figure(figsize=(10, 6))
    plt.plot(model_a_history.history['loss'], label='Model A Loss')
    plt.plot(model_b_history.history['loss'], label='Model B Loss')
    plt.title('Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Simulate training histories
np.random.seed(42)
epochs = 50
model_a_history = {'loss': np.random.rand(epochs) * 0.5 + 0.5}
model_b_history = {'loss': np.random.rand(epochs) * 0.4 + 0.3}

plot_loss_curves(model_a_history, model_b_history)
```

Slide 3: Triển khai mô hình hiệu quả

Để đảm bảo triển khai hiệu quả các mô hình ML trong sản xuất, hai phương pháp chính thường được sử dụng: đào tạo các mô hình nhỏ từ đầu hoặc sử dụng chắt lọc kiến ​​thức để chuyển kiến ​​thức từ mô hình lớn hơn sang mô hình nhỏ hơn. Cả hai phương pháp đều nhằm mục đích giảm yêu cầu tính toán và sử dụng bộ nhớ trong môi trường sản xuất.

```python
def small_model():
    return models.Sequential([
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

small_model = small_model()
print(f"Small model parameter count: {small_model.count_params():,}")
```

Slide 4: Chắt lọc kiến ​​thức (KD)

Chắt lọc kiến ​​thức là một kỹ thuật trong đó một mô hình nhỏ hơn, đơn giản hơn (học sinh) được đào tạo để bắt chước đầu ra của một mô hình lớn hơn, phức tạp hơn (giáo viên). Quá trình này cho phép mô hình học sinh được hưởng lợi từ kiến ​​thức mà mô hình giáo viên thu được trong khi vẫn duy trì kích thước nhỏ hơn và yêu cầu tính toán thấp hơn.

```python
import tensorflow as tf

def knowledge_distillation_loss(y_true, y_pred, teacher_pred, temperature=2.0):
    soft_targets = tf.nn.softmax(teacher_pred / temperature)
    soft_prob = tf.nn.softmax(y_pred / temperature)
    return tf.keras.losses.categorical_crossentropy(soft_targets, soft_prob) * (temperature ** 2)

# Example usage
teacher_model = create_cnn_model()
student_model = small_model()

# Train student model using KD loss
# ...
```

Trang trình bày 5: DistilBERT: Một ví dụ thực tế

DistilBERT là một ví dụ đáng chú ý về việc chắt lọc kiến ​​thức trong xử lý ngôn ngữ tự nhiên. Đây là phiên bản nhỏ hơn của mô hình BERT, giữ lại khoảng 97% khả năng của BERT trong khi nhỏ hơn 40%. Việc giảm kích thước đáng kể này làm cho DistilBERT phù hợp hơn để triển khai trong môi trường hạn chế về tài nguyên.

```python
from transformers import DistilBertTokenizer, DistilBertModel
import torch

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

text = "Knowledge distillation helps create efficient models."
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)

print(f"Output shape: {outputs.last_hidden_state.shape}")
print(f"Model size: {sum(p.numel() for p in model.parameters()):,} parameters")
```

Slide 6: Hạn chế của việc chắt lọc kiến ​​thức

Trong thực tế, việc chắt lọc kiến ​​thức có một số hạn chế. Có giới hạn về mức độ mà mô hình học sinh có thể học được từ mô hình giáo viên ở quy mô nhất định. Ngoài ra, đối với một mô hình giáo viên nhất định, có một quy mô tối thiểu cho mô hình học sinh mà dưới đây việc chuyển giao kiến ​​thức hiệu quả sẽ trở thành một thách thức.

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_kd_effectiveness(teacher_sizes, student_sizes, effectiveness):
    plt.figure(figsize=(10, 6))
    plt.imshow(effectiveness, cmap='viridis', aspect='auto')
    plt.colorbar(label='KD Effectiveness')
    plt.xlabel('Student Model Size')
    plt.ylabel('Teacher Model Size')
    plt.title('Knowledge Distillation Effectiveness')
    plt.xticks(range(len(student_sizes)), student_sizes)
    plt.yticks(range(len(teacher_sizes)), teacher_sizes)
    plt.show()

teacher_sizes = [1e6, 5e6, 1e7, 5e7, 1e8]
student_sizes = [1e5, 5e5, 1e6, 5e6, 1e7]
effectiveness = np.random.rand(len(teacher_sizes), len(student_sizes))

plot_kd_effectiveness(teacher_sizes, student_sizes, effectiveness)
```

Slide 7: Phương pháp tiếp cận trợ giảng giáo viên

Để giải quyết những hạn chế của việc chắt lọc kiến ​​thức trực tiếp, có thể đưa ra một mô hình trung gian gọi là “trợ giảng”. Cách tiếp cận này bao gồm một quy trình gồm hai bước: đầu tiên, mô hình trợ lý học từ mô hình giáo viên và sau đó là mô hình học sinh học từ mô hình trợ lý.

```python
def create_teacher_model():
    return models.Sequential([
        layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

def create_assistant_model():
    return models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

teacher_model = create_teacher_model()
assistant_model = create_assistant_model()
student_model = small_model()

# Implement two-step KD process
# ...
```

Trang trình bày 8: Lợi ích của phương pháp trợ lý giáo viên

Phương pháp trợ lý giáo viên có thể nâng cao đáng kể hiệu suất và hiệu quả của mô hình học sinh cuối cùng. Mặc dù nó bổ sung thêm một bước đào tạo nhưng lợi ích thường lớn hơn chi phí tính toán bổ sung, đặc biệt là trong môi trường sản xuất nơi hiệu quả của mô hình là rất quan trọng.

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_model_comparison(models, accuracies):
    plt.figure(figsize=(10, 6))
    plt.bar(models, accuracies)
    plt.title('Model Accuracy Comparison')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.01, f'{v:.2f}', ha='center')
    plt.show()

models = ['Teacher', 'Assistant', 'Student (Direct KD)', 'Student (TA KD)']
accuracies = [0.95, 0.92, 0.88, 0.91]

plot_model_comparison(models, accuracies)
```

Trang trình bày 9: Ví dụ thực tế: Phân loại hình ảnh

Hãy xem xét một nhiệm vụ phân loại hình ảnh để xác định các loại trái cây khác nhau. Chúng tôi sẽ sử dụng MobileNetV2 được đào tạo trước làm mô hình giáo viên và tạo CNN tùy chỉnh nhỏ hơn làm mô hình học sinh. Mục tiêu là đạt được hiệu suất tương đương với kích thước mô hình nhỏ hơn nhiều.

```python
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Teacher model (pre-trained MobileNetV2)
teacher_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
teacher_model = tf.keras.Sequential([
    teacher_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(5, activation='softmax')
])

# Student model (custom CNN)
student_model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(5, activation='softmax')
])

# Data preparation
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = datagen.flow_from_directory(
    'path/to/fruit/dataset',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

# Knowledge distillation training
# ...

print(f"Teacher model size: {teacher_model.count_params():,} parameters")
print(f"Student model size: {student_model.count_params():,} parameters")
```

Trang trình bày 10: Ví dụ thực tế: Phân loại văn bản

Trong ví dụ này, chúng tôi sẽ sử dụng BERT làm mô hình giáo viên và mô hình dựa trên LSTM đơn giản hơn làm sinh viên để phân tích cảm tính về các bài đánh giá phim. Mục tiêu là tạo ra một mô hình nhẹ hơn phù hợp để triển khai trên thiết bị di động trong khi vẫn duy trì hiệu suất tốt.

```python
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf

# Teacher model (BERT)
teacher_model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Student model (LSTM-based)
max_length = 128
vocab_size = 10000

student_model = tf.keras.Sequential([
    layers.Embedding(vocab_size, 100, input_length=max_length),
    layers.LSTM(64, return_sequences=True),
    layers.LSTM(32),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Prepare data
# ...

# Knowledge distillation training
# ...

print(f"Teacher model size: {teacher_model.count_params():,} parameters")
print(f"Student model size: {student_model.count_params():,} parameters")
```

Slide 11: Đánh giá chắt lọc kiến ​​thức

Để đánh giá hiệu quả chắt lọc kiến ​​thức, chúng ta cần so sánh hiệu quả thực hiện của mô hình sinh viên được đào tạo có và không có KD. Chúng tôi sẽ sử dụng các số liệu như độ chính xác, thời gian suy luận và kích thước mô hình để đánh giá sự cân bằng giữa hiệu suất và hiệu quả.

```python
import time

def evaluate_model(model, test_data, test_labels):
    start_time = time.time()
    predictions = model.predict(test_data)
    inference_time = time.time() - start_time

    accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(test_labels, axis=1))
    model_size = model.count_params()

    return accuracy, inference_time, model_size

# Evaluate teacher model
teacher_accuracy, teacher_time, teacher_size = evaluate_model(teacher_model, test_data, test_labels)

# Evaluate student model (without KD)
student_accuracy, student_time, student_size = evaluate_model(student_model, test_data, test_labels)

# Evaluate student model (with KD)
student_kd_accuracy, student_kd_time, student_kd_size = evaluate_model(student_model_kd, test_data, test_labels)

# Plot results
# ...
```

Slide 12: Những thách thức và cân nhắc

Mặc dù việc chắt lọc kiến ​​thức có thể mang lại hiệu quả cao nhưng vẫn có những thách thức cần xem xét. Chúng bao gồm việc chọn cặp mô hình giáo viên-học sinh phù hợp, xác định nhiệt độ tối ưu để làm giảm phân bố xác suất và cân bằng sự đánh đổi giữa kích thước mô hình và hiệu suất. Điều quan trọng là phải đánh giá cẩn thận các yếu tố này cho từng trường hợp sử dụng cụ thể.

```python
def plot_size_performance_tradeoff(models, sizes, accuracies):
    plt.figure(figsize=(10, 6))
    plt.scatter(sizes, accuracies)
    for i, model in enumerate(models):
        plt.annotate(model, (sizes[i], accuracies[i]))
    plt.xlabel('Model Size (parameters)')
    plt.ylabel('Accuracy')
    plt.title('Model Size vs. Performance Trade-off')
    plt.xscale('log')
    plt.grid(True)
    plt.show()

models = ['Teacher', 'Student (No KD)', 'Student (KD)', 'Student (TA KD)']
sizes = [1e8, 1e6, 1e6, 1e6]
accuracies = [0.95, 0.85, 0.89, 0.91]

plot_size_performance_tradeoff(models, sizes, accuracies)
```

Slide 13: Định hướng và nghiên cứu trong tương lai

Chắt lọc kiến ​​thức tiếp tục là một lĩnh vực nghiên cứu tích cực. Các hướng đi trong tương lai bao gồm khám phá sự chắt lọc nhiều giáo viên, phát triển các biểu diễn trung gian hiệu quả hơn và nghiên cứu nền tảng lý thuyết của việc chuyển giao kiến ​​thức. Những tiến bộ này có thể dẫn đến những mô hình thậm chí còn hiệu quả và mạnh mẽ hơn trong tương lai.

```python
def plot_research_trends():
    years = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
    kd_papers = [10, 25, 50, 100, 200, 350, 500, 700, 900]

    plt.figure(figsize=(10, 6))
    plt.plot(years, kd_papers, marker='o')
    plt.title('Knowledge Distillation Research Trend')
    plt.xlabel('Year')
    plt.ylabel('Number of Published Papers')
    plt.grid(True)
    plt.show()

plot_research_trends()
```

Trang trình bày 14: Tài nguyên bổ sung

Đối với những người quan tâm đến việc tìm hiểu sâu hơn về kỹ thuật chắt lọc kiến ​​thức và nén mô hình, đây là một số tài nguyên có giá trị:

1. "Chắt lọc kiến ​​thức trong mạng lưới thần kinh" của Hinton và cộng sự. (2015) ArXiv: [https://arxiv.org/abs/1503.02531](https://arxiv.org/abs/1503.02531)
2. "TinyBERT: Chắt lọc BERT để hiểu ngôn ngữ tự nhiên" của Jiao et al. (2020) ArXiv: [https://arxiv.org/abs/1909.10351](https://arxiv.org/abs/1909.10351)
3. "Chắt lọc kiến thức: Khảo sát" của Gou và cộng sự. (2021) ArXiv: [https://arxiv.org/abs/2006.05525](https://arxiv.org/abs/2006.05525)
4. "Mạng lưới thần kinh tái sinh" của Furlanello và cộng sự. (2018) ArXiv: [https://arxiv.org/abs/1805.04770](https://arxiv.org/abs/1805.04770)
5. "Chưng cất kiến thức không có dữ liệu cho mạng lưới thần kinh sâu" của Lopes et al. (2017) ArXiv: [https://arxiv.org/abs/1710.07535](https://arxiv.org/abs/1710.07535)

Những bài viết này cung cấp một cái nhìn tổng quan toàn diện về các kỹ thuật chắt lọc kiến ​​thức, ứng dụng của chúng và những tiến bộ gần đây trong lĩnh vực này. Chúng bao gồm cả nền tảng lý thuyết và cách triển khai thực tế, khiến chúng trở thành điểm khởi đầu tuyệt vời cho các nhà nghiên cứu cũng như những người thực hành.