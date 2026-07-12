## Xu hướng giảm độ dốc ngẫu nhiên cho kiến trúc mạng thần kinh
Trang trình bày 1: Xu hướng giảm dần độ dốc ngẫu nhiên (SGD)

Stochastic gradient Descent là một cơ sở hóa học tối ưu hóa thuật toán trong máy học. Mặc dù nó có hiệu quả cao nhưng nó có thể gây ra sự thiên vị trong quá trình đào tạo. Sự kiện thiên nhiên này có thể ảnh hưởng đến hiệu suất và tính năng của màn hình. Hãy cùng khám phá bản chất của thiên vị này và ý nghĩa của nó.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = np.random.randn(100, 1)
y = 2 * X + 1 + np.random.randn(100, 1) * 0.1

# Plot the data
plt.scatter(X, y)
plt.title("Sample Data for Linear Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.show()
```

Trang trình bày 2: Tìm hiểu xu hướng SGD

Sự thiên vị của SGD bắt nguồn từ bản chất ngẫu nhiên của nó. Bằng cách cập nhật các tham số dựa trên các lô nhỏ thay vì toàn bộ dữ liệu, SGD tạo ra sự khác biệt trong quá trình cập nhật tham số. Phương pháp sai này có thể dẫn đến sai lệch trong mô hình cuối cùng, đặc biệt là quy mô lô nhỏ hoặc tỷ lệ học cao.

```python
def sgd_step(X, y, w, b, learning_rate):
    N = len(y)
    y_pred = np.dot(X, w) + b
    dw = (1/N) * np.dot(X.T, (y_pred - y))
    db = (1/N) * np.sum(y_pred - y)
    w -= learning_rate * dw
    b -= learning_rate * db
    return w, b

# Initialize parameters
w = np.random.randn(1, 1)
b = 0
learning_rate = 0.01

# Perform SGD steps
for _ in range(1000):
    w, b = sgd_step(X, y, w, b, learning_rate)

print(f"Learned parameters: w = {w[0][0]:.4f}, b = {b:.4f}")
```

Trang trình bày 3: Tác động của kích thước lô đến độ lệch

Kích thước lô tính bằng SGD ảnh bị ảnh hưởng đáng kể đến sự đánh đổi sai phương pháp. Lô kích thước nhỏ hơn tạo ra nhiều nhiễu loạn trong quá trình cập nhật tham số, có khả năng dẫn đến độ lệch cao hơn. Lô kích thước lớn hơn làm giảm nhiễu nhưng có thể làm chậm quá trình hội tụ.

```python
def train_sgd(X, y, batch_size, epochs):
    w = np.random.randn(1, 1)
    b = 0
    learning_rate = 0.01
    N = len(y)

    for _ in range(epochs):
        for i in range(0, N, batch_size):
            X_batch = X[i:i+batch_size]
            y_batch = y[i:i+batch_size]
            w, b = sgd_step(X_batch, y_batch, w, b, learning_rate)

    return w, b

batch_sizes = [1, 10, 50, 100]
results = []

for batch_size in batch_sizes:
    w, b = train_sgd(X, y, batch_size, epochs=100)
    results.append((batch_size, w[0][0], b))

for batch_size, w, b in results:
    print(f"Batch size: {batch_size}, w = {w:.4f}, b = {b:.4f}")
```

Trang trình bày 4: Tỷ lệ học tập và xu hướng

Tốc độ học bằng SGD cũng đóng một vai trò quan trọng trong việc xác định sai lệch. Tốc độ học cao có thể gây ra các tham số cập nhật quá mức, dẫn đến tốc độ tăng lên và quá trình tạo có thể không ổn định. Ngược lại, tốc độ học có thể dẫn đến chậm và mắc kẹt trong các giải pháp dưới mức tối ưu.

```python
def train_sgd_multi_lr(X, y, learning_rates):
    results = []
    for lr in learning_rates:
        w = np.random.randn(1, 1)
        b = 0

        for _ in range(1000):
            w, b = sgd_step(X, y, w, b, lr)

        results.append((lr, w[0][0], b))

    return results

learning_rates = [0.001, 0.01, 0.1, 1.0]
lr_results = train_sgd_multi_lr(X, y, learning_rates)

for lr, w, b in lr_results:
    print(f"Learning rate: {lr}, w = {w:.4f}, b = {b:.4f}")
```

Slide 5: Động lực để giảm sai lệch

Động lượng là một kỹ thuật được sử dụng để giảm độ lệch trong SGD bằng cách tích lũy đường trung bình động của độ dốc trong quá khứ. Điều này giúp quá trình cập nhật tham số diễn ra suôn sẻ và có thể dẫn đến sự hội tụ nhanh hơn và giảm độ lệch, đặc biệt là trong các tình huống có dữ liệu thưa thớt hoặc độ cong cao.

```python
def sgd_momentum_step(X, y, w, b, v_w, v_b, learning_rate, momentum):
    N = len(y)
    y_pred = np.dot(X, w) + b
    dw = (1/N) * np.dot(X.T, (y_pred - y))
    db = (1/N) * np.sum(y_pred - y)

    v_w = momentum * v_w + learning_rate * dw
    v_b = momentum * v_b + learning_rate * db

    w -= v_w
    b -= v_b

    return w, b, v_w, v_b

# Initialize parameters
w = np.random.randn(1, 1)
b = 0
v_w = np.zeros_like(w)
v_b = 0
learning_rate = 0.01
momentum = 0.9

# Perform SGD with momentum steps
for _ in range(1000):
    w, b, v_w, v_b = sgd_momentum_step(X, y, w, b, v_w, v_b, learning_rate, momentum)

print(f"Learned parameters with momentum: w = {w[0][0]:.4f}, b = {b:.4f}")
```

Trình bày 6: Tỷ lệ học ứng dụng

Các phương pháp học tốc độ thích hợp như Adam hoặc RMSprop có thể giúp giảm thiểu sai lệch bằng cách điều chỉnh tốc độ học cho từng tham số. Các phương pháp này có thể đặc biệt hiệu quả trong các vấn đề có độ dốc thưa hoặc khi xử lý các vật kính không cố định.

```python
def adam_step(X, y, w, b, m_w, m_b, v_w, v_b, t, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
    N = len(y)
    y_pred = np.dot(X, w) + b
    dw = (1/N) * np.dot(X.T, (y_pred - y))
    db = (1/N) * np.sum(y_pred - y)

    m_w = beta1 * m_w + (1 - beta1) * dw
    m_b = beta1 * m_b + (1 - beta1) * db
    v_w = beta2 * v_w + (1 - beta2) * (dw**2)
    v_b = beta2 * v_b + (1 - beta2) * (db**2)

    m_w_hat = m_w / (1 - beta1**t)
    m_b_hat = m_b / (1 - beta1**t)
    v_w_hat = v_w / (1 - beta2**t)
    v_b_hat = v_b / (1 - beta2**t)

    w -= learning_rate * m_w_hat / (np.sqrt(v_w_hat) + epsilon)
    b -= learning_rate * m_b_hat / (np.sqrt(v_b_hat) + epsilon)

    return w, b, m_w, m_b, v_w, v_b

# Initialize parameters for Adam
w = np.random.randn(1, 1)
b = 0
m_w, m_b, v_w, v_b = 0, 0, 0, 0
learning_rate = 0.01

# Perform Adam optimization steps
for t in range(1, 1001):
    w, b, m_w, m_b, v_w, v_b = adam_step(X, y, w, b, m_w, m_b, v_w, v_b, t, learning_rate)

print(f"Learned parameters with Adam: w = {w[0][0]:.4f}, b = {b:.4f}")
```

Trang trình bày 7: Chính quy hóa để chống thành kiến

Các kỹ thuật chính quy hóa như chính quy hóa L1 và L2 có thể giúp giảm độ lệch bằng cách tăng thêm số phạt vào hàm mất mát. Điều này khuyến khích các mô hình đơn giản hơn và có thể ngăn chặn việc trang bị quá mức, thường là dấu hiệu sai lệch trong quá trình đào tạo.

```python
def sgd_step_with_regularization(X, y, w, b, learning_rate, l2_lambda):
    N = len(y)
    y_pred = np.dot(X, w) + b
    dw = (1/N) * np.dot(X.T, (y_pred - y)) + l2_lambda * w
    db = (1/N) * np.sum(y_pred - y)
    w -= learning_rate * dw
    b -= learning_rate * db
    return w, b

# Initialize parameters
w = np.random.randn(1, 1)
b = 0
learning_rate = 0.01
l2_lambda = 0.1

# Perform SGD steps with L2 regularization
for _ in range(1000):
    w, b = sgd_step_with_regularization(X, y, w, b, learning_rate, l2_lambda)

print(f"Learned parameters with L2 regularization: w = {w[0][0]:.4f}, b = {b:.4f}")
```

Trang trình bày 8: Xác thực chéo để đánh giá giá thành kiến ​​trúc

Xác thực chéo là một kỹ thuật mạnh mẽ để đánh giá và giảm thiểu sai lệch trong SGD. Bằng cách huấn luyện các tập hợp dữ liệu khác nhau và đánh giá giá trên các tập hợp dữ liệu đã có sẵn, chúng tôi có thể có được tính mạnh mẽ hơn về hiệu suất của mô hình và phát hiện các sai lệch tiềm ẩn.

```python
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

def cross_validate_sgd(X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    mse_scores = []

    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        w = np.random.randn(1, 1)
        b = 0

        for _ in range(1000):
            w, b = sgd_step(X_train, y_train, w, b, learning_rate=0.01)

        y_pred = np.dot(X_val, w) + b
        mse = mean_squared_error(y_val, y_pred)
        mse_scores.append(mse)

    return np.mean(mse_scores), np.std(mse_scores)

mean_mse, std_mse = cross_validate_sgd(X, y)
print(f"Cross-validation MSE: {mean_mse:.4f} (+/- {std_mse:.4f})")
```

Trang trình bày 9: Các phương pháp tập hợp để giảm sai lệch

Các phương pháp tập hợp, hạn chế như đóng bao và tăng tốc, có thể giúp giảm độ lệch bằng cách kết hợp nhiều mô hình. Những kỹ thuật này tận dụng ý tưởng rằng các mô hình khác nhau có thể nắm bắt các cạnh khác nhau của dữ liệu, có khả năng loại bỏ các thành kiến ​​trúc riêng lẻ.

```python
def train_sgd_ensemble(X, y, n_models=5):
    models = []
    for _ in range(n_models):
        w = np.random.randn(1, 1)
        b = 0
        for _ in range(1000):
            w, b = sgd_step(X, y, w, b, learning_rate=0.01)
        models.append((w, b))
    return models

def predict_ensemble(X, models):
    predictions = []
    for w, b in models:
        y_pred = np.dot(X, w) + b
        predictions.append(y_pred)
    return np.mean(predictions, axis=0)

ensemble_models = train_sgd_ensemble(X, y)
ensemble_predictions = predict_ensemble(X, ensemble_models)

mse = mean_squared_error(y, ensemble_predictions)
print(f"Ensemble MSE: {mse:.4f}")
```

Trang trình bày 10: Lịch trình học tập

Lịch trình tốc độ học có thể giúp giảm sai lệch bằng cách điều chỉnh tốc độ học trong quá trình đào tạo. Các chiến lược phổ biến bao gồm phân rã bậc thang, phân rã theo cấp số nhân và ủ cosine. Những lịch trình này có thể giúp quá trình tối ưu hóa điều hướng bối cảnh tổn thất hiệu quả hơn.

```python
def sgd_with_lr_schedule(X, y, epochs, initial_lr, schedule='step', step_size=500, decay=0.1):
    w = np.random.randn(1, 1)
    b = 0

    for epoch in range(epochs):
        if schedule == 'step':
            lr = initial_lr * (decay ** (epoch // step_size))
        elif schedule == 'exponential':
            lr = initial_lr * (decay ** epoch)
        elif schedule == 'cosine':
            lr = initial_lr * 0.5 * (1 + np.cos(np.pi * epoch / epochs))

        w, b = sgd_step(X, y, w, b, lr)

    return w, b

schedules = ['step', 'exponential', 'cosine']
results = []

for schedule in schedules:
    w, b = sgd_with_lr_schedule(X, y, epochs=1000, initial_lr=0.1, schedule=schedule)
    results.append((schedule, w[0][0], b))

for schedule, w, b in results:
    print(f"Schedule: {schedule}, w = {w:.4f}, b = {b:.4f}")
```

Trang trình bày 11: Chuẩn hóa hàng hóa

Chuẩn hóa hàng loạt là một kỹ thuật có thể giúp giảm sự chuyển đồng biến nội bộ và giảm thiểu sai lệch trong chiều sâu mạng lưới thần kinh. Bằng cách chuẩn hóa đầu vào cho mỗi lớp, nó có thể ổn định quá trình học tập và có khả năng cải thiện khả năng độc hóa.

```python
import torch
import torch.nn as nn

class BatchNormModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 10)
        self.bn1 = nn.BatchNorm1d(10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

# Convert numpy arrays to PyTorch tensors
X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y)

# Create and train the model
model = BatchNormModel()
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()

print(f"Final loss: {loss.item():.4f}")
```

Slide 12: Ví dụ thực tế: Phân loại hình ảnh

Trong các nhiệm vụ phân loại hình ảnh, sai lệch SGD có thể biểu hiện dưới dạng hiệu suất kém trên một số lớp hoặc loại hình ảnh nhất định. Ví dụ: một mô hình được đào tạo để phân loại động vật có thể tỏ ra thiên vị đối với các loài phổ biến hơn hoặc gặp khó khăn với những hình ảnh được chụp từ những góc độ khác thường.

```python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True)

# Define a simple CNN
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Train the model
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # Just 2 epochs for demonstration
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch + 1}, loss: {running_loss / len(trainloader):.3f}')

print('Finished Training')
```

Slide 13: Ví dụ thực tế: Xử lý ngôn ngữ tự nhiên

Trong các tác vụ NLP, sai lệch SGD có thể dẫn đến các mô hình hoạt động kém trên một số loại văn bản nhất định hoặc thể hiện những sai lệch không mong muốn. Ví dụ: mô hình phân tích tình cảm có thể gặp khó khăn với sự mỉa mai hoặc thể hiện sự thiên vị đối với các nhóm nhân khẩu học nhất định.

```python
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text, text_lengths):
        embedded = self.embedding(text)
        packed_embedded = pack_padded_sequence(embedded, text_lengths)
        packed_output, hidden = self.rnn(packed_embedded)
        output, output_lengths = pad_packed_sequence(packed_output)
        return self.fc(hidden.squeeze(0))

# Pseudo-training loop
vocab_size = 10000
embedding_dim = 100
hidden_dim = 256
output_dim = 2  # Binary sentiment

model = RNN(vocab_size, embedding_dim, hidden_dim, output_dim)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

for epoch in range(5):  # 5 epochs for demonstration
    for batch in range(100):  # Assume 100 batches per epoch
        # In a real scenario, you'd load actual text data here
        text = torch.randint(0, vocab_size, (20, 32))  # (seq_len, batch_size)
        text_lengths = torch.randint(1, 21, (32,))
        labels = torch.randint(0, 2, (32,))

        optimizer.zero_grad()
        predictions = model(text, text_lengths)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch: {epoch+1}, Loss: {loss.item():.4f}')

print('Training complete')
```

Trang trình bày 14: Giảm thiểu sai lệch về SGD

Để giảm thiểu sự thiên vị trong SGD, hãy xem xét các chiến lược sau:

1. Sử dụng cỡ lô lớn hơn hoặc SGD lô nhỏ
2. Triển khai phương pháp tỷ lệ học tập thích ứng (Adam, RMSprop)
3. Áp dụng kỹ thuật chính quy hóa (L1, L2, dropout)
4. Sử dụng xác thực chéo để điều chỉnh siêu tham số
5. Sử dụng phương pháp tập hợp để kết hợp nhiều mô hình
6. Thực hiện lịch trình tỷ lệ học tập
7. Áp dụng chuẩn hóa hàng loạt trong mạng sâu
8. Xử lý trước và cân bằng dữ liệu của bạn một cách cẩn thận
9. Thường xuyên đánh giá mô hình của bạn trên các bộ thử nghiệm đa dạng
10. Nhận thức được những sai lệch tiềm ẩn trong dữ liệu đào tạo của bạn

Những kỹ thuật này có thể giúp giảm tác động của sai lệch SGD và cải thiện hiệu suất tổng thể cũng như tính công bằng cho các mô hình của bạn.

Trang trình bày 15: Tài nguyên bổ sung

Để biết thêm thông tin về các kỹ thuật tối ưu hóa và sai lệch SGD, hãy xem xét các tài nguyên sau:

1. "Phương pháp tối ưu hóa cho máy học quy mô lớn" của Léon Bottou, Frank E. Curtis và Jorge Nocedal ArXiv: [https://arxiv.org/abs/1606.04838](https://arxiv.org/abs/1606.04838)
2. "Adam: Phương pháp tối ưu hóa ngẫu nhiên" của Diederik P. Kingma và Jimmy Ba ArXiv: [https://arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980)
3. "Về sự hội tụ của Adam và xa hơn" của Sashank J. Reddi, Satyen Kale và Sanjiv Kumar ArXiv: [https://arxiv.org/abs/1904.09237](https://arxiv.org/abs/1904.09237)

Các bài viết này cung cấp các phân tích chuyên sâu về SGD, các biến thể của nó và các đặc tính hội tụ của chúng, mang lại những hiểu biết sâu sắc có giá trị về bản chất của sai lệch tối ưu hóa trong học máy.
