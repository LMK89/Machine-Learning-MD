## Xây dựng đường dốc ngẫu nhiên gốc từ đầu trong Python
Trang trình bày 1: Giới thiệu về Giảm dần ngẫu nhiên độ dốc (SGD)

Tự nhiên giảm dần độ dốc là một cơ sở hóa học tối ưu hóa kỹ thuật được sử dụng trong máy học để giảm thiểu những tổn thất nhỏ. Đây là một phương pháp cập nhật vòng lặp phương pháp dựa trên mô hình tham số dựa trên độ dốc của hàm mất mát đối với các tham số đó. Không giống như truyền tải độ dốc, SGD chỉ sử dụng một tập hợp dữ liệu (lô nhỏ) trong mỗi chu kỳ, giúp hiệu quả hơn đối với các dữ liệu lớn.

```python
import numpy as np
import matplotlib.pyplot as plt

# Simple linear regression model
def predict(X, w, b):
    return X * w + b

# Mean Squared Error loss function
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Generate sample data
np.random.seed(42)
X = np.random.rand(100, 1)
y = 2 * X + 1 + np.random.randn(100, 1) * 0.1

plt.scatter(X, y)
plt.title("Sample Data for Linear Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.show()
```

Slide 2: Độ dốc tính toán

Độ dốc là hàm riêng của các hàm mất đối với từng tham số. Đối với mô hình hồi phục tuyến tính đơn giản của chúng ta, chúng ta cần tính toán độ dốc cho số (w) và độ lệch (b).

```python
def calculate_gradients(X, y, y_pred, w, b):
    m = len(y)
    dw = (2/m) * np.sum(X * (y_pred - y))
    db = (2/m) * np.sum(y_pred - y)
    return dw, db

# Test gradient calculation
w, b = 0, 0
y_pred = predict(X, w, b)
dw, db = calculate_gradients(X, y, y_pred, w, b)
print(f"Initial gradients: dw = {dw:.4f}, db = {db:.4f}")
```

Slide 3: Quy định cập nhật SGD

Quy tắc cập nhật SGD điều chỉnh các tham số theo hướng ngược lại với độ dốc, được chia tỷ lệ theo tốc độ học. Quá trình này được lặp lại với số lần cố định lặp lại hoặc cho đến khi hội tụ.

```python
def sgd_update(w, b, dw, db, learning_rate):
    w -= learning_rate * dw
    b -= learning_rate * db
    return w, b

# Example update
learning_rate = 0.1
w, b = sgd_update(w, b, dw, db, learning_rate)
print(f"Updated parameters: w = {w:.4f}, b = {b:.4f}")
```

Slide 4: Lựa chọn hàng loạt nhỏ

Trong SGD, chúng tôi sử dụng các lô nhỏ để ước tính độ dốc. Điều này bao gồm việc chọn ngẫu nhiên một tập dữ liệu cho mỗi vòng lặp, giúp giảm chi phí tính toán và gây nhiễu cho quá trình tối ưu hóa, có khả năng giúp thoát khỏi bộ tối thiểu cục bộ.

```python
def get_mini_batch(X, y, batch_size):
    indices = np.random.randint(0, len(X), batch_size)
    return X[indices], y[indices]

# Example mini-batch selection
batch_size = 32
X_batch, y_batch = get_mini_batch(X, y, batch_size)
print(f"Mini-batch shapes: X = {X_batch.shape}, y = {y_batch.shape}")
```

Slide 5: Triển khai vòng đào tạo SGD

Bây giờ chúng ta sẽ phát triển vòng đào tạo chính SGD, kết hợp tất cả các thành phần trước đó. Chúng ta sẽ lặp lại một số nguyên nguyên được chỉ định cụ thể, chọn các lô nhỏ, độ dốc tính toán và cập nhật các tham số.

```python
def train_sgd(X, y, learning_rate, batch_size, epochs):
    w, b = 0, 0
    losses = []

    for epoch in range(epochs):
        for _ in range(len(X) // batch_size):
            X_batch, y_batch = get_mini_batch(X, y, batch_size)
            y_pred = predict(X_batch, w, b)
            dw, db = calculate_gradients(X_batch, y_batch, y_pred, w, b)
            w, b = sgd_update(w, b, dw, db, learning_rate)

        # Calculate and store loss for the entire dataset
        y_pred = predict(X, w, b)
        loss = mse_loss(y, y_pred)
        losses.append(loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    return w, b, losses

# Train the model
w, b, losses = train_sgd(X, y, learning_rate=0.1, batch_size=32, epochs=100)
print(f"Final parameters: w = {w:.4f}, b = {b:.4f}")
```

Slide 6: Trực quan hóa tiến trình đào tạo

Để hiểu SGD thuật toán của chúng tôi đang hoạt động như thế nào, chúng tôi có thể giải quyết lỗi theo thời gian và đường dẫn cuối cùng.

```python
# Plot loss over time
plt.plot(losses)
plt.title("Loss vs. Epoch")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.show()

# Plot final regression line
plt.scatter(X, y)
plt.plot(X, predict(X, w, b), color='red')
plt.title("Final Regression Line")
plt.xlabel("X")
plt.ylabel("y")
plt.show()
```

Slide 7: Điều chỉnh siêu thông số

Hiệu suất của SGD phụ thuộc vào một siêu tham số, bao gồm tốc độ học và lô kích thước. Hãy thử nghiệm các giá trị khác nhau để xem hoạt động của chúng đến quá trình đào tạo.

```python
learning_rates = [0.01, 0.1, 1.0]
batch_sizes = [8, 32, 128]

for lr in learning_rates:
    for bs in batch_sizes:
        w, b, losses = train_sgd(X, y, learning_rate=lr, batch_size=bs, epochs=100)
        plt.plot(losses, label=f"LR={lr}, BS={bs}")

plt.title("Loss vs. Epoch for Different Hyperparameters")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.legend()
plt.show()
```

Trang trình bày 8: Lập kế hoạch tỷ lệ học tập

Để cải thiện khả năng tích lũy, chúng tôi có thể phát triển việc thiết lập tốc độ học, giúp giảm tốc độ học tập theo thời gian. Điều này cho phép cập nhật lớn hơn khi bắt đầu đào tạo và điều chỉnh tốt hơn về cuối.

```python
def exponential_decay(initial_lr, decay_rate, epoch):
    return initial_lr * (decay_rate ** epoch)

def train_sgd_with_lr_decay(X, y, initial_lr, decay_rate, batch_size, epochs):
    w, b = 0, 0
    losses = []

    for epoch in range(epochs):
        lr = exponential_decay(initial_lr, decay_rate, epoch)

        for _ in range(len(X) // batch_size):
            X_batch, y_batch = get_mini_batch(X, y, batch_size)
            y_pred = predict(X_batch, w, b)
            dw, db = calculate_gradients(X_batch, y_batch, y_pred, w, b)
            w, b = sgd_update(w, b, dw, db, lr)

        y_pred = predict(X, w, b)
        loss = mse_loss(y, y_pred)
        losses.append(loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, LR: {lr:.4f}, Loss: {loss:.4f}")

    return w, b, losses

w, b, losses = train_sgd_with_lr_decay(X, y, initial_lr=0.1, decay_rate=0.99, batch_size=32, epochs=100)
plt.plot(losses)
plt.title("Loss vs. Epoch with Learning Rate Decay")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.show()
```

Trang trình bày 9: Động lực

Động lực là một kỹ thuật giúp tăng tốc độ SGD theo hướng thích hợp và làm giảm dao động. Nó thực hiện điều này bằng cách bổ sung một phần cập nhật giai đoạn trước vào bản cập nhật hiện tại.

```python
def train_sgd_with_momentum(X, y, learning_rate, momentum, batch_size, epochs):
    w, b = 0, 0
    v_w, v_b = 0, 0
    losses = []

    for epoch in range(epochs):
        for _ in range(len(X) // batch_size):
            X_batch, y_batch = get_mini_batch(X, y, batch_size)
            y_pred = predict(X_batch, w, b)
            dw, db = calculate_gradients(X_batch, y_batch, y_pred, w, b)

            v_w = momentum * v_w - learning_rate * dw
            v_b = momentum * v_b - learning_rate * db

            w += v_w
            b += v_b

        y_pred = predict(X, w, b)
        loss = mse_loss(y, y_pred)
        losses.append(loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    return w, b, losses

w, b, losses = train_sgd_with_momentum(X, y, learning_rate=0.01, momentum=0.9, batch_size=32, epochs=100)
plt.plot(losses)
plt.title("Loss vs. Epoch with Momentum")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.show()
```

Trang trình bày 10: Ví dụ thực tế: Phân loại hình ảnh

SGD được sử dụng rộng rãi trong công việc huấn luyện mạng nơ-ron để phân loại hình ảnh. Hãy tạo một ví dụ đơn giản bằng cách sử dụng MNIST file dữ liệu.

```python
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

# Load MNIST dataset
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train MLP classifier using SGD
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=10, alpha=1e-4,
                    solver='sgd', verbose=10, random_state=1,
                    learning_rate_init=0.1)

mlp.fit(X_train, y_train)
print(f"Training set score: {mlp.score(X_train, y_train):.4f}")
print(f"Test set score: {mlp.score(X_test, y_test):.4f}")
```

Slide 11: Ví dụ thực tế: Xử lý ngôn ngữ tự nhiên

SGD cũng thường được sử dụng trong công việc đào tạo cách nhúng từ cho các tác vụ xử lý ngôn ngữ tự nhiên. Đây là một ví dụ đơn giản sử dụng mô hình Word2Vec.

```python
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')

# Sample sentences
sentences = [
    "The quick brown fox jumps over the lazy dog",
    "Machine learning is a subset of artificial intelligence",
    "Natural language processing is an important field in AI"
]

# Tokenize sentences
tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in sentences]

# Train Word2Vec model using SGD
model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4, sg=1)

# Find similar words
similar_words = model.wv.most_similar("learning", topn=3)
print("Words similar to 'learning':")
for word, score in similar_words:
    print(f"{word}: {score:.4f}")
```

Slide 12: Các công thức và cân nhắc

Khi phát triển SGD, có một số công thức cần xem xét:

1. Việc chọn các thông số thích hợp (tốc độ học, kích thước lô, v.v.) có thể khó khăn và có thể cần phải điều chỉnh rộng rãi.
2. SGD có thể cảm nhận được đối tượng chia tỷ lệ công việc, vì việc xử lý trước dữ liệu thường là cần thiết.
3. Bản chất ngẫu nhiên của SGD có thể gây khó khăn cho việc tái tạo kết quả một cách chính xác.
4. SGD có thể gặp khó khăn với các điểm yên trong các bài toán tối ưu hóa nhiều chiều.

Để giải quyết những công thức này, hãy cân nhắc sử dụng các phương pháp tốc độ học thích ứng như Adam hoặc RMSprop, phát triển các kỹ thuật khởi tạo hợp lý và sử dụng chính quy hóa để ngăn chặn tình trạng trang quá mạnh.

```python
# Example of feature scaling and regularization
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

sgd_reg = SGDRegressor(loss='squared_error', penalty='l2', alpha=0.0001, max_iter=1000, tol=1e-3)
sgd_reg.fit(X_scaled, y.ravel())

print(f"Coefficients: {sgd_reg.coef_}")
print(f"Intercept: {sgd_reg.intercept_}")
```

Slide 13: Kết luận và định hướng tương lai

Stochastic gradient Descent là một thuật toán tối ưu hóa mạnh mẽ, tạo nên xương sống của nhiều mô hình học. Hiệu quả và khả năng xử lý dữ liệu lớn hơn để nó đặc biệt phù hợp với các ứng dụng học sâu. Khi bạn tiếp tục khám phá SGD, hãy cân nhắc công việc nghiên cứu các kỹ thuật nâng cao cao hơn như:

1. Phương pháp tỷ lệ học thích ứng (Adam, RMSprop, Adagrad)
2. Chuẩn hóa hàng hóa
3. Cắt màu chuyển đổi
4. Phương pháp tối ưu bậc hai

Bằng cách thành công SGD và các biến thể của nó, bạn sẽ được trang web tốt để giải quyết nhiều vấn đề về máy học và đóng góp cho những tiến bộ không ngừng trong lĩnh vực này.

```python
# Visualize the optimization landscape
from mpl_toolkits.mplot3d import Axes3D

def loss_surface(w, b):
    return np.mean((y - (X * w + b)) ** 2)

w_range = np.linspace(-1, 4, 100)
b_range = np.linspace(-1, 4, 100)
W, B = np.meshgrid(w_range, b_range)
Z = np.array([loss_surface(w, b) for w, b in zip(np.ravel(W), np.ravel(B))]).reshape(W.shape)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(W, B, Z, cmap='viridis')
ax.set_xlabel('Weight (w)')
ax.set_ylabel('Bias (b)')
ax.set_zlabel('Loss')
ax.set_title('Loss Surface for Linear Regression')
plt.show()
```

Trang trình bày 14: Tài nguyên bổ sung

Đối với những người muốn tìm hiểu sâu hơn về ngẫu nhiên giảm độ dốc và các ứng dụng của nó, thì đây là một số tài nguyên có giá trị:

1. “Phương pháp tối ưu hóa cho máy học quy mô lớn” của Léon Bottou, Frank E. Curtis và Jorge Nocedal (2018). Có tại: [https://arxiv.org/abs/1606.04838](https://arxiv.org/abs/1606.04838)
2. "Adam: Phương pháp tối ưu hoá ngẫu nhiên" của Diederik P. Kingma và Jimmy Ba (2014). Có tại: [https://arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980)
3. "Tổng quan về các thuật toán tối ưu hóa giảm dần độ dốc" của Sebastian Ruder (2016). Có tại: [https://arxiv.org/abs/1609.04747](https://arxiv.org/abs/1609.04747)

Bài viết này cung cấp các phân tích chuyên sâu về SGD và các biến thể của nó, đưa ra những hiểu biết có giá trị về nền tảng lý thuyết và ứng dụng thực tế của các kỹ thuật tối ưu hóa này.
