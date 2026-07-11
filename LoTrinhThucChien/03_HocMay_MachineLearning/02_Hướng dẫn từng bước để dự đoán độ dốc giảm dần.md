## Hướng dẫn từng bước để dự đoán độ dốc giảm dần
Slide 1: Giới thiệu về Giảm dần độ dốc

Giảm dần độ dốc là một thuật toán tối ưu hóa cơ bản trong học máy được sử dụng để giảm thiểu hàm chi phí và cải thiện hiệu suất mô hình. Nó điều chỉnh lặp đi lặp lại các tham số mô hình để tìm ra giải pháp tối ưu. Quá trình này biến đổi các điểm ngẫu nhiên ban đầu trong không gian tham số thành các dự đoán mạnh mẽ.

```python
import numpy as np
import matplotlib.pyplot as plt

def cost_function(x):
    return x**2 + 5*x + 10

x = np.linspace(-10, 10, 100)
y = cost_function(x)

plt.plot(x, y)
plt.title('Cost Function')
plt.xlabel('Parameter Value')
plt.ylabel('Cost')
plt.show()
```

Slide 2: Điểm khởi đầu

Quá trình đào tạo bắt đầu bằng việc khởi tạo các tham số (trọng số và độ lệch) một cách ngẫu nhiên. Các tham số này biểu thị một điểm trong không gian nhiều chiều, tương ứng với cấu hình mô hình cụ thể và giá trị lỗi.

```python
import numpy as np

# Initialize random parameters
np.random.seed(42)
initial_params = np.random.randn(5)

print("Initial parameters:", initial_params)
```

Slide 3: Mục tiêu - Tìm mức tối thiểu

Mục tiêu của việc giảm độ dốc là tìm ra điểm mà lỗi mô hình (hàm chi phí) được giảm thiểu, được gọi là mức tối thiểu toàn cầu. Điều này đạt được bằng cách di chuyển lặp đi lặp lại về phía các vùng có lỗi thấp hơn.

```python
import numpy as np
import matplotlib.pyplot as plt

def cost_function(x):
    return x**2 + 5*x + 10

x = np.linspace(-10, 10, 100)
y = cost_function(x)

plt.plot(x, y)
plt.title('Cost Function with Global Minimum')
plt.xlabel('Parameter Value')
plt.ylabel('Cost')
plt.plot(-2.5, cost_function(-2.5), 'ro', label='Global Minimum')
plt.legend()
plt.show()
```

Slide 4: Bước 1: Tính gradient

Tại mỗi điểm, chúng tôi tính toán độ dốc, biểu thị hướng đi lên dốc nhất. Vì mục tiêu của chúng tôi là giảm thiểu lỗi nên chúng tôi di chuyển theo hướng ngược lại với độ dốc.

```python
def gradient(x):
    return 2*x + 5

x = 2
grad = gradient(x)
print(f"Gradient at x = {x}: {grad}")

# Visualize gradient
x = np.linspace(-10, 10, 100)
y = cost_function(x)
plt.plot(x, y)
plt.quiver(2, cost_function(2), -1, -gradient(2), scale=20, color='r')
plt.title('Gradient at a Point')
plt.xlabel('Parameter Value')
plt.ylabel('Cost')
plt.show()
```

Slide 5: Bước 2: Cập nhật điểm

Chúng tôi điều chỉnh các tham số bằng cách thực hiện một bước theo hướng ngược lại với độ dốc. Kích thước bước được kiểm soát bởi tốc độ học tập.

```python
def gradient_descent_step(x, learning_rate):
    return x - learning_rate * gradient(x)

x = 2
learning_rate = 0.1
new_x = gradient_descent_step(x, learning_rate)

print(f"Old x: {x}")
print(f"New x: {new_x}")
print(f"Cost reduction: {cost_function(x) - cost_function(new_x)}")
```

Slide 6: Bước 3: Lặp lại cho đến khi hội tụ

Quá trình này lặp đi lặp lại, với mô hình đi qua không gian tham số, cập nhật vị trí của nó theo từng bước và giảm dần lỗi.

```python
def gradient_descent(start_x, learning_rate, num_iterations):
    x = start_x
    history = [x]
    for _ in range(num_iterations):
        x = gradient_descent_step(x, learning_rate)
        history.append(x)
    return x, history

final_x, history = gradient_descent(5, 0.1, 20)

print(f"Final x: {final_x}")
print(f"Final cost: {cost_function(final_x)}")

# Plot optimization path
x = np.linspace(-10, 10, 100)
y = cost_function(x)
plt.plot(x, y)
plt.plot(history, [cost_function(x) for x in history], 'ro-')
plt.title('Gradient Descent Optimization Path')
plt.xlabel('Parameter Value')
plt.ylabel('Cost')
plt.show()
```

Trang trình bày 7: Giảm dần độ dốc hàng loạt

Giảm dần theo lô sử dụng tập dữ liệu đầy đủ cho mỗi bước, cung cấp mức giảm ổn định nhưng có khả năng chậm về mức tối thiểu.

```python
import numpy as np

def batch_gradient_descent(X, y, learning_rate, num_iterations):
    m, n = X.shape
    theta = np.zeros(n)

    for _ in range(num_iterations):
        h = np.dot(X, theta)
        gradient = (1/m) * np.dot(X.T, (h - y))
        theta -= learning_rate * gradient

    return theta

# Example usage
X = np.array([[1, 1], [1, 2], [1, 3]])
y = np.array([1, 2, 3])
theta = batch_gradient_descent(X, y, 0.01, 1000)
print("Optimized parameters:", theta)
```

Trang trình bày 8: Giảm dần độ dốc ngẫu nhiên (SGD)

SGD cập nhật các tham số sau mỗi điểm dữ liệu, giúp quá trình này nhanh hơn nhưng ồn hơn so với phương pháp giảm độ dốc hàng loạt.

```python
import numpy as np

def stochastic_gradient_descent(X, y, learning_rate, num_epochs):
    m, n = X.shape
    theta = np.zeros(n)

    for _ in range(num_epochs):
        for i in range(m):
            random_index = np.random.randint(m)
            xi = X[random_index:random_index+1]
            yi = y[random_index:random_index+1]
            gradient = xi.T.dot(xi.dot(theta) - yi)
            theta -= learning_rate * gradient

    return theta

# Example usage
X = np.array([[1, 1], [1, 2], [1, 3]])
y = np.array([1, 2, 3])
theta = stochastic_gradient_descent(X, y, 0.01, 1000)
print("Optimized parameters:", theta)
```

Trang trình bày 9: Giảm dần độ dốc theo đợt nhỏ

Giảm dần theo đợt nhỏ kết hợp các yếu tố của cả phương pháp hàng loạt và ngẫu nhiên, cân bằng tốc độ và độ chính xác.

```python
import numpy as np

def mini_batch_gradient_descent(X, y, learning_rate, num_epochs, batch_size):
    m, n = X.shape
    theta = np.zeros(n)

    for _ in range(num_epochs):
        indices = np.random.permutation(m)
        X = X[indices]
        y = y[indices]

        for i in range(0, m, batch_size):
            xi = X[i:i+batch_size]
            yi = y[i:i+batch_size]
            gradient = xi.T.dot(xi.dot(theta) - yi) / batch_size
            theta -= learning_rate * gradient

    return theta

# Example usage
X = np.array([[1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6]])
y = np.array([1, 2, 3, 4, 5, 6])
theta = mini_batch_gradient_descent(X, y, 0.01, 1000, 2)
print("Optimized parameters:", theta)
```

Trang trình bày 10: Tỷ lệ học tập

Tốc độ học là một siêu tham số quan trọng kiểm soát kích thước bước trong cập nhật tham số. Tốc độ học quá lớn có thể gây ra sự phân kỳ, trong khi tốc độ quá nhỏ dẫn đến sự hội tụ chậm.

```python
import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(start_x, learning_rate, num_iterations):
    x = start_x
    history = [x]
    for _ in range(num_iterations):
        x = x - learning_rate * (2*x + 5)
        history.append(x)
    return history

x = np.linspace(-10, 10, 100)
y = x**2 + 5*x + 10

plt.figure(figsize=(12, 4))
for lr in [0.01, 0.1, 0.5]:
    history = gradient_descent(8, lr, 20)
    plt.plot(history, [i**2 + 5*i + 10 for i in history], 'o-', label=f'LR = {lr}')

plt.plot(x, y, 'r--')
plt.title('Effect of Learning Rate on Convergence')
plt.xlabel('Parameter Value')
plt.ylabel('Cost')
plt.legend()
plt.show()
```

Trang trình bày 11: Động lực

Động lượng là một kỹ thuật giúp tăng tốc độ giảm độ dốc theo hướng thích hợp và làm giảm dao động.

```python
import numpy as np
import matplotlib.pyplot as plt

def gradient_descent_momentum(start_x, learning_rate, momentum, num_iterations):
    x = start_x
    velocity = 0
    history = [x]
    for _ in range(num_iterations):
        gradient = 2*x + 5
        velocity = momentum * velocity - learning_rate * gradient
        x += velocity
        history.append(x)
    return history

x = np.linspace(-10, 10, 100)
y = x**2 + 5*x + 10

plt.figure(figsize=(12, 4))
history_standard = gradient_descent(8, 0.1, 20)
history_momentum = gradient_descent_momentum(8, 0.1, 0.9, 20)

plt.plot(history_standard, [i**2 + 5*i + 10 for i in history_standard], 'o-', label='Standard GD')
plt.plot(history_momentum, [i**2 + 5*i + 10 for i in history_momentum], 'o-', label='GD with Momentum')
plt.plot(x, y, 'r--')
plt.title('Standard Gradient Descent vs Gradient Descent with Momentum')
plt.xlabel('Parameter Value')
plt.ylabel('Cost')
plt.legend()
plt.show()
```

Slide 12: Ví dụ thực tế: Hồi quy tuyến tính

Giảm dần độ dốc thường được sử dụng trong hồi quy tuyến tính để tìm đường phù hợp nhất cho một tập hợp các điểm dữ liệu.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate sample data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Gradient descent for linear regression
X_b = np.c_[np.ones((100, 1)), X]  # add bias term
theta = np.random.randn(2, 1)

learning_rate = 0.1
n_iterations = 1000
m = 100

for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - learning_rate * gradients

print("Final parameters:", theta.ravel())

# Plot results
plt.scatter(X, y)
plt.plot(X, X_b.dot(theta), color='r')
plt.title('Linear Regression using Gradient Descent')
plt.xlabel('X')
plt.ylabel('y')
plt.show()
```

Slide 13: Ví dụ thực tế: Phân loại hình ảnh

Giảm dần độ dốc là rất quan trọng trong việc đào tạo mạng lưới thần kinh cho các nhiệm vụ phân loại hình ảnh.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
digits = load_digits()
X, y = digits.data, digits.target

# Split and preprocess data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define simple neural network
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def forward(X, weights):
    return sigmoid(np.dot(X, weights))

def backward(X, y, output):
    return np.dot(X.T, (output - y)) / len(y)

# Train network
np.random.seed(42)
n_features = X_train.shape[1]
n_classes = 10
weights = np.random.randn(n_features, n_classes)

learning_rate = 0.01
n_iterations = 1000

for _ in range(n_iterations):
    output = forward(X_train, weights)
    gradient = backward(X_train, np.eye(n_classes)[y_train], output)
    weights -= learning_rate * gradient

# Evaluate
predictions = np.argmax(forward(X_test, weights), axis=1)
accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy:.2f}")

# Visualize a prediction
sample_index = np.random.randint(len(X_test))
sample_image = X_test[sample_index].reshape(8, 8)
sample_prediction = predictions[sample_index]

plt.imshow(sample_image, cmap='gray')
plt.title(f"Prediction: {sample_prediction}")
plt.axis('off')
plt.show()
```

Slide 14: Kết luận và nguồn tài liệu bổ sung

Giảm dần độ dốc là một kỹ thuật tối ưu hóa mạnh mẽ cho phép các mô hình học máy học hỏi từ dữ liệu và đưa ra dự đoán chính xác. Các biến thể của nó, chẳng hạn như SGD và giảm độ dốc theo lô nhỏ, mang lại sự linh hoạt trong việc cân bằng giữa hiệu quả tính toán và độ ổn định hội tụ.

Để khám phá thêm về độ dốc giảm dần và các ứng dụng của nó trong học máy, hãy xem xét các tài nguyên sau:

1. "Gradient Descent Revisited" của S. Ruder (2016), arXiv:1609.04747 URL: [https://arxiv.org/abs/1609.04747](https://arxiv.org/abs/1609.04747)
2. "Tổng quan về các thuật toán tối ưu hóa giảm dần độ dốc" của S. Ruder (2017), arXiv:1609.04747v2 URL: [https://arxiv.org/abs/1609.04747v2](https://arxiv.org/abs/1609.04747v2)
