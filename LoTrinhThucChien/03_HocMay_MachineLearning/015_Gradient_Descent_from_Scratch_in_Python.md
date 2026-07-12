## Giảm độ dốc từ đầu trong Python
Slide 1: Giới thiệu về Độ dốc giảm dần

Độ dốc giảm dần là một cơ sở hóa ưu tiên tối ưu thuật toán được sử dụng trong máy học để giảm thiểu các chức năng tối thiểu của mô hình. Nó điều chỉnh vòng lặp lặp lại các tham số của mô hình theo hướng tăng dần của hàm chi phí.

```python
import numpy as np
import matplotlib.pyplot as plt

def cost_function(x):
    return x**2 + 5*x + 10

x = np.linspace(-10, 10, 100)
y = cost_function(x)

plt.plot(x, y)
plt.title('Cost Function')
plt.xlabel('x')
plt.ylabel('Cost')
plt.show()
```

Trang trình bày 2: Độ dốc

Độ dốc là một cường đạo chức năng riêng biệt hướng đi lên dốc nhất. Trong quá trình giảm độ dốc, chúng tôi chuyển hướng ngược lại để giảm thiểu chi phí.

```python
def gradient(x):
    return 2*x + 5

x = np.linspace(-10, 10, 100)
grad = gradient(x)

plt.plot(x, grad)
plt.title('Gradient of Cost Function')
plt.xlabel('x')
plt.ylabel('Gradient')
plt.axhline(y=0, color='r', linestyle='--')
plt.show()
```

Slide 3: Thuật toán giảm dần cơ sở dữ liệu

Cập nhật thuật toán các vòng tham số đi lặp lại bằng cách trừ đi tốc độ học và độ dốc của các tham số giá trị hiện tại.

```python
def gradient_descent(start_x, learning_rate, num_iterations):
    x = start_x
    for i in range(num_iterations):
        grad = gradient(x)
        x = x - learning_rate * grad
        print(f"Iteration {i+1}: x = {x:.4f}, cost = {cost_function(x):.4f}")
    return x

optimal_x = gradient_descent(start_x=5, learning_rate=0.1, num_iterations=20)
print(f"Optimal x: {optimal_x:.4f}")
```

Slide 4: Tỷ lệ học tập

Bước xác định tốc độ học ở mỗi vòng lặp. Tốc độ học quá lớn có thể vượt quá mức tối thiểu, trong khi tốc độ học quá nhỏ có thể dẫn đến tốc độ học chậm.

```python
learning_rates = [0.01, 0.1, 0.5]
start_x = 5
iterations = 50

for lr in learning_rates:
    x = start_x
    xs = [x]
    for _ in range(iterations):
        x = x - lr * gradient(x)
        xs.append(x)

    plt.plot(range(iterations+1), xs, label=f'LR = {lr}')

plt.legend()
plt.title('Effect of Learning Rate')
plt.xlabel('Iterations')
plt.ylabel('x')
plt.show()
```

Trang trình bày 5: Giảm dần độ dốc ngẫu nhiên

Giảm dần độ dốc ngẫu nhiên (SGD) tính toán độ dốc bằng cách sử dụng một mẫu ngẫu nhiên duy nhất từ ​​​​tập dữ liệu, làm cho nó nhanh hơn và có thể thoát khỏi bộ cực tiểu địa phương dễ dàng hơn.

```python
import random

def stochastic_gradient_descent(data, labels, learning_rate, num_iterations):
    w, b = 0, 0
    for _ in range(num_iterations):
        idx = random.randint(0, len(data)-1)
        x, y = data[idx], labels[idx]
        y_pred = w * x + b
        error = y_pred - y
        w -= learning_rate * error * x
        b -= learning_rate * error
    return w, b

# Example usage
data = [1, 2, 3, 4, 5]
labels = [2, 4, 6, 8, 10]
w, b = stochastic_gradient_descent(data, labels, 0.01, 1000)
print(f"Learned parameters: w = {w:.4f}, b = {b:.4f}")
```

Trang trình bày 6: Giảm dần độ dốc theo chiều nhỏ

Giảm dần độ dốc theo lô nhỏ kết hợp các ưu tiên của độ dốc tăng dần theo lô và ngẫu nhiên bằng cách sử dụng một tập hợp dữ liệu ngẫu nhiên nhỏ cho mỗi lần cập nhật.

```python
def mini_batch_gradient_descent(data, labels, batch_size, learning_rate, num_iterations):
    w, b = 0, 0
    for _ in range(num_iterations):
        batch_indices = np.random.choice(len(data), batch_size, replace=False)
        x_batch = [data[i] for i in batch_indices]
        y_batch = [labels[i] for i in batch_indices]

        grad_w, grad_b = 0, 0
        for x, y in zip(x_batch, y_batch):
            y_pred = w * x + b
            error = y_pred - y
            grad_w += error * x
            grad_b += error

        w -= learning_rate * grad_w / batch_size
        b -= learning_rate * grad_b / batch_size

    return w, b

# Example usage
data = [1, 2, 3, 4, 5]
labels = [2, 4, 6, 8, 10]
w, b = mini_batch_gradient_descent(data, labels, batch_size=2, learning_rate=0.01, num_iterations=1000)
print(f"Learned parameters: w = {w:.4f}, b = {b:.4f}")
```

Trang trình bày 7: Động lực

Động lực giúp tăng tốc độ giảm độ dốc theo hướng thích hợp và làm giảm dao động. Nó thực hiện điều này bằng cách thêm một phần của bản cập nhật or trước vào bản cập nhật or hiện tại.

```python
def momentum_gradient_descent(start_x, learning_rate, momentum, num_iterations):
    x = start_x
    velocity = 0
    for i in range(num_iterations):
        grad = gradient(x)
        velocity = momentum * velocity - learning_rate * grad
        x = x + velocity
        print(f"Iteration {i+1}: x = {x:.4f}, cost = {cost_function(x):.4f}")
    return x

optimal_x = momentum_gradient_descent(start_x=5, learning_rate=0.1, momentum=0.9, num_iterations=20)
print(f"Optimal x: {optimal_x:.4f}")
```

Trình bày 8: Tỷ lệ học ứng dụng

Phương pháp điều chỉnh tốc độ học thích ứng với điều chỉnh tốc độ học cho từng tham số. Một phương pháp phổ biến là AdaGrad, phương pháp này điều chỉnh tốc độ học theo các tham số, thực hiện cập nhật nhỏ hơn cho các tính năng thường xuyên xuất hiện.

```python
def adagrad(start_x, learning_rate, num_iterations):
    x = start_x
    sum_squared_gradients = 0
    epsilon = 1e-8  # Small value to avoid division by zero

    for i in range(num_iterations):
        grad = gradient(x)
        sum_squared_gradients += grad**2
        adjusted_learning_rate = learning_rate / (np.sqrt(sum_squared_gradients) + epsilon)
        x = x - adjusted_learning_rate * grad
        print(f"Iteration {i+1}: x = {x:.4f}, cost = {cost_function(x):.4f}")

    return x

optimal_x = adagrad(start_x=5, learning_rate=1, num_iterations=20)
print(f"Optimal x: {optimal_x:.4f}")
```

Trang trình bày 9: Độ dốc giảm dần cho các hàm đa biến

Trong thực tế, họ thường xử lý nhiều hàm. Độ dốc giảm dần có thể được mở rộng để hoạt động với các hàm này bằng cách tính toán hàm riêng cho từng biến.

```python
def multivariable_cost(x, y):
    return x**2 + y**2

def multivariable_gradient(x, y):
    return np.array([2*x, 2*y])

def multivariable_gradient_descent(start_x, start_y, learning_rate, num_iterations):
    point = np.array([start_x, start_y])

    for i in range(num_iterations):
        grad = multivariable_gradient(point[0], point[1])
        point = point - learning_rate * grad
        cost = multivariable_cost(point[0], point[1])
        print(f"Iteration {i+1}: x = {point[0]:.4f}, y = {point[1]:.4f}, cost = {cost:.4f}")

    return point

optimal_point = multivariable_gradient_descent(start_x=5, start_y=5, learning_rate=0.1, num_iterations=20)
print(f"Optimal point: x = {optimal_point[0]:.4f}, y = {optimal_point[1]:.4f}")
```

Trang trình bày 10: Tăng dần độ dốc trực quan

Hình dung đường dốc tăng dần có thể giúp hiểu được thuật toán lũy tiến đến mức tối thiểu. Vui lòng tạo một đường viền biểu đồ và hiển thị mức độ ưu tiên của đường dẫn.

```python
def plot_gradient_descent(start_x, start_y, learning_rate, num_iterations):
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)
    Z = multivariable_cost(X, Y)

    plt.figure(figsize=(10, 8))
    plt.contour(X, Y, Z, levels=50)

    point = np.array([start_x, start_y])
    path = [point]

    for _ in range(num_iterations):
        grad = multivariable_gradient(point[0], point[1])
        point = point - learning_rate * grad
        path.append(point)

    path = np.array(path)
    plt.plot(path[:, 0], path[:, 1], 'ro-')
    plt.title('Gradient Descent Path')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

plot_gradient_descent(start_x=8, start_y=8, learning_rate=0.1, num_iterations=50)
```

Slide 11: Ví dụ thực tế: Hồi quy tuyến tính

Độ dốc giảm dần thường được sử dụng trong tuyến tính hồi phục để tìm đường phù hợp nhất cho một tập dữ liệu.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate some sample data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Gradient descent for linear regression
def linear_regression_gradient_descent(X, y, learning_rate, num_iterations):
    m = len(y)
    theta = np.random.randn(2, 1)

    for _ in range(num_iterations):
        gradients = 2/m * X.T.dot(X.dot(theta) - y)
        theta = theta - learning_rate * gradients

    return theta

X_b = np.c_[np.ones((100, 1)), X]  # Add bias term
theta = linear_regression_gradient_descent(X_b, y, learning_rate=0.01, num_iterations=1000)

# Plot the results
plt.scatter(X, y)
plt.plot(X, X_b.dot(theta), color='r')
plt.title('Linear Regression using Gradient Descent')
plt.xlabel('X')
plt.ylabel('y')
plt.show()

print(f"Estimated parameters: intercept = {theta[0][0]:.4f}, slope = {theta[1][0]:.4f}")
```

Slide 12: Ví dụ thực tế: Phân loại hình ảnh

Giảm dần độ dốc là rất quan trọng trong công việc đào tạo mạng lưới thần kinh cho các nhiệm vụ phân loại hình ảnh. Vui lòng sử dụng một ví dụ đơn giản cho MNIST data file.

```python
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

# Load and preprocess the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 784) / 255.0
X_test = X_test.reshape(10000, 784) / 255.0

# Create a simple neural network
model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dense(10, activation='softmax')
])

# Compile the model with SGD optimizer
model.compile(optimizer=SGD(learning_rate=0.01),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_split=0.2, batch_size=32)

# Plot the training history
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```

Slide 13: Các công thức và cân nhắc

Việc giảm độ dốc, mặc dù mạnh mẽ nhưng phải đối mặt với các công thức như bị kẹt trong cực tiểu cục bộ, hội tụ chậm đối với các vấn đề không được điều chỉnh và nhu cầu điều chỉnh siêu kỹ thuật số cẩn thận. Các biến có thể nâng cao như Adam và RMSprop giải quyết một số vấn đề này.

```python
import numpy as np
import matplotlib.pyplot as plt

def complex_function(x):
    return np.sin(x) + 0.1 * x**2

x = np.linspace(-10, 10, 1000)
y = complex_function(x)

plt.figure(figsize=(12, 6))
plt.plot(x, y)
plt.title('Complex Function with Multiple Local Minima')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.axhline(y=0, color='r', linestyle='--')
plt.show()
```

Trang trình bày 14: Tài nguyên bổ sung

Đối với những người muốn tìm hiểu sâu hơn về các thuật toán tối ưu hóa và giảm độ dốc, dưới đây là một số tài nguyên được xuất bản:

1. "Tối ưu hóa cho máy học" của Suvrit Sra, Sebastian Nowozin và Stephen J. Wright (Nhà xuất bản MIT)
2. "Xem lại phần gốc theo độ dốc: Một góc nhìn mới dựa trên việc đi theo con đường" của Bin Shi et al. (arXiv:2008.11266)
3. "Tổng quan về các thuật toán tối ưu hóa giảm dần độ dốc" của Sebastian Ruder (arXiv:1609.04747)

Bạn có thể tìm thấy những tờ giấy này trên ArXiv.org bằng cách tìm kiếm ID arXiv tương ứng của chúng.
