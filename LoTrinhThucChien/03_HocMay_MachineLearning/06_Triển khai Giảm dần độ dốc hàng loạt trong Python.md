## Triển khai Giảm dần độ dốc hàng loạt trong Python
Trang trình bày 1: Giới thiệu về Giảm dần độ dốc hàng loạt

Giảm dần độ dốc hàng loạt là một thuật toán tối ưu hóa cơ bản được sử dụng trong học máy để giảm thiểu hàm chi phí của mô hình. Nó cập nhật các tham số mô hình bằng cách tính toán độ dốc của toàn bộ tập dữ liệu huấn luyện trong mỗi lần lặp. Cách tiếp cận này đảm bảo sự hội tụ ổn định nhưng có thể tốn kém về mặt tính toán đối với các tập dữ liệu lớn.

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
```

Trang trình bày 2: Hàm chi phí và Độ dốc

Hàm chi phí đo lường sự khác biệt giữa giá trị dự đoán và giá trị thực tế. Đối với hồi quy tuyến tính, chúng tôi sử dụng Lỗi bình phương trung bình (MSE). Độ dốc của hàm chi phí đối với các tham số cho biết hướng đi lên dốc nhất.

```python
def cost_function(X, y, theta):
    m = len(y)
    predictions = np.dot(X, theta)
    cost = (1/(2*m)) * np.sum((predictions - y)**2)
    return cost

def gradient(X, y, theta):
    m = len(y)
    predictions = np.dot(X, theta)
    grad = (1/m) * np.dot(X.T, (predictions - y))
    return grad
```

Trang trình bày 3: Triển khai Trình tối ưu hóa

Trình tối ưu hóa Giảm dần độ dốc hàng loạt của chúng tôi sẽ lặp qua một số bước cố định, cập nhật các tham số trong mỗi lần lặp dựa trên độ dốc được tính toán.

```python
def batch_gradient_descent(X, y, learning_rate, num_iterations):
    theta = np.zeros(X.shape[1])
    cost_history = []

    for _ in range(num_iterations):
        grad = gradient(X, y, theta)
        theta -= learning_rate * grad
        cost = cost_function(X, y, theta)
        cost_history.append(cost)

    return theta, cost_history
```

Slide 4: Chuẩn bị dữ liệu

Trước khi áp dụng trình tối ưu hóa, chúng ta cần chuẩn bị dữ liệu của mình. Điều này bao gồm việc chuẩn hóa và thêm thuật ngữ sai lệch vào ma trận đặc trưng của chúng tôi.

```python
def normalize_features(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std

def add_bias_term(X):
    return np.c_[np.ones((X.shape[0], 1)), X]

# Example usage
X_raw = np.random.randn(100, 3)
y = np.random.randn(100)

X_normalized = normalize_features(X_raw)
X = add_bias_term(X_normalized)
```

Slide 5: Điều chỉnh siêu tham số

Tốc độ học và số lần lặp là các siêu tham số quan trọng. Tốc độ học quá cao có thể gây ra sự phân kỳ, trong khi tốc độ học quá thấp có thể dẫn đến sự hội tụ chậm.

```python
learning_rates = [0.001, 0.01, 0.1, 1.0]
iterations = [100, 500, 1000]

best_cost = float('inf')
best_params = None

for lr in learning_rates:
    for iters in iterations:
        theta, cost_history = batch_gradient_descent(X, y, lr, iters)
        final_cost = cost_history[-1]
        if final_cost < best_cost:
            best_cost = final_cost
            best_params = (lr, iters)

print(f"Best parameters: Learning Rate = {best_params[0]}, Iterations = {best_params[1]}")
```

Slide 6: Trực quan hóa sự hội tụ

Việc vẽ đồ thị hàm chi phí qua các lần lặp giúp chúng ta hiểu được hành vi hội tụ của trình tối ưu hóa.

```python
import matplotlib.pyplot as plt

def plot_convergence(cost_history):
    plt.plot(range(len(cost_history)), cost_history)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Convergence of Batch Gradient Descent')
    plt.show()

# Assuming we've run our optimizer
theta, cost_history = batch_gradient_descent(X, y, 0.01, 1000)
plot_convergence(cost_history)
```

Slide 7: Ví dụ thực tế: Dự đoán giá nhà

Hãy áp dụng trình tối ưu hóa Batch gradient Descent của chúng tôi để dự đoán giá nhà dựa trên các đặc điểm như diện tích và số phòng ngủ.

```python
# Simulated dataset
np.random.seed(42)
square_feet = np.random.randint(1000, 5000, 1000)
bedrooms = np.random.randint(1, 6, 1000)
prices = 100000 + 100 * square_feet + 20000 * bedrooms + np.random.randn(1000) * 50000

X = np.column_stack((square_feet, bedrooms))
y = prices

X_normalized = normalize_features(X)
X_with_bias = add_bias_term(X_normalized)

theta, cost_history = batch_gradient_descent(X_with_bias, y, 0.01, 1000)

print("Learned parameters:", theta)
plot_convergence(cost_history)
```

Slide 8: Đưa ra dự đoán

Sau khi có các tham số được tối ưu hóa, chúng tôi có thể sử dụng chúng để đưa ra dự đoán về dữ liệu mới.

```python
def predict(X, theta):
    return np.dot(X, theta)

# New house: 2500 sq ft, 3 bedrooms
new_house = np.array([[2500, 3]])
new_house_normalized = (new_house - np.mean(X, axis=0)) / np.std(X, axis=0)
new_house_with_bias = add_bias_term(new_house_normalized)

predicted_price = predict(new_house_with_bias, theta)
print(f"Predicted price for a 2500 sq ft house with 3 bedrooms: ${predicted_price[0]:.2f}")
```

Slide 9: Xử lý sự không hội tụ

Đôi khi, trình tối ưu hóa có thể không hội tụ do các vấn đề như tốc độ học tập cao hoặc dữ liệu không được điều chỉnh. Chúng ta có thể triển khai tính năng dừng sớm để xử lý việc này.

```python
def batch_gradient_descent_with_early_stopping(X, y, learning_rate, max_iterations, tolerance=1e-6):
    theta = np.zeros(X.shape[1])
    cost_history = []

    for i in range(max_iterations):
        prev_theta = theta.()
        grad = gradient(X, y, theta)
        theta -= learning_rate * grad
        cost = cost_function(X, y, theta)
        cost_history.append(cost)

        if np.all(np.abs(theta - prev_theta) < tolerance):
            print(f"Converged after {i+1} iterations")
            break

    return theta, cost_history

# Example usage
theta, cost_history = batch_gradient_descent_with_early_stopping(X_with_bias, y, 0.01, 10000)
```

Trang trình bày 10: Giảm dần từng đợt nhỏ

Đối với các tập dữ liệu lớn hơn, chúng ta có thể sử dụng Giảm dần độ dốc hàng loạt nhỏ, kết hợp các ưu điểm của cả Giảm dần độ dốc hàng loạt và ngẫu nhiên.

```python
def mini_batch_gradient_descent(X, y, learning_rate, num_iterations, batch_size):
    m, n = X.shape
    theta = np.zeros(n)
    cost_history = []

    for _ in range(num_iterations):
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for i in range(0, m, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            grad = gradient(X_batch, y_batch, theta)
            theta -= learning_rate * grad

        cost = cost_function(X, y, theta)
        cost_history.append(cost)

    return theta, cost_history

# Example usage
theta, cost_history = mini_batch_gradient_descent(X_with_bias, y, 0.01, 1000, 32)
```

Slide 11: Ví dụ thực tế: Phân loại hoa diên vĩ

Hãy sử dụng trình tối ưu hóa Batch gradient Descent của chúng tôi cho tác vụ phân loại trên tập dữ liệu Iris nổi tiếng.

```python
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load and prepare the data
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Add bias term
X_train_with_bias = add_bias_term(X_train_scaled)
X_test_with_bias = add_bias_term(X_test_scaled)

# Train the model (using one-vs-rest strategy for multiclass)
theta_list = []
for class_label in range(3):
    y_binary = (y_train == class_label).astype(int)
    theta, _ = batch_gradient_descent(X_train_with_bias, y_binary, 0.01, 1000)
    theta_list.append(theta)

# Make predictions
def predict_iris(X, theta_list):
    predictions = np.array([predict(X, theta) for theta in theta_list]).T
    return np.argmax(predictions, axis=1)

y_pred = predict_iris(X_test_with_bias, theta_list)
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy on test set: {accuracy:.2f}")
```

Slide 12: Chính quy hóa

Để ngăn chặn việc trang bị quá mức, chúng ta có thể thêm tính chính quy vào hàm chi phí và tính toán độ dốc.

```python
def cost_function_regularized(X, y, theta, lambda_):
    m = len(y)
    predictions = np.dot(X, theta)
    cost = (1/(2*m)) * np.sum((predictions - y)**2)
    regularization = (lambda_ / (2*m)) * np.sum(theta[1:]**2)  # Exclude bias term
    return cost + regularization

def gradient_regularized(X, y, theta, lambda_):
    m = len(y)
    predictions = np.dot(X, theta)
    grad = (1/m) * np.dot(X.T, (predictions - y))
    grad[1:] += (lambda_ / m) * theta[1:]  # Regularize all but the bias term
    return grad

def batch_gradient_descent_regularized(X, y, learning_rate, num_iterations, lambda_):
    theta = np.zeros(X.shape[1])
    cost_history = []

    for _ in range(num_iterations):
        grad = gradient_regularized(X, y, theta, lambda_)
        theta -= learning_rate * grad
        cost = cost_function_regularized(X, y, theta, lambda_)
        cost_history.append(cost)

    return theta, cost_history

# Example usage
lambda_ = 0.1
theta_reg, cost_history_reg = batch_gradient_descent_regularized(X_with_bias, y, 0.01, 1000, lambda_)
```

Trang trình bày 13: Giảm dần độ dốc dựa trên động lượng

Động lượng có thể giúp tăng tốc độ hội tụ, đặc biệt ở những khu vực có độ dốc nhỏ nhưng nhất quán.

```python
def momentum_gradient_descent(X, y, learning_rate, num_iterations, momentum=0.9):
    theta = np.zeros(X.shape[1])
    velocity = np.zeros_like(theta)
    cost_history = []

    for _ in range(num_iterations):
        grad = gradient(X, y, theta)
        velocity = momentum * velocity - learning_rate * grad
        theta += velocity
        cost = cost_function(X, y, theta)
        cost_history.append(cost)

    return theta, cost_history

# Example usage
theta_momentum, cost_history_momentum = momentum_gradient_descent(X_with_bias, y, 0.01, 1000)
plot_convergence(cost_history_momentum)
```

Trang trình bày 14: So sánh các trình tối ưu hóa

Hãy so sánh hiệu suất của các trình tối ưu hóa khác nhau của chúng tôi trên cùng một tập dữ liệu.

```python
import time

optimizers = [
    ("Batch GD", batch_gradient_descent),
    ("Mini-Batch GD", lambda X, y, lr, iters: mini_batch_gradient_descent(X, y, lr, iters, 32)),
    ("Momentum GD", momentum_gradient_descent),
    ("Regularized GD", lambda X, y, lr, iters: batch_gradient_descent_regularized(X, y, lr, iters, 0.1))
]

results = {}

for name, optimizer in optimizers:
    start_time = time.time()
    theta, cost_history = optimizer(X_with_bias, y, 0.01, 1000)
    end_time = time.time()

    results[name] = {
        "final_cost": cost_history[-1],
        "time": end_time - start_time
    }

for name, result in results.items():
    print(f"{name}: Final Cost = {result['final_cost']:.4f}, Time = {result['time']:.2f} seconds")

# Plot convergence for all optimizers
plt.figure(figsize=(12, 8))
for name, optimizer in optimizers:
    _, cost_history = optimizer(X_with_bias, y, 0.01, 1000)
    plt.plot(range(len(cost_history)), cost_history, label=name)

plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Convergence Comparison of Different Optimizers')
plt.legend()
plt.show()
```

Trang trình bày 15: Tài nguyên bổ sung

Để hiểu sâu hơn về độ dốc giảm dần và các biến thể của nó, hãy xem xét khám phá các tài liệu học thuật sau:

1. "Tổng quan về các thuật toán tối ưu hóa giảm độ dốc" của Sebastian Ruder (2016) ArXiv: [https://arxiv.org/abs/1609.04747](https://arxiv.org/abs/1609.04747)
2. "Các phương pháp phân cấp thích ứng cho việc học trực tuyến và tối ưu hóa ngẫu nhiên" của Duchi et al. (2011) ArXiv: [https://arxiv.org/abs/1101.3618](https://arxiv.org/abs/1101.3618)
3. "Adam: Phương pháp tối ưu hóa ngẫu nhiên" của Kingma và Ba (2014) ArXiv: [https://arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980)

Các bài viết này cung cấp phân tích chuyên sâu và so sánh các thuật toán tối ưu hóa khác nhau, bao gồm các kỹ thuật nâng cao không được đề cập trong bài trình bày này.
