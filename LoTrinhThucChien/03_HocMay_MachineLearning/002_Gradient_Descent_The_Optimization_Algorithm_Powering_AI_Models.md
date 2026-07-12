## Giảm dần độ dốc Thuật toán tối ưu hóa hỗ trợ các mô hình AI
Slide 1: Giới thiệu về Giảm dần độ dốc

gradient Descent là một thuật toán tối ưu hóa cơ bản trong học máy và trí tuệ nhân tạo. Nó được sử dụng để giảm thiểu hàm số bằng cách di chuyển lặp đi lặp lại theo hướng đi xuống dốc nhất. Trong AI, nó là xương sống của việc đào tạo mạng lưới thần kinh và các mô hình khác.

```python
import matplotlib.pyplot as plt
import numpy as np

def f(x):
    return x**2 + 5*np.sin(x)

x = np.linspace(-10, 10, 100)
y = f(x)

plt.plot(x, y)
plt.title('Function to Optimize')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()
```

Trang trình bày 2: Độ dốc

Độ dốc là một vectơ đạo hàm riêng hướng về hướng đi lên dốc nhất. Để thu nhỏ một hàm, chúng ta di chuyển theo hướng ngược lại của gradient.

```python
def gradient(x):
    return 2*x + 5*np.cos(x)

x = np.linspace(-10, 10, 100)
grad = gradient(x)

plt.plot(x, grad)
plt.title('Gradient of the Function')
plt.xlabel('x')
plt.ylabel('Gradient')
plt.axhline(y=0, color='r', linestyle='--')
plt.show()
```

Trang trình bày 3: Quy trình tối ưu hóa

Giảm dần độ dốc cập nhật lặp đi lặp lại các tham số bằng cách trừ độ dốc nhân với tốc độ học. Quá trình này tiếp tục cho đến khi đạt được sự hội tụ hoặc đạt đến số lần lặp tối đa.

```python
def gradient_descent(start, learn_rate, n_iter):
    x = start
    for i in range(n_iter):
        grad = gradient(x)
        x = x - learn_rate * grad
    return x

result = gradient_descent(start=5, learn_rate=0.1, n_iter=100)
print(f"Optimized x: {result}")
print(f"Optimized f(x): {f(result)}")
```

Slide 4: Tỷ lệ học tập

Tốc độ học tập là một siêu tham số quan trọng trong gradient Descent. Nó xác định kích thước bước ở mỗi lần lặp. Nếu nó quá nhỏ thì sự hội tụ sẽ chậm. Nếu nó quá lớn, thuật toán có thể vượt quá mức tối thiểu.

```python
learning_rates = [0.01, 0.1, 0.5]
colors = ['r', 'g', 'b']

for lr, c in zip(learning_rates, colors):
    x = 5
    path = [x]
    for _ in range(20):
        x = x - lr * gradient(x)
        path.append(x)
    plt.plot(path, f(np.array(path)), c=c, label=f'LR = {lr}')

plt.legend()
plt.title('Effect of Learning Rate')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()
```

Trang trình bày 5: Tối thiểu cục bộ và toàn cầu

Độ dốc giảm dần có thể bị kẹt ở cực tiểu cục bộ, đặc biệt đối với các hàm không lồi. Đây là lý do tại sao việc khởi tạo và các kỹ thuật khác như động lượng lại quan trọng.

```python
def complex_function(x):
    return x**4 - 4*x**3 - 2*x**2 + 12*x

x = np.linspace(-2, 3, 100)
y = complex_function(x)

plt.plot(x, y)
plt.title('Function with Multiple Minima')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()
```

Trang trình bày 6: Giảm dần độ dốc ngẫu nhiên

Trong thực tế, đặc biệt đối với các tập dữ liệu lớn, chúng ta thường sử dụng Stochastic gradient Descent (SGD). SGD tính toán độ dốc chỉ bằng cách sử dụng một tập hợp con nhỏ (lô nhỏ) dữ liệu ở mỗi lần lặp.

```python
def sgd(X, y, learning_rate, n_epochs):
    w = np.zeros(X.shape[1])
    for epoch in range(n_epochs):
        for i in range(X.shape[0]):
            gradient = 2 * X[i] * (np.dot(X[i], w) - y[i])
            w -= learning_rate * gradient
    return w

# Example usage
X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([5, 11, 17])
w = sgd(X, y, learning_rate=0.01, n_epochs=1000)
print("Optimized weights:", w)
```

Trang trình bày 7: Động lực

Động lượng là một kỹ thuật để tăng tốc Độ dốc giảm dần bằng cách thêm một phần của bản cập nhật trước đó vào bản cập nhật hiện tại. Điều này giúp vượt qua các mức tối thiểu và cao nguyên cục bộ.

```python
def momentum_gd(gradient, start, learn_rate, momentum, n_iter):
    x = start
    v = 0
    for _ in range(n_iter):
        grad = gradient(x)
        v = momentum * v - learn_rate * grad
        x += v
    return x

result = momentum_gd(gradient, start=5, learn_rate=0.1, momentum=0.9, n_iter=100)
print(f"Optimized x: {result}")
print(f"Optimized f(x): {f(result)}")
```

Trang trình bày 8: Tỷ lệ học tập thích ứng

Các thuật toán như AdaGrad, RMSProp và Adam điều chỉnh tốc độ học cho từng tham số. Điều này có thể dẫn đến sự hội tụ nhanh hơn và hiệu suất tốt hơn.

```python
def adagrad(gradient, start, learn_rate, n_iter):
    x = start
    g_sum = 0
    for _ in range(n_iter):
        grad = gradient(x)
        g_sum += grad**2
        x -= (learn_rate / (np.sqrt(g_sum) + 1e-8)) * grad
    return x

result = adagrad(gradient, start=5, learn_rate=0.1, n_iter=100)
print(f"Optimized x: {result}")
print(f"Optimized f(x): {f(result)}")
```

Trang trình bày 9: Giảm dần độ dốc trong Mạng thần kinh

Trong mạng lưới thần kinh, gradient Descent được sử dụng để cập nhật trọng số và độ lệch. Lan truyền ngược được sử dụng để tính toán độ dốc một cách hiệu quả.

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def nn_forward(X, W1, W2):
    Z1 = np.dot(X, W1)
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2)
    A2 = sigmoid(Z2)
    return A1, A2

def nn_backward(X, y, A1, A2, W2):
    m = X.shape[0]
    dZ2 = A2 - y
    dW2 = np.dot(A1.T, dZ2) / m
    dZ1 = np.dot(dZ2, W2.T) * A1 * (1 - A1)
    dW1 = np.dot(X.T, dZ1) / m
    return dW1, dW2

# Example usage
X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
y = np.array([[0], [1], [1], [0]])
W1 = np.random.randn(3, 4)
W2 = np.random.randn(4, 1)

for _ in range(10000):
    A1, A2 = nn_forward(X, W1, W2)
    dW1, dW2 = nn_backward(X, y, A1, A2, W2)
    W1 -= 0.1 * dW1
    W2 -= 0.1 * dW2

print("Final predictions:", nn_forward(X, W1, W2)[1])
```

Slide 10: Thực hành giảm độ dốc

Trong các ứng dụng trong thế giới thực, gradient Descent được sử dụng trong nhiều tác vụ học máy khác nhau, chẳng hạn như hệ thống đề xuất đào tạo, mô hình xử lý ngôn ngữ tự nhiên và thuật toán thị giác máy tính.

```python
import numpy as np

# Simple linear regression using gradient descent
def linear_regression_gd(X, y, learning_rate, n_iterations):
    m, n = X.shape
    theta = np.zeros(n)

    for _ in range(n_iterations):
        h = np.dot(X, theta)
        gradient = np.dot(X.T, (h - y)) / m
        theta -= learning_rate * gradient

    return theta

# Generate sample data
np.random.seed(42)
X = np.column_stack((np.ones(100), np.random.rand(100, 1)))
y = 2 + 3 * X[:, 1] + np.random.randn(100) * 0.1

# Train the model
theta = linear_regression_gd(X, y, learning_rate=0.1, n_iterations=1000)

print("Estimated coefficients:", theta)
```

Slide 11: Những thách thức và hạn chế

Mặc dù gradient Descent rất mạnh mẽ nhưng nó phải đối mặt với những thách thức như hội tụ chậm đối với các vấn đề không có điều kiện, khó khăn với các điểm yên ngựa và độ nhạy đối với việc chia tỷ lệ các biến đầu vào.

```python
import numpy as np
import matplotlib.pyplot as plt

def rosenbrock(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2

x = np.linspace(-2, 2, 100)
y = np.linspace(-1, 3, 100)
X, Y = np.meshgrid(x, y)
Z = rosenbrock(X, Y)

plt.contour(X, Y, Z, levels=np.logspace(-1, 3, 20))
plt.colorbar()
plt.title('Rosenbrock Function - A Challenging Optimization Landscape')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
```

Trang trình bày 12: Vượt quá độ dốc Vanilla

Các kỹ thuật nâng cao như gradient liên hợp, phương pháp Quasi-Newton (ví dụ: BFGS) và tối ưu hóa không có Hessian đôi khi có thể hoạt động tốt hơn gradient Descent tiêu chuẩn.

```python
from scipy.optimize import minimize

def rosenbrock(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

# Using BFGS algorithm
result = minimize(rosenbrock, [0, 0], method='BFGS')

print("Optimized solution:", result.x)
print("Optimized value:", result.fun)
```

Slide 13: Ví dụ thực tế: Phân loại hình ảnh

Giảm dần độ dốc rất quan trọng trong việc đào tạo mạng thần kinh tích chập (CNN) cho các tác vụ phân loại hình ảnh, chẳng hạn như xác định đối tượng trong ảnh hoặc nhận dạng chữ viết tay.

```python
import numpy as np

def conv2d(image, kernel):
    h, w = image.shape
    k_h, k_w = kernel.shape
    output = np.zeros((h-k_h+1, w-k_w+1))
    for i in range(h-k_h+1):
        for j in range(w-k_w+1):
            output[i,j] = np.sum(image[i:i+k_h, j:j+k_w] * kernel)
    return output

# Simple edge detection kernel
kernel = np.array([[-1, -1, -1],
                   [-1,  8, -1],
                   [-1, -1, -1]])

# Example image (5x5 grayscale)
image = np.array([[0, 0, 0, 0, 0],
                  [0, 1, 1, 1, 0],
                  [0, 1, 1, 1, 0],
                  [0, 1, 1, 1, 0],
                  [0, 0, 0, 0, 0]])

result = conv2d(image, kernel)
print("Convolution result (edge detection):")
print(result)
```

Slide 14: Ví dụ thực tế: Xử lý ngôn ngữ tự nhiên

Trong các tác vụ NLP như phân tích tình cảm hoặc dịch ngôn ngữ, gradient Descent tối ưu hóa các tham số của mạng thần kinh tái phát (RNN) hoặc máy biến áp để nắm bắt các mẫu ngôn ngữ phức tạp.

```python
import numpy as np

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum(axis=0)

def simple_rnn_step(x, h, W_hh, W_xh, W_hy):
    h_next = np.tanh(np.dot(W_hh, h) + np.dot(W_xh, x))
    y = softmax(np.dot(W_hy, h_next))
    return h_next, y

# Example usage
vocab_size = 1000
hidden_size = 100
output_size = 5  # e.g., 5 sentiment classes

# Initialize weights randomly
W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
W_xh = np.random.randn(hidden_size, vocab_size) * 0.01
W_hy = np.random.randn(output_size, hidden_size) * 0.01

# Example input (one-hot encoded word)
x = np.zeros(vocab_size)
x[42] = 1  # Assuming word index 42

h = np.zeros(hidden_size)
h_next, y = simple_rnn_step(x, h, W_hh, W_xh, W_hy)

print("Predicted sentiment probabilities:", y)
```

Trang trình bày 15: Tài nguyên bổ sung

Đối với những người muốn tìm hiểu sâu hơn về các kỹ thuật tối ưu hóa và Giảm dần độ dốc, đây là một số tài nguyên có giá trị:

1. "Phương pháp tối ưu hóa cho học máy quy mô lớn" của Bottou et al. (2018) ArXiv: [https://arxiv.org/abs/1606.04838](https://arxiv.org/abs/1606.04838)
2. "Tổng quan về các thuật toán tối ưu hóa giảm dần độ dốc" của Ruder (2016) ArXiv: [https://arxiv.org/abs/1609.04747](https://arxiv.org/abs/1609.04747)
3. "Adam: Phương pháp tối ưu hóa ngẫu nhiên" của Kingma và Ba (2014) ArXiv: [https://arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980)

Các bài viết này cung cấp cái nhìn tổng quan toàn diện và phân tích chuyên sâu về các thuật toán Giảm dần độ dốc khác nhau và các ứng dụng của chúng trong học máy.
