## Cách đặt tên tốt nhất trong Python
Slide 1: Tên biến rõ ràng và mang tính mô tả

Tên biến tốt phải mang tính mô tả, có ý nghĩa và tuân theo quy ước đặt tên của Python. Chúng phải chỉ rõ mục đích và nội dung của biến trong khi vẫn duy trì khả năng đọc được.

```python
# Bad naming
x = ['apple', 'banana', 'orange']
n = len(x)
r = []
for i in range(n):
    r.append(x[i].upper())

# Good naming
fruits = ['apple', 'banana', 'orange']
fruits_count = len(fruits)
uppercase_fruits = []
for index in range(fruits_count):
    uppercase_fruits.append(fruits[index].upper())

print(uppercase_fruits)  # Output: ['APPLE', 'BANANA', 'ORANGE']
```

Slide 2: Mẫu đặt tên hàm

Tên hàm phải hướng đến hành động, sử dụng động từ để mô tả mục đích của chúng. Chúng phải tuân theo quy ước snake\_case và chỉ rõ việc chuyển đổi hoặc tính toán đang được thực hiện.

```python
# Bad naming
def proc(lst):
    return [x for x in lst if x > 0]

# Good naming
def filter_positive_numbers(number_list):
    return [num for num in number_list if num > 0]

numbers = [-2, 0, 3, -1, 5]
result = filter_positive_numbers(numbers)
print(result)  # Output: [3, 5]
```

Slide 3: Quy ước đặt tên lớp

Các lớp đại diện cho các đối tượng và nên sử dụng quy ước đặt tên PascalCase. Tên phải là một danh từ mô tả rõ ràng thực thể được mô hình hóa, với tên thuộc tính rõ ràng và mang tính mô tả.

```python
# Bad naming
class calc:
    def __init__(self, x, y):
        self.a = x
        self.b = y

    def p(self):
        return self.a + self.b

# Good naming
class Calculator:
    def __init__(self, first_number, second_number):
        self.first_number = first_number
        self.second_number = second_number

    def calculate_sum(self):
        return self.first_number + self.second_number

calc = Calculator(5, 3)
print(calc.calculate_sum())  # Output: 8
```

Trang trình bày 4: Các hằng số và biến cấp mô-đun

Các hằng số nên sử dụng chữ in hoa có dấu gạch dưới, được đặt ở cấp độ mô-đun. Tên của chúng phải chỉ rõ mục đích và bối cảnh sử dụng.

```python
# Bad naming
x = 3.14159
max_s = 100
def_timeout = 30

# Good naming
PI = 3.14159
MAX_CONNECTIONS = 100
DEFAULT_TIMEOUT_SECONDS = 30

def calculate_circle_area(radius):
    return PI * radius ** 2

print(calculate_circle_area(5))  # Output: 78.53975
```

Slide 5: Đặt tên biến Iterator và vòng lặp

Khi làm việc với các trình vòng lặp và vòng lặp, hãy sử dụng các tên có ý nghĩa đại diện cho các phần tử riêng lẻ đang được xử lý, tránh các biến có một chữ cái ngoại trừ các phép toán đơn giản.

```python
# Bad naming
for x in range(len(l)):
    for y in l[x]:
        print(y)

# Good naming
matrix = [[1, 2, 3], [4, 5, 6]]
for row_index in range(len(matrix)):
    for element in matrix[row_index]:
        print(f"Processing element: {element}")
```

Trang trình bày 6: Ví dụ thực tế - Đường ống xử lý dữ liệu

Một ví dụ thực tế thể hiện quy ước đặt tên trong quy trình xử lý dữ liệu để phân tích giao dịch của khách hàng.

```python
class TransactionProcessor:
    def __init__(self, transaction_data):
        self.transaction_data = transaction_data
        self.processed_transactions = []

    def filter_valid_transactions(self, minimum_amount=0):
        return [
            transaction for transaction in self.transaction_data
            if transaction['amount'] > minimum_amount
        ]

    def calculate_total_revenue(self, transactions):
        return sum(transaction['amount'] for transaction in transactions)

# Example usage
daily_transactions = [
    {'id': 1, 'amount': 100.0, 'customer': 'John'},
    {'id': 2, 'amount': -50.0, 'customer': 'Alice'},
    {'id': 3, 'amount': 75.0, 'customer': 'Bob'}
]

processor = TransactionProcessor(daily_transactions)
valid_transactions = processor.filter_valid_transactions()
total_revenue = processor.calculate_total_revenue(valid_transactions)
print(f"Total Revenue: ${total_revenue}")  # Output: Total Revenue: $175.0
```

Trang trình bày 7: Đặt tên trong bối cảnh học máy

Tên biến và hàm học máy phải phản ánh các khái niệm toán học trong khi vẫn có thể đọc được. Các quy ước chung bao gồm sử dụng chữ thường cho vectơ và chữ hoa cho ma trận.

```python
import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Weight matrices use uppercase, biases lowercase
        self.W1 = np.random.randn(input_size, hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b1 = np.zeros((1, hidden_size))
        self.b2 = np.zeros((1, output_size))

    def forward_propagation(self, input_data):
        self.hidden_layer = np.tanh(np.dot(input_data, self.W1) + self.b1)
        self.output_layer = np.dot(self.hidden_layer, self.W2) + self.b2
        return self.output_layer

# Example usage
model = NeuralNetwork(3, 4, 2)
sample_input = np.array([[1.0, 0.5, -0.2]])
prediction = model.forward_propagation(sample_input)
```

Trang trình bày 8: Đặt tên theo ngữ nghĩa trong cấu trúc dữ liệu

Khi triển khai cấu trúc dữ liệu, tên phải phản ánh mục đích và hành vi của cấu trúc, giúp mã tự ghi lại và dễ bảo trì hơn.

```python
class BinarySearchTree:
    def __init__(self, value):
        self.value = value
        self.left_child = None
        self.right_child = None

    def insert_node(self, new_value):
        if new_value < self.value:
            if self.left_child is None:
                self.left_child = BinarySearchTree(new_value)
            else:
                self.left_child.insert_node(new_value)
        else:
            if self.right_child is None:
                self.right_child = BinarySearchTree(new_value)
            else:
                self.right_child.insert_node(new_value)

# Usage example
root = BinarySearchTree(10)
root.insert_node(5)
root.insert_node(15)
```

Trang trình bày 9: Ví dụ thực tế - Phân tích chuỗi thời gian

Một ví dụ toàn diện hiển thị các quy ước đặt tên trong quá trình triển khai phân tích và xử lý dữ liệu chuỗi thời gian.

```python
class TimeSeriesAnalyzer:
    def __init__(self, time_series_data):
        self.raw_data = time_series_data
        self.processed_data = None
        self.seasonal_components = None

    def remove_outliers(self, window_size=5, threshold=2):
        rolling_mean = np.mean(self.raw_data)
        rolling_std = np.std(self.raw_data)
        z_scores = np.abs((self.raw_data - rolling_mean) / rolling_std)
        self.processed_data = self.raw_data[z_scores < threshold]

    def extract_seasonality(self, period_length):
        if self.processed_data is None:
            self.processed_data = self.raw_data
        self.seasonal_components = np.array([
            np.mean(self.processed_data[i::period_length])
            for i in range(period_length)
        ])
        return self.seasonal_components

# Example usage
monthly_temperatures = np.array([20, 22, 25, 28, 30, 32, 31, 29, 26, 23, 21, 19])
analyzer = TimeSeriesAnalyzer(monthly_temperatures)
seasonal_pattern = analyzer.extract_seasonality(period_length=12)
```

Trang trình chiếu 10: Xử lý ngoại lệ và thông báo lỗi

Thông báo lỗi và tên ngoại lệ phải rõ ràng, cụ thể và cung cấp thông tin có thể thực hiện được. Tên phải chỉ ra loại lỗi và bối cảnh của nó.

```python
class DatabaseConnectionError(Exception):
    """Custom exception for database connection failures"""
    pass

class DataValidator:
    def __init__(self, connection_string):
        self.connection_string = connection_string

    def validate_user_input(self, user_data):
        required_fields = {'username', 'email', 'age'}
        missing_fields = required_fields - set(user_data.keys())

        if missing_fields:
            raise ValueError(
                f"Missing required fields: {', '.join(missing_fields)}"
            )

        if not isinstance(user_data['age'], int):
            raise TypeError(
                f"Age must be an integer, got {type(user_data['age']).__name__}"
            )

# Example usage
validator = DataValidator("postgresql://localhost:5432/mydb")
try:
    validator.validate_user_input({'username': 'john', 'email': 'john@example.com'})
except ValueError as error:
    print(f"Validation error: {error}")
```

Slide 11: Thiết kế giao diện và tên phương thức

Tên phương thức trong giao diện và lớp trừu tượng phải truyền đạt rõ ràng hợp đồng và hành vi dự kiến ​​của chúng, tập trung vào hành động đang được thực hiện.

```python
from abc import ABC, abstractmethod

class DataProcessor(ABC):
    @abstractmethod
    def preprocess_raw_data(self, data):
        """Transform raw data into processable format"""
        pass

    @abstractmethod
    def extract_features(self, preprocessed_data):
        """Extract relevant features from preprocessed data"""
        pass

class ImageProcessor(DataProcessor):
    def preprocess_raw_data(self, image_data):
        self.normalized_image = image_data / 255.0
        return self.normalized_image

    def extract_features(self, preprocessed_data):
        # Example feature extraction
        return {
            'mean_intensity': preprocessed_data.mean(),
            'std_intensity': preprocessed_data.std()
        }

processor = ImageProcessor()
sample_data = np.random.rand(28, 28)
features = processor.extract_features(processor.preprocess_raw_data(sample_data))
```

Trang trình bày 12: Không gian tên và Tổ chức mô-đun

Các không gian tên và tên mô-đun có cấu trúc tốt giúp tổ chức mã một cách hợp lý và ngăn ngừa xung đột đặt tên trong khi vẫn duy trì các mối phụ thuộc rõ ràng.

```python
# file: data_processing/transformers.py
class DataTransformer:
    def __init__(self, transformation_type):
        self.transformation_type = transformation_type

    def transform(self, data):
        if self.transformation_type == 'normalize':
            return (data - data.mean()) / data.std()
        elif self.transformation_type == 'standardize':
            return (data - data.min()) / (data.max() - data.min())

# file: data_processing/validators.py
class DataValidator:
    @staticmethod
    def check_missing_values(data):
        return data.isnull().sum()

# Usage
from data_processing.transformers import DataTransformer
from data_processing.validators import DataValidator

transformer = DataTransformer('normalize')
validator = DataValidator()
```

Slide 13: Triển khai toán học và thuật toán

Tên hàm toán học phải cân bằng giữa tính rõ ràng với quy ước toán học, sử dụng tên mô tả để biết chi tiết triển khai trong khi vẫn giữ nguyên ký hiệu chuẩn.

```python
import numpy as np

def calculate_matrix_operation(X, learning_rate=0.01):
    """
    Implements the following operation:
    $$H = \tanh(XW + b)$$
    $$Y = \text{softmax}(H)$$
    """
    num_samples, num_features = X.shape

    # Initialize weights and bias
    weights = np.random.randn(num_features, num_features)
    bias = np.zeros(num_features)

    # Forward pass
    hidden_layer = np.tanh(np.dot(X, weights) + bias)

    # Softmax calculation
    exp_scores = np.exp(hidden_layer)
    probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    return probabilities

# Example usage
input_data = np.random.randn(10, 5)
output_probabilities = calculate_matrix_operation(input_data)
```

Trang trình bày 14: Tài nguyên bổ sung

* Quy ước đặt tên học máy: [https://arxiv.org/abs/2004.08900](https://arxiv.org/abs/2004.08900)
* Mẫu mã sạch: [https://arxiv.org/abs/1909.08593](https://arxiv.org/abs/1909.08593)
* Đặt tên kiến trúc mạng thần kinh: [https://arxiv.org/abs/2006.12672](https://arxiv.org/abs/2006.12672)
* Các phương pháp hay nhất cho cụm từ tìm kiếm:
    * "Các phương pháp hay nhất về quy ước đặt tên Python"
    * "Mẫu đặt tên mã sạch"
    * "Hướng dẫn đặt tên kỹ thuật phần mềm"
