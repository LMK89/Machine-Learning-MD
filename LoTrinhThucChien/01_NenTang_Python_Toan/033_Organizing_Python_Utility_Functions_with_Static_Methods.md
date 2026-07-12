## Tổ chức các hàm tiện ích Python bằng các phương thức tĩnh
Slide 1: Tìm hiểu các phương thức tĩnh

Các phương thức tĩnh đóng vai trò là các hàm tiện ích thuộc về một không gian tên lớp nhưng hoạt động độc lập với trạng thái lớp hoặc thể hiện. Chúng cung cấp một cách rõ ràng để tổ chức các chức năng liên quan mà không yêu cầu tạo phiên bản, làm cho mã trở nên mô-đun hơn và dễ bảo trì hơn.

```python
class MathOperations:
    @staticmethod
    def calculate_factorial(n):
        if n == 0 or n == 1:
            return 1
        return n * MathOperations.calculate_factorial(n - 1)

# Using the static method without instantiation
result = MathOperations.calculate_factorial(5)
print(f"Factorial of 5: {result}")  # Output: Factorial of 5: 120
```

Slide 2: Comparing Instance, Class, and Static Methods

Understanding the distinctions between method types is crucial for proper implementation. Instance methods can access instance attributes, class methods can modify class state, while static methods operate independently of both instance and class state.

```python
class DataProcessor:
    data_format = "csv"  # class variable

    def __init__(self, data):
        self.data = data  # instance variable

    def process_data(self):  # instance method
        return f"Processing {self.data}"

    @classmethod
    def change_format(cls, new_format):  # class method
        cls.data_format = new_format
        return cls.data_format

    @staticmethod
    def validate_format(format_type):  # static method
        return format_type in ["csv", "json", "xml"]

# Usage demonstration
processor = DataProcessor("sample_data")
print(processor.process_data())  # Output: Processing sample_data
print(DataProcessor.change_format("json"))  # Output: json
print(DataProcessor.validate_format("yaml"))  # Output: False
```

Slide 3: Phương pháp tĩnh trong xác thực dữ liệu

Các phương thức tĩnh vượt trội trong việc thực hiện các tác vụ xác thực không yêu cầu trạng thái đối tượng. Chúng có thể được sử dụng để xác minh các tham số đầu vào, kiểm tra định dạng dữ liệu hoặc xác thực cài đặt cấu hình trước khi khởi tạo đối tượng.

```python
class InputValidator:
    @staticmethod
    def validate_email(email):
        import re
        pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
        return bool(re.match(pattern, email))

    @staticmethod
    def validate_phone(phone):
        import re
        pattern = r'^\+?1?\d{9,15}$'
        return bool(re.match(pattern, phone))

# Validation examples
print(InputValidator.validate_email("user@example.com"))  # Output: True
print(InputValidator.validate_phone("+1234567890"))  # Output: True
print(InputValidator.validate_email("invalid.email"))  # Output: False
```

Slide 4: Tính toán toán học bằng phương pháp tĩnh

Các phương thức tĩnh đặc biệt hữu ích để thực hiện các phép toán mà vẫn nhất quán trên tất cả các phiên bản của một lớp. Những phương thức này có thể được gọi trực tiếp mà không cần khởi tạo lớp.

```python
class Statistics:
    @staticmethod
    def mean(numbers):
        return sum(numbers) / len(numbers)

    @staticmethod
    def variance(numbers):
        mean = Statistics.mean(numbers)
        return sum((x - mean) ** 2 for x in numbers) / len(numbers)

    @staticmethod
    def standard_deviation(numbers):
        return Statistics.variance(numbers) ** 0.5

# Statistical calculations
data = [1, 2, 3, 4, 5]
print(f"Mean: {Statistics.mean(data):.2f}")  # Output: Mean: 3.00
print(f"Standard Deviation: {Statistics.standard_deviation(data):.2f}")  # Output: Standard Deviation: 1.41
```

Slide 5: Thao tác với file bằng phương pháp tĩnh

Các phương thức tĩnh cung cấp một cách tinh tế để xử lý các thao tác tệp không yêu cầu dữ liệu dành riêng cho phiên bản. Họ có thể gói gọn các mẫu xử lý tệp phổ biến trong khi vẫn duy trì mã sạch và có thể tái sử dụng.

```python
class FileHandler:
    @staticmethod
    def read_json(filepath):
        import json
        try:
            with open(filepath, 'r') as file:
                return json.load(file)
        except Exception as e:
            return f"Error reading file: {str(e)}"

    @staticmethod
    def write_json(data, filepath):
        import json
        try:
            with open(filepath, 'w') as file:
                json.dump(data, file, indent=4)
            return True
        except Exception as e:
            return f"Error writing file: {str(e)}"

# Example usage
data = {"name": "John", "age": 30}
FileHandler.write_json(data, "user.json")
loaded_data = FileHandler.read_json("user.json")
print(loaded_data)  # Output: {'name': 'John', 'age': 30}
```

Trang trình bày 6: Các phương thức tĩnh cho các thao tác ngày và giờ

Các phương thức tĩnh có thể xử lý hiệu quả các phép tính và chuyển đổi ngày và giờ mà không cần duy trì bất kỳ trạng thái phiên bản nào. Cách tiếp cận này đặc biệt hữu ích khi làm việc với các múi giờ và định dạng ngày khác nhau trên một ứng dụng.

```python
from datetime import datetime, timezone

class DateTimeUtil:
    @staticmethod
    def to_unix_timestamp(dt_str, format="%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(dt_str, format)
            return int(dt.timestamp())
        except ValueError as e:
            return f"Error: {str(e)}"

    @staticmethod
    def from_unix_timestamp(timestamp):
        try:
            return datetime.fromtimestamp(timestamp, tz=timezone.utc)
        except ValueError as e:
            return f"Error: {str(e)}"

# Example usage
timestamp = DateTimeUtil.to_unix_timestamp("2024-01-01 12:00:00")
print(f"Unix Timestamp: {timestamp}")  # Output: Unix Timestamp: 1704110400
print(f"DateTime: {DateTimeUtil.from_unix_timestamp(timestamp)}")
# Output: DateTime: 2024-01-01 12:00:00+00:00
```

Slide 7: Mã hóa dữ liệu bằng phương pháp tĩnh

Các phương thức tĩnh cung cấp một giao diện rõ ràng cho các hoạt động mã hóa và giải mã, giúp việc triển khai bảo mật dễ bảo trì hơn và có thể tái sử dụng trên các phần khác nhau của ứng dụng.

```python
import base64
from cryptography.fernet import Fernet

class Encryptor:
    @staticmethod
    def generate_key():
        return Fernet.generate_key()

    @staticmethod
    def encrypt_message(message: str, key: bytes) -> str:
        f = Fernet(key)
        encrypted = f.encrypt(message.encode())
        return base64.b64encode(encrypted).decode()

    @staticmethod
    def decrypt_message(encrypted_message: str, key: bytes) -> str:
        f = Fernet(key)
        decrypted = f.decrypt(base64.b64decode(encrypted_message))
        return decrypted.decode()

# Example usage
key = Encryptor.generate_key()
message = "Secret message"
encrypted = Encryptor.encrypt_message(message, key)
decrypted = Encryptor.decrypt_message(encrypted, key)
print(f"Original: {message}")  # Output: Original: Secret message
print(f"Encrypted: {encrypted}")  # Output: Encrypted: [encrypted string]
print(f"Decrypted: {decrypted}")  # Output: Decrypted: Secret message
```

Slide 8: Các phương pháp tĩnh trong xử lý ảnh

Khi xử lý các tác vụ xử lý hình ảnh không yêu cầu duy trì trạng thái giữa các hoạt động, các phương pháp tĩnh cung cấp một cách tiếp cận rõ ràng và hiệu quả để thực hiện các chức năng xử lý hình ảnh khác nhau.

```python
import numpy as np
from PIL import Image

class ImageProcessor:
    @staticmethod
    def resize_image(image_array: np.ndarray, scale_factor: float) -> np.ndarray:
        height, width = image_array.shape[:2]
        new_height = int(height * scale_factor)
        new_width = int(width * scale_factor)

        img = Image.fromarray(image_array)
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)
        return np.array(resized_img)

    @staticmethod
    def apply_grayscale(image_array: np.ndarray) -> np.ndarray:
        return np.dot(image_array[..., :3], [0.2989, 0.5870, 0.1140])

# Example usage (assuming you have an image)
# image = np.array(Image.open('image.jpg'))
# resized = ImageProcessor.resize_image(image, 0.5)
# grayscale = ImageProcessor.apply_grayscale(image)
```

Slide 9: Thao tác cơ sở dữ liệu với phương thức tĩnh

Các phương thức tĩnh vượt trội trong việc xử lý các hoạt động cơ sở dữ liệu độc lập với trạng thái phiên bản, cung cấp giao diện rõ ràng cho các tương tác cơ sở dữ liệu chung trong khi vẫn duy trì sự tách biệt các mối quan tâm.

```python
import sqlite3
from typing import List, Dict, Any

class DatabaseHandler:
    @staticmethod
    def execute_query(query: str, params: tuple = None) -> List[Dict[str, Any]]:
        try:
            with sqlite3.connect('database.db') as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)

                result = [dict(row) for row in cursor.fetchall()]
                return result
        except sqlite3.Error as e:
            return [{"error": str(e)}]

# Example usage
query = "SELECT * FROM users WHERE age > ?"
results = DatabaseHandler.execute_query(query, (25,))
print(f"Query results: {results}")
```

Trang trình bày 10: Các phương thức tĩnh trong xử lý phản hồi API

Các phương thức tĩnh cung cấp một cách tinh tế để chuẩn hóa định dạng phản hồi API và xử lý lỗi trên một ứng dụng, đảm bảo các kiểu giao tiếp nhất quán.

```python
from typing import Union, Dict, Any
import json

class APIResponseHandler:
    @staticmethod
    def success_response(data: Any, message: str = "Success") -> Dict:
        return {
            "status": "success",
            "message": message,
            "data": data,
            "error": None
        }

    @staticmethod
    def error_response(error: Union[str, Exception], code: int = 400) -> Dict:
        return {
            "status": "error",
            "message": str(error),
            "data": None,
            "error_code": code
        }

    @staticmethod
    def format_response(response: Dict) -> str:
        return json.dumps(response, indent=2)

# Example usage
data = {"user_id": 123, "name": "John Doe"}
success = APIResponseHandler.success_response(data)
error = APIResponseHandler.error_response("Invalid input", 400)
print(APIResponseHandler.format_response(success))
print(APIResponseHandler.format_response(error))
```

Slide 11: Phương pháp tĩnh cho cơ chế bộ nhớ đệm

Các phương thức tĩnh có thể triển khai các cơ chế lưu vào bộ đệm hiệu quả để duy trì trạng thái bộ đệm ở cấp lớp trong khi cung cấp giao diện rõ ràng cho các hoạt động của bộ đệm. Cách tiếp cận này tối ưu hóa hiệu suất mà không cần chi phí cụ thể cho từng phiên bản.

```python
from functools import wraps
from time import time

class CacheManager:
    _cache = {}
    _cache_expiry = {}

    @staticmethod
    def cache_with_ttl(ttl_seconds=300):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
                current_time = time()

                if key in CacheManager._cache:
                    if current_time - CacheManager._cache_expiry[key] < ttl_seconds:
                        return CacheManager._cache[key]

                result = func(*args, **kwargs)
                CacheManager._cache[key] = result
                CacheManager._cache_expiry[key] = current_time
                return result
            return wrapper
        return decorator

# Example usage
@CacheManager.cache_with_ttl(ttl_seconds=60)
def expensive_operation(x):
    import time
    time.sleep(2)  # Simulate expensive operation
    return x * x

print(expensive_operation(5))  # Takes 2 seconds
print(expensive_operation(5))  # Instant (cached)
```

Slide 12: Các phương pháp tĩnh trong triển khai mạng nơ-ron

Các phương pháp tĩnh xử lý hiệu quả các tính toán mạng thần kinh, cung cấp giao diện rõ ràng cho các chức năng kích hoạt và tính toán tổn thất mà vẫn nhất quán trên các kiến ​​trúc mạng khác nhau.

```python
import numpy as np

class NeuralNetworkUtils:
    @staticmethod
    def sigmoid(x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        """Derivative of sigmoid function"""
        sx = NeuralNetworkUtils.sigmoid(x)
        return sx * (1 - sx)

    @staticmethod
    def categorical_cross_entropy(y_true, y_pred):
        """Calculate categorical cross-entropy loss"""
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

# Example usage
x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
print(f"Sigmoid output: {NeuralNetworkUtils.sigmoid(x)}")
print(f"Sigmoid derivative: {NeuralNetworkUtils.sigmoid_derivative(x)}")

y_true = np.array([[1, 0, 0], [0, 1, 0]])
y_pred = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]])
print(f"Cross-entropy loss: {NeuralNetworkUtils.categorical_cross_entropy(y_true, y_pred)}")
```

Trang trình bày 13: Kết quả và phân tích hiệu suất

Phần giới thiệu triển khai này thể hiện những lợi ích thực tế của các phương pháp tĩnh trong các tình huống thực tế, từ việc tổ chức mã được cải thiện đến tối ưu hóa hiệu suất.

```python
import time
import statistics

class PerformanceMetrics:
    @staticmethod
    def measure_execution_time(func, *args, iterations=1000):
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            func(*args)
            end = time.perf_counter()
            times.append(end - start)

        return {
            'mean': statistics.mean(times),
            'median': statistics.median(times),
            'std_dev': statistics.stdev(times)
        }

# Performance comparison example
def regular_function(x):
    return x * x

class MathOps:
    @staticmethod
    def static_square(x):
        return x * x

# Measure performance
regular_metrics = PerformanceMetrics.measure_execution_time(regular_function, 5)
static_metrics = PerformanceMetrics.measure_execution_time(MathOps.static_square, 5)

print("Regular Function Metrics:", regular_metrics)
print("Static Method Metrics:", static_metrics)
```

Trang trình bày 14: Tài nguyên bổ sung

* Python hiệu quả: 90 cách cụ thể để viết Python tốt hơn [https://www.google.com/search?q=effect+python+90+spec+ways+to+write+better+python](https://www.google.com/search?q=effect+python+90+spec+ways+to+write+better+python)
* Mẫu thiết kế Python: Dành cho mã đẹp mắt và bền vững [https://www.google.com/search?q=python+design+patterns+book](https://www.google.com/search?q=python+design+patterns+book)
* Lập trình Python nâng cao: Các phương pháp thực hành và mẫu thiết kế tốt nhất [https://arxiv.org/abs/cs.SE/2103.11928](https://arxiv.org/abs/cs.SE/2103.11928)
* Mã sạch trong Python: Nguyên tắc tái cấu trúc [https://www.google.com/search?q=clean+code+python+best+practices](https://www.google.com/search?q=clean+code+python+best+practices)
* Phương thức tĩnh và kế thừa trong lập trình hướng đối tượng [https://www.google.com/search?q=static+methods+inheritance+python+research](https://www.google.com/search?q=static+methods+inheritance+python+research)
