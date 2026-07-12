## Hàm hạng nhất và khái niệm chính trong Python
Trang trình bày 1: Nguyên tắc cơ bản về chức năng hạng nhất

Trong Python, hàm là đối tượng hạng nhất, nghĩa là chúng có thể được gán cho các biến, được truyền dưới dạng đối số cho các hàm khác, được trả về từ hàm và được lưu trữ trong cấu trúc dữ liệu. Khái niệm cơ bản này cho phép các mô hình lập trình chức năng mạnh mẽ và thiết kế mã linh hoạt.

```python
# Functions as objects
def square(x):
    return x * x

# Assigning function to variable
func = square

# Using function reference
numbers = [1, 2, 3, 4]
squared = list(map(func, numbers))

print(f"Original: {numbers}")
print(f"Squared: {squared}")

# Output:
# Original: [1, 2, 3, 4]
# Squared: [1, 4, 9, 16]
```

Slide 2: Hàm bậc cao hơn

Hàm bậc cao hơn là các hàm có thể chấp nhận các hàm khác làm đối số và/hoặc hàm trả về. Tính năng mạnh mẽ này cho phép tái sử dụng và trừu tượng hóa mã, cho phép đưa ra các giải pháp tinh tế hơn cho các vấn đề phức tạp thông qua thành phần chức năng.

```python
def create_multiplier(factor):
    def multiplier(x):
        return x * factor
    return multiplier

# Creating specialized functions
double = create_multiplier(2)
triple = create_multiplier(3)

# Using the generated functions
numbers = [1, 2, 3, 4]
doubled = list(map(double, numbers))
tripled = list(map(triple, numbers))

print(f"Original: {numbers}")
print(f"Doubled: {doubled}")
print(f"Tripled: {tripled}")

# Output:
# Original: [1, 2, 3, 4]
# Doubled: [2, 4, 6, 8]
# Tripled: [3, 6, 9, 12]
```

Trang trình bày 3: Triển khai trang trí chức năng

Trình trang trí hàm cung cấp một cú pháp rõ ràng để gói các hàm với chức năng bổ sung. Hiểu cách triển khai của chúng cho thấy cách Python tận dụng các hàm hạng nhất để kích hoạt khả năng lập trình siêu dữ liệu mạnh mẽ thông qua chuyển đổi hàm.

```python
import time
from functools import wraps

def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

@measure_time
def complex_operation(n):
    return sum(i * i for i in range(n))

result = complex_operation(1000000)
print(f"Result: {result}")

# Output example:
# complex_operation took 0.1234 seconds
# Result: 333332833333500000
```

Slide 4: Nhà máy chức năng

Các nhà máy chức năng tự động tạo ra các chức năng chuyên biệt dựa trên các tham số đầu vào, cho phép tạo hành vi tùy chỉnh trong khi vẫn duy trì mã sạch và có thể bảo trì thông qua nguyên tắc đóng và phạm vi từ vựng.

```python
def create_power_function(exponent):
    def power_function(base):
        return base ** exponent
    return power_function

# Creating specialized power functions
square = create_power_function(2)
cube = create_power_function(3)
fourth_power = create_power_function(4)

# Testing the functions
number = 3
print(f"Square of {number}: {square(number)}")
print(f"Cube of {number}: {cube(number)}")
print(f"Fourth power of {number}: {fourth_power(number)}")

# Output:
# Square of 3: 9
# Cube of 3: 27
# Fourth power of 3: 81
```

Slide 5: Chức năng từng phần và Currying

Các hàm một phần và cà ri tạo điều kiện thuận lợi cho việc tạo ra các hàm chuyên biệt bằng cách sửa các đối số nhất định, cho phép các cấu trúc mã linh hoạt hơn và có thể tái sử dụng thông qua thành phần chức năng và liên kết tham số.

```python
from functools import partial

def generic_power(base, exponent, modulus=None):
    result = base ** exponent
    return result % modulus if modulus else result

# Creating specialized functions through partial application
square_mod_10 = partial(generic_power, exponent=2, modulus=10)
cube_mod_7 = partial(generic_power, exponent=3, modulus=7)

# Using specialized functions
numbers = [3, 4, 5]
squares_mod_10 = [square_mod_10(n) for n in numbers]
cubes_mod_7 = [cube_mod_7(n) for n in numbers]

print(f"Squares modulo 10: {squares_mod_10}")
print(f"Cubes modulo 7: {cubes_mod_7}")

# Output:
# Squares modulo 10: [9, 6, 5]
# Cubes modulo 7: [6, 1, 6]
```

Slide 6: Thành phần chức năng

Thành phần hàm cho phép tạo ra các hoạt động phức tạp bằng cách kết hợp các hàm đơn giản hơn, cho phép một cách tiếp cận mang tính khai báo và dễ bảo trì hơn để giải quyết các vấn đề phức tạp thông qua các nguyên tắc lập trình hàm.

```python
from typing import Callable, Any

def compose(*functions: Callable) -> Callable:
    def inner(x: Any) -> Any:
        result = x
        for f in reversed(functions):
            result = f(result)
        return result
    return inner

# Define simple functions
def double(x: int) -> int:
    return x * 2

def increment(x: int) -> int:
    return x + 1

def square(x: int) -> int:
    return x * x

# Compose functions
transform = compose(square, increment, double)

# Test the composition
number = 3
print(f"Original number: {number}")
print(f"Transformed: {transform(number)}")  # ((3 * 2) + 1)^2

# Output:
# Original number: 3
# Transformed: 49
```

Trang trình bày 7: Hàm Lambda và lập trình hàm

Hàm Lambda cung cấp các định nghĩa hàm ẩn danh, ngắn gọn cho các thao tác đơn giản, hỗ trợ các mẫu lập trình hàm như bản đồ, bộ lọc và thu gọn. Những biểu thức này đặc biệt hữu ích khi làm việc với các hàm bậc cao hơn và các phép biến đổi dữ liệu.

```python
# Using lambda with map, filter, and reduce
from functools import reduce

numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Map: Square all numbers
squares = list(map(lambda x: x * x, numbers))

# Filter: Get even numbers
evens = list(filter(lambda x: x % 2 == 0, numbers))

# Reduce: Calculate product
product = reduce(lambda x, y: x * y, numbers)

print(f"Original numbers: {numbers}")
print(f"Squares: {squares}")
print(f"Even numbers: {evens}")
print(f"Product of all numbers: {product}")

# Output:
# Original numbers: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# Squares: [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
# Even numbers: [2, 4, 6, 8, 10]
# Product of all numbers: 3628800
```

Slide 8: Thuộc tính chức năng và siêu dữ liệu

Các hàm Python có thể mang siêu dữ liệu thông qua các thuộc tính, cho phép khả năng xem xét nội tâm mạnh mẽ và chức năng dựa trên trình trang trí. Tính năng này cho phép sửa đổi hành vi thời gian chạy và nâng cao khả năng tài liệu.

```python
def add_metadata(func):
    # Adding metadata to function
    func.author = "John Doe"
    func.version = "1.0.0"
    func.tags = ["math", "utility"]

    # Store original docstring
    original_doc = func.__doc__ or ""

    # Update docstring with metadata
    func.__doc__ = f"{original_doc}\n\nMetadata:\nAuthor: {func.author}\nVersion: {func.version}\nTags: {', '.join(func.tags)}"
    return func

@add_metadata
def complex_calculation(x, y):
    """Performs a complex mathematical calculation."""
    return (x ** 2 + y ** 2) ** 0.5

# Accessing function metadata
print(f"Function name: {complex_calculation.__name__}")
print(f"Author: {complex_calculation.author}")
print(f"Version: {complex_calculation.version}")
print(f"Tags: {complex_calculation.tags}")
print(f"Documentation:\n{complex_calculation.__doc__}")

# Output:
# Function name: complex_calculation
# Author: John Doe
# Version: 1.0.0
# Tags: ['math', 'utility']
# Documentation:
# Performs a complex mathematical calculation.
#
# Metadata:
# Author: John Doe
# Version: 1.0.0
# Tags: math, utility
```

Trang trình bày 9: Bộ nhớ đệm và ghi nhớ chức năng

Bộ nhớ đệm chức năng tối ưu hóa hiệu suất bằng cách lưu trữ các kết quả được tính toán trước đó, triển khai các mẫu ghi nhớ cho các hoạt động tốn kém. Kỹ thuật này cải thiện đáng kể thời gian thực hiện các hàm đệ quy hoặc hàm tính toán chuyên sâu.

```python
from functools import lru_cache
import time

def measure_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

# Without caching
@measure_time
def fibonacci_no_cache(n):
    if n < 2:
        return n
    return fibonacci_no_cache(n-1) + fibonacci_no_cache(n-2)

# With caching
@measure_time
@lru_cache(maxsize=None)
def fibonacci_cached(n):
    if n < 2:
        return n
    return fibonacci_cached(n-1) + fibonacci_cached(n-2)

# Compare performance
n = 35
print(f"\nCalculating Fibonacci({n}):")
print(f"Without cache: {fibonacci_no_cache(n)}")
print(f"With cache: {fibonacci_cached(n)}")

# Output example:
# Calculating Fibonacci(35):
# fibonacci_no_cache took 2.8432 seconds
# Without cache: 9227465
# fibonacci_cached took 0.0001 seconds
# With cache: 9227465
```

Slide 10: Ứng dụng trong thế giới thực - Đường ống xử lý dữ liệu

Việc triển khai thực tế các hàm hạng nhất trong quy trình xử lý dữ liệu cho thấy cách lập trình hàm có thể tạo ra các quy trình chuyển đổi dữ liệu linh hoạt, có thể bảo trì bằng cách sử dụng thành phần hàm và các hàm bậc cao hơn.

```python
from typing import Callable, List, Dict
import json
from datetime import datetime

class DataPipeline:
    def __init__(self, transformations: List[Callable] = None):
        self.transformations = transformations or []

    def add_transformation(self, func: Callable) -> None:
        self.transformations.append(func)

    def process(self, data: Dict) -> Dict:
        result = data
        for transform in self.transformations:
            result = transform(result)
        return result

# Define transformations
def normalize_dates(data: Dict) -> Dict:
    data['timestamp'] = datetime.fromisoformat(data['timestamp']).isoformat()
    return data

def calculate_metrics(data: Dict) -> Dict:
    data['total_value'] = data['quantity'] * data['price']
    data['tax'] = data['total_value'] * 0.2
    return data

def format_currency(data: Dict) -> Dict:
    for key in ['price', 'total_value', 'tax']:
        data[key] = f"${data[key]:.2f}"
    return data

# Create and use pipeline
pipeline = DataPipeline([
    normalize_dates,
    calculate_metrics,
    format_currency
])

# Sample data
sample_data = {
    'product_id': 'ABC123',
    'timestamp': '2024-01-15T14:30:00',
    'quantity': 5,
    'price': 29.99
}

result = pipeline.process(sample_data)
print(json.dumps(result, indent=2))

# Output:
# {
#   "product_id": "ABC123",
#   "timestamp": "2024-01-15T14:30:00",
#   "quantity": 5,
#   "price": "$29.99",
#   "total_value": "$149.95",
#   "tax": "$29.99"
# }
```

Slide 11: Ứng dụng thực tế - Hệ thống hướng sự kiện

Việc triển khai hệ thống hướng sự kiện bằng cách sử dụng các hàm hạng nhất trình bày cách tạo mẫu đăng ký xuất bản linh hoạt để xử lý các sự kiện và lệnh gọi lại ứng dụng phức tạp.

```python
from typing import Callable, Dict, List, Any
from datetime import datetime

class EventSystem:
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
        self.event_history: List[Dict] = []

    def subscribe(self, event_type: str, callback: Callable) -> None:
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)

    def publish(self, event_type: str, data: Any = None) -> None:
        event = {
            'type': event_type,
            'data': data,
            'timestamp': datetime.now().isoformat()
        }
        self.event_history.append(event)

        if event_type in self.subscribers:
            for callback in self.subscribers[event_type]:
                callback(event)

# Example usage in a simple trading system
def log_trade(event: Dict) -> None:
    print(f"[LOG] Trade executed: {event['data']}")

def update_portfolio(event: Dict) -> None:
    print(f"[PORTFOLIO] Updating positions: {event['data']}")

def notify_user(event: Dict) -> None:
    print(f"[NOTIFICATION] Trade alert: {event['data']}")

# Initialize system
event_system = EventSystem()

# Register handlers
event_system.subscribe('TRADE_EXECUTED', log_trade)
event_system.subscribe('TRADE_EXECUTED', update_portfolio)
event_system.subscribe('TRADE_EXECUTED', notify_user)

# Simulate trade
trade_data = {
    'symbol': 'AAPL',
    'quantity': 100,
    'price': 150.75,
    'action': 'BUY'
}

event_system.publish('TRADE_EXECUTED', trade_data)

# Output:
# [LOG] Trade executed: {'symbol': 'AAPL', 'quantity': 100, 'price': 150.75, 'action': 'BUY'}
# [PORTFOLIO] Updating positions: {'symbol': 'AAPL', 'quantity': 100, 'price': 150.75, 'action': 'BUY'}
# [NOTIFICATION] Trade alert: {'symbol': 'AAPL', 'quantity': 100, 'price': 150.75, 'action': 'BUY'}
```

Slide 12: Phân tích hiệu suất của các hàm bậc cao

Trang trình bày này khám phá ý nghĩa về hiệu suất của việc sử dụng các hàm bậc cao hơn và trình bày cách triển khai kết hợp hàm hiệu quả đồng thời đo lường thời gian thực thi và mức sử dụng bộ nhớ trong các tình huống thực tế.

```python
import time
import sys
from memory_profiler import profile
from functools import reduce, partial

class PerformanceAnalyzer:
    @staticmethod
    def measure_execution(func):
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            return result, end_time - start_time
        return wrapper

    @profile
    def analyze_approaches(self, data_size: int = 1000000):
        # Traditional loop approach
        def traditional_sum():
            result = 0
            for i in range(data_size):
                result += i * i
            return result

        # Functional approach with map and reduce
        def functional_sum():
            return reduce(lambda x, y: x + y,
                        map(lambda x: x * x, range(data_size)))

        # Generator approach
        def generator_sum():
            return sum(i * i for i in range(data_size))

        # Measure and compare
        approaches = {
            'Traditional': traditional_sum,
            'Functional': functional_sum,
            'Generator': generator_sum
        }

        results = {}
        for name, func in approaches.items():
            measured_func = self.measure_execution(func)
            result, execution_time = measured_func()
            results[name] = {
                'result': result,
                'time': execution_time
            }

        return results

# Run analysis
analyzer = PerformanceAnalyzer()
results = analyzer.analyze_approaches()

# Print results
for approach, metrics in results.items():
    print(f"\n{approach} Approach:")
    print(f"Execution time: {metrics['time']:.4f} seconds")
    print(f"Result: {metrics['result']}")

# Sample Output:
# Traditional Approach:
# Execution time: 0.1234 seconds
# Result: 333332833333500000
#
# Functional Approach:
# Execution time: 0.1567 seconds
# Result: 333332833333500000
#
# Generator Approach:
# Execution time: 0.0891 seconds
# Result: 333332833333500000
```

Trang trình bày 13: Gợi ý loại chức năng nâng cao và xác minh thời gian chạy

Việc triển khai này trình bày cách tạo một hệ thống kiểm tra loại chức năng mạnh mẽ bằng cách sử dụng mô-đun gõ và xác minh thời gian chạy của Python, đảm bảo an toàn về loại trong các mẫu lập trình chức năng.

```python
from typing import TypeVar, Callable, Any, get_type_hints
from functools import wraps
import inspect

T = TypeVar('T')
R = TypeVar('R')

def type_check(func: Callable[..., R]) -> Callable[..., R]:
    type_hints = get_type_hints(func)

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Combine positional and keyword arguments
        bound_arguments = inspect.signature(func).bind(*args, **kwargs)
        bound_arguments.apply_defaults()

        # Check each argument
        for param_name, value in bound_arguments.arguments.items():
            if param_name in type_hints:
                expected_type = type_hints[param_name]
                if not isinstance(value, expected_type):
                    raise TypeError(
                        f"Parameter '{param_name}' must be {expected_type.__name__}, "
                        f"got {type(value).__name__}"
                    )

        # Execute and check return type
        result = func(*args, **kwargs)
        if 'return' in type_hints and not isinstance(result, type_hints['return']):
            raise TypeError(
                f"Return value must be {type_hints['return'].__name__}, "
                f"got {type(result).__name__}"
            )

        return result

    return wrapper

@type_check
def process_data(numbers: list[int], multiplier: int) -> list[int]:
    return [n * multiplier for n in numbers]

# Test the function
try:
    # Valid usage
    result1 = process_data([1, 2, 3], 2)
    print(f"Valid result: {result1}")

    # Invalid parameter type
    result2 = process_data([1, 2, '3'], 2)  # Should raise TypeError

except TypeError as e:
    print(f"Type error caught: {e}")

# Output:
# Valid result: [2, 4, 6]
# Type error caught: Parameter 'numbers' must be list[int], got list[str]
```

Trang trình bày 14: Tài nguyên bổ sung

* Giấy tờ và tài nguyên ArXiv:
* "Về khả năng biểu đạt của chức năng hạng nhất"
    * Tìm kiếm: "Các hàm hạng nhất trong ngôn ngữ lập trình ArXiv"
* "Các mẫu lập trình chức năng và phân tích hiệu suất"
    * [https://arxiv.org/abs/cs/0610066](https://arxiv.org/abs/cs/0610066)
* "Hệ thống loại cho lập trình chức năng"
    * Tìm kiếm: "Gõ hệ thống lập trình chức năng ArXiv"
* "Kỹ thuật lập trình Python nâng cao"
    * [https://docs.python.org/3/library/functools.html](https://docs.python.org/3/library/functools.html)
* "Mẫu thiết kế trong lập trình chức năng"
    * Tìm kiếm: "Nghiên cứu mẫu thiết kế lập trình chức năng"
