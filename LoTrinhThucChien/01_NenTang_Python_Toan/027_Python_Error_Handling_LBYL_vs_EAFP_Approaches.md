## Xử lý lỗi Python Phương pháp tiếp cận LBYL và EAFP
Trang trình bày 1: Tìm hiểu mô hình lập trình LBYL và EAFP

Python cung cấp hai cách tiếp cận chính để xử lý các lỗi tiềm ẩn: Nhìn trước khi bạn nhảy (LBYL) và Dễ dàng xin tha thứ hơn là cho phép (EAFP). Những mô hình này thể hiện những triết lý cơ bản khác nhau trong việc xử lý lỗi và thiết kế mã.

```python
# LBYL Example (Look Before You Leap)
def divide_lbyl(x, y):
    if y != 0:  # Check condition before proceeding
        return x / y
    else:
        return "Cannot divide by zero"

# EAFP Example (Easier to Ask for Forgiveness than Permission)
def divide_eafp(x, y):
    try:
        return x / y
    except ZeroDivisionError:
        return "Cannot divide by zero"

# Example usage
print(divide_lbyl(10, 2))  # Output: 5.0
print(divide_lbyl(10, 0))  # Output: Cannot divide by zero
print(divide_eafp(10, 2))  # Output: 5.0
print(divide_eafp(10, 0))  # Output: Cannot divide by zero
```

Trang trình bày 2: Phân tích hiệu suất của LBYL so với EAFP

EAFP thường hoạt động tốt hơn trong Python vì nó phù hợp với thiết kế bên trong của Python. Chi phí kiểm tra các điều kiện trong LBYL có thể tích lũy, đặc biệt khi xử lý nhiều điều kiện hoặc cấu trúc lồng nhau.

```python
import timeit
import statistics

def benchmark_approaches():
    # Setup dictionary for testing
    data = {'key1': 'value1', 'key2': 'value2'}

    # LBYL approach
    def lbyl_test():
        if 'key1' in data and data['key1'] is not None:
            return data['key1']
        return None

    # EAFP approach
    def eafp_test():
        try:
            return data['key1']
        except (KeyError, TypeError):
            return None

    # Benchmark both approaches
    lbyl_time = timeit.repeat(lbyl_test, number=1000000)
    eafp_time = timeit.repeat(eafp_test, number=1000000)

    print(f"LBYL average: {statistics.mean(lbyl_time):.6f} seconds")
    print(f"EAFP average: {statistics.mean(eafp_time):.6f} seconds")

benchmark_approaches()
```

Trang trình bày 3: Trình quản lý bối cảnh và EAFP

Trình quản lý bối cảnh minh họa các nguyên tắc EAFP bằng cách tự động xử lý việc quản lý và dọn dẹp tài nguyên. Cách tiếp cận này đảm bảo xử lý tài nguyên phù hợp ngay cả khi có trường hợp ngoại lệ xảy ra trong quá trình thực thi.

```python
class DatabaseConnection:
    def __init__(self, connection_string):
        self.connection_string = connection_string

    def __enter__(self):
        print(f"Connecting to database: {self.connection_string}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Closing database connection")
        if exc_type is not None:
            print(f"An error occurred: {exc_val}")
        return False  # Propagate exceptions

# Usage example
try:
    with DatabaseConnection("postgresql://localhost:5432/mydb") as db:
        raise ValueError("Simulated error")
except ValueError as e:
    print(f"Caught error: {e}")
```

Slide 4: Practical Implementation in File Handling

File operations demonstrate the superiority of EAFP in real-world scenarios. This implementation shows how to handle multiple potential errors while maintaining clean, readable code.

```python
def process_file_eafp(filename):
    try:
        with open(filename, 'r') as file:
            content = file.read()
            numbers = [float(num) for num in content.split()]
            return sum(numbers) / len(numbers)
    except FileNotFoundError:
        print(f"File {filename} not found")
        return None
    except ValueError:
        print("File contains invalid numbers")
        return None
    except ZeroDivisionError:
        print("File is empty")
        return None

# Example usage with different scenarios
print(process_file_eafp("valid.txt"))    # Processes valid file
print(process_file_eafp("missing.txt"))  # Handles missing file
print(process_file_eafp("invalid.txt"))  # Handles invalid content
```

Trang trình bày 5: Truy cập thuộc tính động bằng EAFP

Mô hình EAFP tỏa sáng khi xử lý các lệnh gọi phương thức và truy cập thuộc tính động, cung cấp cách tiếp cận Pythonic và hiệu quả hơn để xử lý các tương tác đối tượng.

```python
class DynamicObject:
    def __init__(self):
        self.existing_attr = "I exist"

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return f"Created dynamic attribute: {name}"

obj = DynamicObject()

# Demonstrate dynamic attribute access
print(obj.existing_attr)      # Output: I exist
print(obj.nonexistent_attr)   # Output: Created dynamic attribute: nonexistent_attr

# Example with method calls
def safe_call(obj, method_name, *args, **kwargs):
    try:
        method = getattr(obj, method_name)
        return method(*args, **kwargs)
    except AttributeError:
        return f"Method {method_name} not found"
    except Exception as e:
        return f"Error executing {method_name}: {str(e)}"

print(safe_call(obj, "existing_attr"))
print(safe_call(obj, "unknown_method"))
```

Trang trình bày 6: Phân cấp ngoại lệ để xử lý lỗi tùy chỉnh

Việc hiểu và triển khai hệ thống phân cấp ngoại lệ tùy chỉnh cho phép xử lý lỗi hiệu quả trong các ứng dụng lớn hơn. Việc triển khai này thể hiện cách tạo các ngoại lệ dành riêng cho từng miền trong khi vẫn duy trì triết lý EAFP.

```python
class DataProcessingError(Exception):
    """Base exception for data processing errors"""
    pass

class ValidationError(DataProcessingError):
    """Raised when data validation fails"""
    def __init__(self, field, message):
        self.field = field
        self.message = message
        super().__init__(f"{field}: {message}")

class ProcessingError(DataProcessingError):
    """Raised when data processing fails"""
    pass

def process_data(data):
    try:
        if not isinstance(data, dict):
            raise ValidationError("input", "Must be a dictionary")
        if "age" in data and (not isinstance(data["age"], int) or data["age"] < 0):
            raise ValidationError("age", "Must be a positive integer")

        # Process the data
        return {"processed": True, "data": data}

    except ValidationError as e:
        print(f"Validation failed: {e}")
        return None
    except Exception as e:
        raise ProcessingError(f"Unexpected error: {str(e)}")

# Example usage
print(process_data({"name": "John", "age": 30}))  # Valid data
print(process_data({"name": "John", "age": -5}))  # Invalid age
print(process_data([1, 2, 3]))  # Invalid input type
```

Trang trình bày 7: Ứng dụng trong thế giới thực: Đường ống xử lý dữ liệu

Việc triển khai thực tế các nguyên tắc EAFP trong quy trình xử lý dữ liệu, trình bày cách xử lý các tình trạng lỗi khác nhau trong khi vẫn duy trì khả năng đọc và độ mạnh của mã.

```python
import json
from datetime import datetime

class DataPipeline:
    def __init__(self):
        self.transformations = []
        self.error_log = []

    def add_transformation(self, func):
        self.transformations.append(func)

    def log_error(self, step, error):
        self.error_log.append({
            'timestamp': datetime.now().isoformat(),
            'step': step,
            'error': str(error)
        })

    def process(self, data):
        result = data
        for idx, transform in enumerate(self.transformations):
            try:
                result = transform(result)
            except Exception as e:
                self.log_error(f"Step {idx}", e)
                return None
        return result

# Example transformations
def validate_json(data):
    return json.loads(data) if isinstance(data, str) else data

def normalize_dates(data):
    if 'date' in data:
        data['date'] = datetime.strptime(
            data['date'], '%Y-%m-%d'
        ).isoformat()
    return data

# Usage example
pipeline = DataPipeline()
pipeline.add_transformation(validate_json)
pipeline.add_transformation(normalize_dates)

# Process valid data
valid_data = '{"date": "2024-01-01", "value": 100}'
print(pipeline.process(valid_data))

# Process invalid data
invalid_data = '{"date": "invalid-date", "value": 100}'
print(pipeline.process(invalid_data))
print(f"Errors: {pipeline.error_log}")
```

Trang trình bày 8: Trình trang trí với EAFP để xác thực chức năng

Việc triển khai các trình trang trí bằng nguyên tắc EAFP cung cấp một cách rõ ràng để xử lý xác thực đầu vào của hàm và xử lý lỗi mà không làm lộn xộn logic chức năng chính.

```python
from functools import wraps
import time

def retry_with_backoff(max_retries=3, initial_delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    print(f"Attempt {attempt + 1} failed: {str(e)}")
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff

            raise last_exception
        return wrapper
    return decorator

@retry_with_backoff(max_retries=3, initial_delay=1)
def unstable_network_call(url):
    import random
    if random.random() < 0.7:  # 70% chance of failure
        raise ConnectionError("Network unstable")
    return f"Success: {url}"

# Example usage
try:
    result = unstable_network_call("http://example.com")
    print(result)
except ConnectionError as e:
    print(f"Final failure: {e}")
```

Trang trình bày 9: Chuỗi ngoại lệ nâng cao

Chuỗi ngoại lệ cho phép duy trì bối cảnh lỗi ban đầu trong khi đưa ra các ngoại lệ cụ thể hơn. Việc triển khai này thể hiện khả năng xử lý lỗi phức tạp trong một hệ thống mô-đun.

```python
class DatabaseError(Exception):
    pass

class NetworkError(Exception):
    pass

class ServiceError(Exception):
    pass

def fetch_data_from_db():
    try:
        # Simulate database operation
        raise ConnectionError("Database connection failed")
    except ConnectionError as e:
        raise DatabaseError("Database operation failed") from e

def fetch_from_network():
    try:
        # Simulate network call
        raise TimeoutError("Network timeout")
    except TimeoutError as e:
        raise NetworkError("Network operation failed") from e

def service_operation():
    try:
        fetch_data_from_db()
        fetch_from_network()
    except (DatabaseError, NetworkError) as e:
        raise ServiceError("Service operation failed") from e

# Example usage with full traceback
try:
    service_operation()
except ServiceError as e:
    print(f"Top level error: {e}")
    print("\nOriginal cause:", e.__cause__)
    print("\nFull traceback:")
    import traceback
    traceback.print_exc()
```

Trang trình bày 10: Xử lý lỗi theo ngữ cảnh

Việc triển khai xử lý lỗi nhận biết ngữ cảnh cho phép phản hồi lỗi động dựa trên môi trường thực thi và bối cảnh hoạt động.

```python
import contextlib
from typing import Optional, Any
from dataclasses import dataclass
from enum import Enum, auto

class Environment(Enum):
    DEVELOPMENT = auto()
    STAGING = auto()
    PRODUCTION = auto()

@dataclass
class ExecutionContext:
    environment: Environment
    debug: bool
    user_id: Optional[str] = None

class ContextualErrorHandler:
    def __init__(self, context: ExecutionContext):
        self.context = context
        self.errors = []

    @contextlib.contextmanager
    def handle_errors(self, operation_name: str):
        try:
            yield
        except Exception as e:
            self.errors.append({
                'operation': operation_name,
                'error': str(e),
                'type': type(e).__name__
            })

            if self.context.environment == Environment.DEVELOPMENT:
                print(f"Debug info for {operation_name}:", str(e))
            elif self.context.environment == Environment.PRODUCTION:
                if self.context.user_id:
                    print(f"Error for user {self.context.user_id}")
                else:
                    print("System error occurred")
            raise

# Example usage
context = ExecutionContext(
    environment=Environment.DEVELOPMENT,
    debug=True,
    user_id="user123"
)

handler = ContextualErrorHandler(context)

def risky_operation():
    with handler.handle_errors("data_processing"):
        raise ValueError("Invalid data format")

try:
    risky_operation()
except ValueError:
    print("Error log:", handler.errors)
```

Trang trình bày 11: Xử lý lỗi không đồng bộ

Việc quản lý lỗi trong mã không đồng bộ đòi hỏi sự chú ý đặc biệt để đảm bảo việc truyền và xử lý lỗi thích hợp trên các coroutine.

```python
import asyncio
from typing import Optional
from contextlib import AsyncExitStack

class AsyncResource:
    async def __aenter__(self):
        print("Acquiring async resource")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        print("Releasing async resource")
        if exc_type is not None:
            print(f"Handled error: {exc_val}")

    async def process(self):
        await asyncio.sleep(1)
        return "Processed data"

async def process_with_timeout(timeout: float) -> Optional[str]:
    try:
        async with AsyncExitStack() as stack:
            resource = await stack.enter_async_context(AsyncResource())

            # Run with timeout
            result = await asyncio.wait_for(
                resource.process(),
                timeout=timeout
            )
            return result
    except asyncio.TimeoutError:
        print("Operation timed out")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

# Example usage
async def main():
    # Successful case
    result1 = await process_with_timeout(2.0)
    print("Result 1:", result1)

    # Timeout case
    result2 = await process_with_timeout(0.5)
    print("Result 2:", result2)

asyncio.run(main())
```

Trang trình bày 12: Kiểm tra dựa trên thuộc tính với EAFP

Thử nghiệm dựa trên thuộc tính kết hợp với các nguyên tắc EAFP đảm bảo hoạt động mã mạnh mẽ trên nhiều loại đầu vào trong khi vẫn duy trì các mẫu xử lý lỗi Pythonic.

```python
from hypothesis import given, strategies as st
from typing import List, Any
import math

class NumberProcessor:
    def process_numbers(self, numbers: List[float]) -> float:
        try:
            cleaned = [n for n in numbers if isinstance(n, (int, float))]
            if not cleaned:
                raise ValueError("No valid numbers provided")
            return sum(cleaned) / len(cleaned)
        except TypeError:
            raise ValueError("Invalid input type")

    def calculate_statistics(self, numbers: List[float]) -> dict:
        try:
            mean = self.process_numbers(numbers)
            variance = sum((x - mean) ** 2 for x in numbers) / len(numbers)
            return {
                'mean': mean,
                'std_dev': math.sqrt(variance),
                'count': len(numbers)
            }
        except Exception as e:
            return {'error': str(e)}

# Property-based tests
@given(st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=1))
def test_processor(numbers):
    processor = NumberProcessor()
    try:
        result = processor.calculate_statistics(numbers)
        assert 'mean' in result
        assert 'std_dev' in result
        assert result['count'] == len(numbers)
    except Exception as e:
        assert 'error' in result

# Example usage
processor = NumberProcessor()
print(processor.calculate_statistics([1.0, 2.0, 3.0, 4.0, 5.0]))
print(processor.calculate_statistics([]))  # Handles empty list
print(processor.calculate_statistics(['invalid']))  # Handles invalid input
```

Trang trình bày 13: Xác thực dữ liệu theo thời gian thực với EAFP

Triển khai hệ thống xác thực dữ liệu theo thời gian thực bằng nguyên tắc EAFP, trình bày cách xử lý dữ liệu truyền phát với các yêu cầu xác thực phức tạp.

```python
from datetime import datetime
from typing import Generator, Dict, Any
import json

class DataValidator:
    def __init__(self):
        self._validators = {}
        self.setup_validators()

    def setup_validators(self):
        def validate_timestamp(value: str) -> datetime:
            return datetime.fromisoformat(value)

        def validate_numeric(value: Any) -> float:
            return float(value)

        self._validators = {
            'timestamp': validate_timestamp,
            'value': validate_numeric
        }

    def validate_stream(self, data_stream: Generator[Dict, None, None]):
        for item in data_stream:
            try:
                validated = {}
                for field, validator in self._validators.items():
                    if field in item:
                        validated[field] = validator(item[field])
                yield validated
            except Exception as e:
                yield {'error': f"Validation failed: {str(e)}", 'data': item}

# Example usage
def generate_test_data():
    test_data = [
        {'timestamp': '2024-01-01T12:00:00', 'value': '42.5'},
        {'timestamp': 'invalid', 'value': '42.5'},
        {'timestamp': '2024-01-01T12:00:00', 'value': 'not_a_number'},
        {'timestamp': '2024-01-01T12:00:00', 'value': '43.2'}
    ]
    for item in test_data:
        yield item

validator = DataValidator()
for result in validator.validate_stream(generate_test_data()):
    print(json.dumps(result, default=str, indent=2))
```

Trang trình bày 14: Tài nguyên bổ sung

* Nghiên cứu tài liệu, tài liệu để hiểu sâu hơn:

* arxiv.org/abs/computing/0702072 - "Xử lý ngoại lệ: Các vấn đề và ký hiệu được đề xuất"
* [https://peps.python.org/pep-0463/](https://peps.python.org/pep-0463/) - Xử lý ngoại lệ Python PEP
* [https://dl.acm.org/doi/10.1145/1988042.1988046](https://dl.acm.org/doi/10.1145/1988042.1988046) - "Xử lý ngoại lệ: Nghiên cứu thực địa về Java"

* Đề xuất đọc cho các khái niệm nâng cao:

* [https://www.python.org/dev/peps/pep-3134/](https://www.python.org/dev/peps/pep-3134/) - Chuỗi ngoại lệ và dấu vết nhúng
* [https://docs.python.org/3/tutorial/errors.html](https://docs.python.org/3/tutorial/errors.html) - Tài liệu xử lý ngoại lệ Python
* [https://google.github.io/styleguide/pyguide.html#24-Exceptions](https://google.github.io/styleguide/pyguide.html#24-Exceptions) - Hướng dẫn về kiểu ngoại lệ của Google Python

* Nguồn lực cộng đồng:

* [https://stackoverflow.com/questions/tagged/python+Exception-handling](https://stackoverflow.com/questions/tagged/python+Exception-handling) - Xử lý ngoại lệ Python trong Stack Overflow
* [https://realpython.com/python-Exceptions/](https://realpython.com/python-Exceptions/) - Hướng dẫn xử lý ngoại lệ Python thực
* [https://pypi.org/project/better-Exceptions/](https://pypi.org/project/better-Exceptions/) - Tài liệu về gói ngoại lệ tốt hơn
