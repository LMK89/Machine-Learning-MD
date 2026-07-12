## Cách tạo ngoại lệ tùy chỉnh trong Python
Trang trình bày 1: Tìm hiểu các ngoại lệ tùy chỉnh

Các ngoại lệ tùy chỉnh trong Python mở rộng lớp Ngoại lệ tích hợp để tạo các cơ chế xử lý lỗi chuyên dụng. Điều này cho phép các nhà phát triển xác định các điều kiện lỗi dành riêng cho ứng dụng và cung cấp các thông báo lỗi có ý nghĩa phù hợp với yêu cầu chương trình của họ.

```python
# Basic structure of a custom exception
class CustomError(Exception):
    def __init__(self, message="A custom error occurred"):
        self.message = message
        super().__init__(self.message)
```

Slide 2: Creating a Domain-Specific Exception

A well-designed custom exception should encapsulate domain-specific error conditions and relevant error data. This enables precise error handling and debugging by capturing contextual information about the error state.

```python
class InvalidWeightError(Exception):
    def __init__(self, weight, message=None):
        self.weight = weight
        self.message = message or f"Invalid weight value: {weight}"
        super().__init__(self.message)

    def __str__(self):
        return f"Weight Error: {self.message}"
```

Trang trình bày 3: Triển khai Công cụ tính trọng lượng với ngoại lệ tùy chỉnh

Công cụ tính trọng lượng thể hiện cách sử dụng thực tế các ngoại lệ tùy chỉnh bằng cách xác thực các tham số đầu vào và đưa ra các lỗi thích hợp khi không đáp ứng các điều kiện. Điều này đảm bảo xử lý lỗi mạnh mẽ trong các ứng dụng trong thế giới thực.

```python
def calculate_moon_weight(earth_weight):
    try:
        if not isinstance(earth_weight, (int, float)):
            raise InvalidWeightError(earth_weight, "Weight must be a number")
        if earth_weight < 0 or earth_weight > 300:
            raise InvalidWeightError(earth_weight, "Weight must be between 0 and 300 kg")

        return earth_weight * 0.165
    except InvalidWeightError as e:
        print(f"Error: {e}")
        return None
```

Slide 4: Hierarchical Exception Structure

Custom exceptions can form a hierarchy to represent different categories of errors while maintaining a common base. This approach enables more granular error handling and improved code organization.

```python
class WeightError(Exception):
    """Base exception for weight-related errors"""
    pass

class NegativeWeightError(WeightError):
    def __init__(self, weight):
        super().__init__(f"Weight cannot be negative: {weight}")

class ExcessiveWeightError(WeightError):
    def __init__(self, weight, limit):
        super().__init__(f"Weight {weight} exceeds limit of {limit}")
```

Trang trình bày 5: Thuộc tính ngoại lệ nâng cao

Các ứng dụng phức tạp thường yêu cầu ngoại lệ để mang dữ liệu bổ sung cho mục đích gỡ lỗi và ghi nhật ký. Các ngoại lệ tùy chỉnh có thể bao gồm các thuộc tính và phương pháp chuyên biệt để nâng cao khả năng báo cáo lỗi.

```python
class DataValidationError(Exception):
    def __init__(self, value, expected_type, constraints=None):
        self.value = value
        self.expected_type = expected_type
        self.constraints = constraints or {}
        self.timestamp = datetime.now()
        message = self._build_message()
        super().__init__(message)

    def _build_message(self):
        return f"Validation failed for value {self.value} (type: {type(self.value)})"
```

Trang trình bày 6: Quản lý bối cảnh ngoại lệ

Các ngoại lệ tùy chỉnh có thể được tích hợp với trình quản lý bối cảnh để đảm bảo xử lý và dọn dẹp tài nguyên phù hợp, ngay cả khi xảy ra lỗi trong quá trình thực thi.

```python
class DatabaseConnection:
    class ConnectionError(Exception):
        def __init__(self, operation, details):
            self.operation = operation
            self.details = details
            super().__init__(f"Database {operation} failed: {details}")

    def __enter__(self):
        if not self.connect():
            raise self.ConnectionError("connect", "Unable to establish connection")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
```

Trang trình bày 7: Ví dụ thực tế - Đường ống xử lý dữ liệu

Ví dụ này minh họa cách triển khai thực tế các ngoại lệ tùy chỉnh trong quy trình xử lý dữ liệu, xử lý các tình trạng lỗi khác nhau có thể xảy ra trong quá trình chuyển đổi dữ liệu.

```python
class DataProcessingError(Exception):
    def __init__(self, stage, error_type, details):
        self.stage = stage
        self.error_type = error_type
        self.details = details
        super().__init__(f"Error in {stage}: {error_type} - {details}")

def process_dataset(data):
    try:
        if not isinstance(data, list):
            raise DataProcessingError("validation", "TypeError", "Input must be a list")

        processed = []
        for idx, item in enumerate(data):
            if not item.strip():
                raise DataProcessingError("processing", "ValueError",
                                       f"Empty value at index {idx}")
            processed.append(item.upper())
        return processed
    except DataProcessingError as e:
        print(f"Processing failed: {e}")
        return None
```

Trang trình bày 8: Chuỗi ngoại lệ

Chuỗi ngoại lệ cho phép giữ lại lỗi ban đầu trong khi đưa ra một ngoại lệ mới, cụ thể hơn. Điều này duy trì bối cảnh lỗi đầy đủ cho mục đích gỡ lỗi.

```python
class FileProcessingError(Exception):
    pass

def process_config_file(filename):
    try:
        with open(filename) as f:
            config = json.load(f)
    except FileNotFoundError as e:
        raise FileProcessingError(
            f"Configuration file {filename} not found"
        ) from e
    except json.JSONDecodeError as e:
        raise FileProcessingError(
            f"Invalid JSON in {filename}"
        ) from e
```

Trang trình bày 9: Ngoại lệ tùy chỉnh với mã lỗi

Các ngoại lệ tùy chỉnh có thể kết hợp các mã lỗi để cung cấp khả năng xử lý lỗi được tiêu chuẩn hóa trên một ứng dụng. Cách tiếp cận này tạo điều kiện thuận lợi cho việc xử lý lỗi tự động và quốc tế hóa các thông báo lỗi.

```python
class SystemError(Exception):
    def __init__(self, code, message):
        self.code = code
        self.message = message
        super().__init__(f"Error {code}: {message}")

    @classmethod
    def resource_not_found(cls, resource_id):
        return cls("E404", f"Resource {resource_id} not found")

    @classmethod
    def permission_denied(cls, operation):
        return cls("E403", f"Permission denied for operation: {operation}")
```

Trang trình bày 10: Ngoại lệ tùy chỉnh với tích hợp ghi nhật ký

Việc tích hợp khả năng ghi nhật ký vào các ngoại lệ tùy chỉnh cho phép tự động theo dõi các lần xuất hiện lỗi và đơn giản hóa việc gỡ lỗi trong môi trường sản xuất.

```python
import logging
from datetime import datetime

class LoggedError(Exception):
    def __init__(self, message, logger=None):
        self.timestamp = datetime.now()
        self.logger = logger or logging.getLogger(__name__)

        super().__init__(message)
        self._log_error()

    def _log_error(self):
        error_info = {
            'message': str(self),
            'timestamp': self.timestamp,
            'type': self.__class__.__name__
        }
        self.logger.error(f"Error occurred: {error_info}")
```

Trang trình bày 11: Ví dụ thực tế - Trình xử lý yêu cầu API

Việc triển khai này cho thấy cách sử dụng các ngoại lệ tùy chỉnh trong trình xử lý yêu cầu API để quản lý các loại lỗi yêu cầu khác nhau và cung cấp phản hồi thích hợp.

```python
class APIError(Exception):
    def __init__(self, status_code, message, details=None):
        self.status_code = status_code
        self.message = message
        self.details = details or {}
        super().__init__(message)

class RequestHandler:
    def process_request(self, request_data):
        try:
            if not request_data:
                raise APIError(400, "Empty request body")

            if 'auth_token' not in request_data:
                raise APIError(401, "Missing authentication token")

            if not self.validate_token(request_data['auth_token']):
                raise APIError(403, "Invalid authentication token")

            return self.handle_validated_request(request_data)

        except APIError as e:
            return {
                'status': 'error',
                'code': e.status_code,
                'message': e.message,
                'details': e.details
            }
```

Trang trình bày 12: Trang trí xử lý ngoại lệ

Việc tạo một trình trang trí để xử lý ngoại lệ mang lại một cách rõ ràng để triển khai việc xử lý lỗi nhất quán trên nhiều chức năng trong khi vẫn duy trì khả năng đọc mã.

```python
from functools import wraps

def handle_exceptions(error_map=None):
    error_map = error_map or {}

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_type = type(e)
                if error_type in error_map:
                    raise error_map[error_type](str(e))
                raise
        return wrapper
    return decorator

@handle_exceptions({
    ValueError: CustomError,
    KeyError: DataValidationError
})
def process_data(data):
    # Function implementation
    pass
```

Trang trình bày 13: Truy nguyên ngoại lệ nâng cao

Các ngoại lệ tùy chỉnh có thể được nâng cao bằng thông tin truy nguyên chi tiết để cung cấp khả năng gỡ lỗi toàn diện trong các ứng dụng phức tạp.

```python
import traceback
import sys

class DetailedError(Exception):
    def __init__(self, message, **context):
        super().__init__(message)
        self.context = context
        self.traceback = self._capture_traceback()

    def _capture_traceback(self):
        exc_type, exc_value, exc_traceback = sys.exc_info()
        if exc_traceback:
            return ''.join(traceback.format_tb(exc_traceback))
        return ''.join(traceback.format_stack()[:-1])

    def get_error_details(self):
        return {
            'message': str(self),
            'context': self.context,
            'traceback': self.traceback
        }
```

Trang trình bày 14: Tài nguyên bổ sung

* Xây dựng các ngoại lệ Python tốt hơn: [https://arxiv.org/abs/cs/0701072](https://arxiv.org/abs/cs/0701072)
* Các mẫu xử lý ngoại lệ trong hệ thống quy mô lớn: [https://ieeexplore.ieee.org/document/8445076](https://ieeexplore.ieee.org/document/8445076)
* Các phương pháp hay nhất để xử lý ngoại lệ Python: [https://docs.python.org/3/tutorial/errors.html](https://docs.python.org/3/tutorial/errors.html)
* Các mẫu xử lý lỗi trong hệ thống phân tán: [https://www.sciencedirect.com/science/article/pii/S0167642309000343](https://www.sciencedirect.com/science/article/pii/S0167642309000343)
* Xử lý ngoại lệ Python - Chủ đề nâng cao: [https://realpython.com/python-Exceptions/](https://realpython.com/python-Exceptions/)
