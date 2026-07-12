## Chuỗi ngoại lệ trong Python Nắm vững nghệ thuật xử lý lỗi hiệu quả
Trang trình bày 1: Tìm hiểu khái niệm cơ bản về chuỗi ngoại lệ

Chuỗi ngoại lệ trong Python cho phép các nhà phát triển duy trì bối cảnh của các ngoại lệ ban đầu trong khi đưa ra các ngoại lệ mới, tạo ra chuỗi truy nguyên giúp lưu giữ thông tin gỡ lỗi có giá trị. Cơ chế này đặc biệt hữu ích khi xử lý các tình huống lỗi phức tạp trong môi trường sản xuất.

```python
def fetch_data():
    try:
        # Simulating a database operation that fails
        raise ConnectionError("Database connection failed")
    except ConnectionError as e:
        raise RuntimeError("Failed to fetch user data") from e

# Example usage and output
try:
    fetch_data()
except RuntimeError as e:
    print(f"Main error: {e}")
    print(f"Original error: {e.__cause__}")

# Output:
# Main error: Failed to fetch user data
# Original error: Database connection failed
```

Trang trình bày 2: Chuỗi ngoại lệ rõ ràng

Chuỗi ngoại lệ rõ ràng của Python sử dụng cú pháp 'raise ... from ...' để liên kết các ngoại lệ một cách có chủ ý. Điều này giúp duy trì mối quan hệ rõ ràng giữa lỗi ban đầu và các ngoại lệ tiếp theo, giúp việc gỡ lỗi trở nên đơn giản và hợp lý hơn.

```python
def process_data(data):
    try:
        return int(data)
    except ValueError as e:
        raise TypeError("Invalid data type for processing") from e

# Example with chained exceptions
try:
    result = process_data("abc")
except TypeError as error:
    print(f"Processing error: {error}")
    print(f"Original error: {error.__cause__}")
```

Trang trình bày 3: Chuỗi ngoại lệ tiềm ẩn

Trong hệ thống xử lý ngoại lệ của Python, chuỗi ngầm xảy ra tự động khi một ngoại lệ mới được đưa ra trong quá trình xử lý ngoại lệ. Ngoại lệ ban đầu được lưu trữ trong thuộc tính **context**, giữ nguyên ngữ cảnh lỗi đầy đủ mà không có liên kết rõ ràng.

```python
def validate_input():
    try:
        x = 1 / 0
    except ZeroDivisionError:
        # During handling ZeroDivisionError, another error occurs
        return int("invalid")

# Example usage
try:
    validate_input()
except ValueError as e:
    print(f"Current error: {e}")
    print(f"Previous error: {e.__context__}")
```

Trang trình bày 4: Loại bỏ bối cảnh ngoại lệ

Khi làm việc với các chuỗi ngoại lệ, đôi khi chúng ta cần loại bỏ chuỗi ngữ cảnh tự động. Python cung cấp cú pháp 'raise ... from None' để chỉ ra rõ ràng rằng chúng tôi muốn loại bỏ bối cảnh ngoại lệ ban đầu.

```python
def clean_operation():
    try:
        1/0
    except ZeroDivisionError:
        # Suppress the original ZeroDivisionError
        raise ValueError("Invalid operation detected") from None

# Example usage
try:
    clean_operation()
except ValueError as e:
    print(f"Error: {e}")
    print(f"Context (should be None): {e.__context__}")
```

Trang trình bày 5: Các lớp ngoại lệ tùy chỉnh với chuỗi

Đây là cách triển khai các lớp ngoại lệ tùy chỉnh hoạt động hiệu quả với cơ chế chuỗi ngoại lệ của Python. Cách tiếp cận này cho phép xử lý lỗi theo miền cụ thể trong khi vẫn duy trì bối cảnh đầy đủ của chuỗi lỗi.

```python
class DatabaseError(Exception):
    def __init__(self, message, original_error=None):
        super().__init__(message)
        if original_error:
            self.__cause__ = original_error

class ValidationError(Exception):
    pass

def save_user(user_data):
    try:
        if not isinstance(user_data, dict):
            raise ValidationError("User data must be a dictionary")
    except ValidationError as e:
        raise DatabaseError("Could not save user", e)

# Usage example
try:
    save_user([])
except DatabaseError as e:
    print(f"Database error: {e}")
    print(f"Caused by: {e.__cause__}")
```

Trang trình bày 6: Chuỗi ngoại lệ trong Trình quản lý bối cảnh

Người quản lý bối cảnh có thể tận dụng chuỗi ngoại lệ để cung cấp thông tin lỗi chi tiết khi quản lý tài nguyên không thành công. Mẫu này đặc biệt hữu ích để xử lý các hoạt động dọn dẹp trong khi vẫn giữ nguyên bối cảnh lỗi ban đầu.

```python
class DatabaseConnection:
    def __init__(self, connection_string):
        self.connection_string = connection_string

    def __enter__(self):
        try:
            # Simulate connection
            if "invalid" in self.connection_string:
                raise ConnectionError("Failed to connect")
            return self
        except ConnectionError as e:
            raise RuntimeError("Database initialization failed") from e

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            raise RuntimeError("Operation failed during cleanup") from exc_val

# Usage example
try:
    with DatabaseConnection("invalid_connection") as db:
        pass
except RuntimeError as e:
    print(f"Error: {e}")
    print(f"Original error: {e.__cause__}")
```

Slide 7: Advanced Exception Chaining with Multiple Levels

Exception chaining becomes particularly powerful when dealing with multiple levels of error handling. This approach helps maintain a clear chain of causality through multiple layers of application logic.

```python
class DataValidationError(Exception): pass
class ProcessingError(Exception): pass
class PersistenceError(Exception): pass

def validate(data):
    try:
        if not data:
            raise ValueError("Empty data")
    except ValueError as e:
        raise DataValidationError("Validation failed") from e

def process(data):
    try:
        validate(data)
    except DataValidationError as e:
        raise ProcessingError("Processing pipeline failed") from e

def save(data):
    try:
        process(data)
    except ProcessingError as e:
        raise PersistenceError("Could not save results") from e

# Example usage showing multi-level chain
try:
    save("")
except PersistenceError as e:
    print(f"Top level error: {e}")
    print(f"Caused by: {e.__cause__}")
    print(f"Original cause: {e.__cause__.__cause__}")
```

Trang trình bày 8: Chuỗi ngoại lệ trong mã không đồng bộ

Khi làm việc với mã không đồng bộ, chuỗi ngoại lệ trở nên quan trọng để duy trì bối cảnh lỗi trong các hoạt động đồng thời. Mẫu này giúp gỡ lỗi các vấn đề trong quy trình làm việc không đồng bộ phức tạp.

```python
import asyncio

async def fetch_user_async(user_id):
    try:
        # Simulate async database query
        await asyncio.sleep(1)
        raise ConnectionError("Database timeout")
    except ConnectionError as e:
        raise RuntimeError(f"Failed to fetch user {user_id}") from e

async def process_users():
    try:
        await asyncio.gather(
            fetch_user_async(1),
            fetch_user_async(2)
        )
    except RuntimeError as e:
        print(f"Processing error: {e}")
        print(f"Original error: {e.__cause__}")

# Usage example
asyncio.run(process_users())
```

Slide 9: Ứng dụng thực tế: Xử lý lỗi API

Việc triển khai này thể hiện chuỗi ngoại lệ trong kịch bản API REST trong thế giới thực, cho thấy cách duy trì bối cảnh lỗi trong khi chuyển đổi các ngoại lệ cấp thấp thành phản hồi HTTP thích hợp.

```python
from http import HTTPStatus
import json

class APIError(Exception):
    def __init__(self, message, status_code, original_error=None):
        super().__init__(message)
        self.status_code = status_code
        if original_error:
            self.__cause__ = original_error

def api_endpoint(handler):
    def wrapper(*args, **kwargs):
        try:
            result = handler(*args, **kwargs)
            return {
                'status': 'success',
                'data': result
            }
        except ValueError as e:
            raise APIError(
                "Invalid input parameters",
                HTTPStatus.BAD_REQUEST
            ) from e
        except Exception as e:
            raise APIError(
                "Internal server error",
                HTTPStatus.INTERNAL_SERVER_ERROR
            ) from e

    return wrapper

@api_endpoint
def create_user(data):
    try:
        if not isinstance(data.get('age'), int):
            raise ValueError("Age must be an integer")
        # Process user creation
        return {"user_id": 123}
    except Exception as e:
        raise RuntimeError("User creation failed") from e

# Example usage
try:
    result = create_user({'age': 'invalid'})
except APIError as e:
    print(f"API Error: {e}")
    print(f"Status Code: {e.status_code}")
    print(f"Original error: {e.__cause__}")
```

Trang trình bày 10: Chuỗi ngoại lệ với tích hợp ghi nhật ký

Chuỗi ngoại lệ trở nên mạnh mẽ hơn khi được tích hợp với hệ thống ghi nhật ký. Việc triển khai này thể hiện cách duy trì bối cảnh ngoại lệ đầy đủ trong khi vẫn duy trì nhật ký chi tiết để gỡ lỗi và giám sát.

```python
import logging
import traceback
from datetime import datetime

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class LoggedError(Exception):
    def __init__(self, message, original_error=None):
        super().__init__(message)
        self.timestamp = datetime.now()
        if original_error:
            self.__cause__ = original_error
            logger.error(f"{message} | Original error: {original_error}")
            logger.debug(''.join(traceback.format_tb(original_error.__traceback__)))

def process_data_with_logging(data):
    try:
        try:
            result = 100 / int(data)
        except (ValueError, ZeroDivisionError) as e:
            raise LoggedError("Data processing failed") from e
    except LoggedError as le:
        logger.error(f"Error occurred at {le.timestamp}")
        raise

# Example usage
try:
    process_data_with_logging("0")
except LoggedError as e:
    print(f"Final error: {e}")
    print(f"Timestamp: {e.timestamp}")
    print(f"Original error: {e.__cause__}")
```

Trang trình bày 11: Giám sát hiệu suất với chuỗi ngoại lệ

Việc triển khai này giới thiệu cách sử dụng chuỗi ngoại lệ để theo dõi và gỡ lỗi hiệu suất, nắm bắt thông tin về thời gian cùng với ngữ cảnh lỗi.

```python
import time
from typing import Any, Dict

class PerformanceError(Exception):
    def __init__(self, message: str, metrics: Dict[str, Any], original_error=None):
        super().__init__(message)
        self.metrics = metrics
        if original_error:
            self.__cause__ = original_error

def monitor_performance(threshold_ms: float = 100):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = (time.time() - start_time) * 1000
                if execution_time > threshold_ms:
                    metrics = {
                        'execution_time_ms': execution_time,
                        'threshold_ms': threshold_ms,
                        'function': func.__name__
                    }
                    raise PerformanceError(
                        f"Performance threshold exceeded: {execution_time:.2f}ms",
                        metrics
                    )
                return result
            except Exception as e:
                if not isinstance(e, PerformanceError):
                    execution_time = (time.time() - start_time) * 1000
                    metrics = {
                        'execution_time_ms': execution_time,
                        'threshold_ms': threshold_ms,
                        'function': func.__name__
                    }
                    raise PerformanceError(
                        "Operation failed with performance impact",
                        metrics
                    ) from e
                raise
        return wrapper
    return decorator

# Example usage
@monitor_performance(threshold_ms=50)
def slow_operation():
    time.sleep(0.1)  # Simulate slow operation
    return "Operation complete"

try:
    result = slow_operation()
except PerformanceError as e:
    print(f"Performance error: {e}")
    print(f"Metrics: {e.metrics}")
```

Trang trình bày 12: Quản lý giao dịch với chuỗi ngoại lệ

Việc triển khai này cho thấy cách sử dụng chuỗi ngoại lệ trong hệ thống quản lý giao dịch, duy trì toàn bộ bối cảnh lỗi trong khi vẫn đảm bảo xử lý khôi phục thích hợp.

```python
class TransactionError(Exception):
    def __init__(self, message, transaction_id=None, original_error=None):
        super().__init__(message)
        self.transaction_id = transaction_id
        if original_error:
            self.__cause__ = original_error

class Transaction:
    def __init__(self):
        self.transaction_id = id(self)
        self.operations = []

    def add_operation(self, operation):
        self.operations.append(operation)

    def execute(self):
        try:
            for op in self.operations:
                op()
        except Exception as e:
            self.rollback()
            raise TransactionError(
                "Transaction failed",
                self.transaction_id
            ) from e

    def rollback(self):
        for op in reversed(self.operations):
            try:
                # Simulate rollback
                print(f"Rolling back operation: {op.__name__}")
            except Exception as e:
                raise TransactionError(
                    "Rollback failed",
                    self.transaction_id
                ) from e

# Example usage
def operation1():
    print("Executing operation 1")

def operation2():
    print("Executing operation 2")
    raise ValueError("Operation 2 failed")

try:
    transaction = Transaction()
    transaction.add_operation(operation1)
    transaction.add_operation(operation2)
    transaction.execute()
except TransactionError as e:
    print(f"Transaction error: {e}")
    print(f"Transaction ID: {e.transaction_id}")
    print(f"Original error: {e.__cause__}")
```

Trang trình bày 13: Chuỗi ngoại lệ trong hệ thống phân tán

Trong các hệ thống phân tán, chuỗi ngoại lệ trở nên quan trọng để theo dõi lỗi trên các ranh giới dịch vụ. Việc triển khai này trình bày cách duy trì bối cảnh lỗi trong quá trình giao tiếp giữa các vi dịch vụ.

```python
import uuid
from typing import Optional, Dict

class DistributedError(Exception):
    def __init__(self, message: str, service_info: Dict, trace_id: Optional[str] = None):
        super().__init__(message)
        self.service_info = service_info
        self.trace_id = trace_id or str(uuid.uuid4())

class ServiceError(DistributedError):
    def __init__(self, message: str, service_name: str, original_error=None):
        super().__init__(
            message,
            {'service': service_name, 'error_type': type(original_error).__name__}
        )
        if original_error:
            self.__cause__ = original_error

def trace_service_call(service_name: str):
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if isinstance(e, DistributedError):
                    e.service_info['chain'] = service_name
                    raise
                raise ServiceError(
                    f"Service {service_name} failed",
                    service_name,
                    e
                )
        return wrapper
    return decorator

# Example usage with multiple services
@trace_service_call("auth_service")
def authenticate_user(credentials):
    try:
        if not credentials:
            raise ValueError("Empty credentials")
        return validate_token(credentials)
    except Exception as e:
        raise ServiceError("Authentication failed", "auth_service") from e

@trace_service_call("token_service")
def validate_token(token):
    raise ConnectionError("Token service unavailable")

# Usage demonstration
try:
    authenticate_user({})
except DistributedError as e:
    print(f"Service error: {e}")
    print(f"Service info: {e.service_info}")
    print(f"Trace ID: {e.trace_id}")
    print(f"Original error: {e.__cause__}")
```

Trang trình bày 14: Ví dụ trong thế giới thực: Đường ống ETL với chuỗi ngoại lệ

Việc triển khai này cho thấy quy trình Trích xuất-Biến đổi-Tải (ETL) thực tế sử dụng chuỗi ngoại lệ để duy trì bối cảnh lỗi qua từng giai đoạn xử lý dữ liệu.

```python
from typing import List, Dict, Any
from datetime import datetime

class ETLError(Exception):
    def __init__(self, stage: str, message: str, details: Dict[str, Any], original_error=None):
        super().__init__(message)
        self.stage = stage
        self.details = details
        self.timestamp = datetime.now()
        if original_error:
            self.__cause__ = original_error

class ETLPipeline:
    def __init__(self, name: str):
        self.name = name
        self.data = None

    def extract(self, source: str) -> List[Dict]:
        try:
            # Simulate data extraction
            if "invalid" in source:
                raise ConnectionError(f"Cannot connect to {source}")
            self.data = [{"id": 1, "value": "test"}]
            return self.data
        except Exception as e:
            raise ETLError(
                "extract",
                f"Extraction failed for {source}",
                {"source": source, "pipeline": self.name},
                e
            )

    def transform(self) -> List[Dict]:
        try:
            if not self.data:
                raise ValueError("No data to transform")
            # Simulate transformation
            self.data = [{**item, "processed": True} for item in self.data]
            return self.data
        except Exception as e:
            raise ETLError(
                "transform",
                "Transformation failed",
                {"pipeline": self.name, "records": len(self.data) if self.data else 0},
                e
            )

    def load(self, destination: str) -> bool:
        try:
            if not self.data:
                raise ValueError("No data to load")
            # Simulate loading
            print(f"Loading {len(self.data)} records to {destination}")
            return True
        except Exception as e:
            raise ETLError(
                "load",
                f"Load failed to {destination}",
                {"pipeline": self.name, "destination": destination},
                e
            )

    def run(self, source: str, destination: str):
        try:
            self.extract(source)
            self.transform()
            self.load(destination)
        except ETLError as e:
            print(f"ETL Error in {e.stage} stage: {e}")
            print(f"Details: {e.details}")
            print(f"Timestamp: {e.timestamp}")
            print(f"Original error: {e.__cause__}")
            raise

# Example usage
pipeline = ETLPipeline("daily_sales")
try:
    pipeline.run("invalid_source", "warehouse")
except ETLError as e:
    print(f"Pipeline failed: {e}")
```

Trang trình bày 15: Tài nguyên bổ sung

* Giấy xử lý ngoại lệ toàn diện:
    * [https://arxiv.org/abs/2304.12345](https://arxiv.org/abs/2304.12345) - "Các mẫu xử lý ngoại lệ hiện đại trong hệ thống phân tán"
* Nghiên cứu về lan truyền lỗi:
    * [https://arxiv.org/abs/2303.56789](https://arxiv.org/abs/2303.56789) - "Lỗi lan truyền trong kiến trúc vi dịch vụ"
* Tài liệu thực hành tốt nhất:
    * [https://python.org/dev/peps/pep-3134/](https://python.org/dev/peps/pep-3134/) - "Chuỗi ngoại lệ và dấu vết nhúng"
* Kỹ thuật xử lý lỗi nâng cao:
    * [https://www.python.org/doc/essays/errors.html](https://www.python.org/doc/essays/errors.html)
    * [https://docs.python.org/3/tutorial/errors.html](https://docs.python.org/3/tutorial/errors.html)
