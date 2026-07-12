## Làm chủ các lớp dữ liệu Python
Trang trình bày 1: Giới thiệu về các lớp dữ liệu Python

Lớp dữ liệu là một tính năng mạnh mẽ được giới thiệu trong Python 3.7 giúp đơn giản hóa việc tạo các lớp chủ yếu được sử dụng để lưu trữ dữ liệu. Chúng tự động tạo ra các phương thức đặc biệt như **init**(), **repr**() và **eq**(), giảm mã soạn sẵn trong khi vẫn duy trì các định nghĩa lớp rõ ràng.

```python
from dataclasses import dataclass

# Traditional class implementation
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f'Point(x={self.x}, y={self.y})'

    def __eq__(self, other):
        if not isinstance(other, Point):
            return NotImplemented
        return (self.x, self.y) == (other.x, other.y)

# Equivalent Data Class implementation
@dataclass
class PointDataClass:
    x: float
    y: float

# Usage example
p1 = PointDataClass(1.0, 2.0)
print(p1)  # Output: PointDataClass(x=1.0, y=2.0)
```

Trang trình bày 2: Giá trị mặc định và loại trường

Lớp Dữ liệu hỗ trợ gợi ý loại và giá trị mặc định, cung cấp tài liệu mã tốt hơn và kiểm tra loại thời gian chạy khi kết hợp với các công cụ như mypy. Các trường có thể được khởi tạo với giá trị mặc định hoặc được đặt tùy chọn bằng cách sử dụng Không có.

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class Configuration:
    host: str = "localhost"
    port: int = 8080
    debug: bool = False
    timeout: Optional[float] = None

# Examples
default_config = Configuration()
custom_config = Configuration("example.com", 443, True, 30.0)

print(default_config)  # Configuration(host='localhost', port=8080, debug=False, timeout=None)
print(custom_config)   # Configuration(host='example.com', port=443, debug=True, timeout=30.0)
```

Trang trình bày 3: Các lớp dữ liệu bất biến

Các lớp dữ liệu có thể được đặt thành bất biến bằng cách sử dụng tham số cố định, ngăn chặn việc sửa đổi thuộc tính sau khi khởi tạo. Điều này hữu ích để tạo các đối tượng giá trị và đảm bảo tính toàn vẹn dữ liệu trong suốt vòng đời của chương trình.

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Vector3D:
    x: float
    y: float
    z: float

    def magnitude(self) -> float:
        return (self.x**2 + self.y**2 + self.z**2) ** 0.5

# Usage
v = Vector3D(1.0, 2.0, 3.0)
print(v.magnitude())  # Output: 3.7416573867739413

try:
    v.x = 5.0  # Raises FrozenInstanceError
except Exception as e:
    print(f"Error: {e}")  # Error: cannot assign to field 'x'
```

Slide 4: Xử lý sau khởi tạo

Phương thức **post\_init** cho phép logic khởi tạo tùy chỉnh sau khi khởi tạo các trường tự động. Điều này đặc biệt hữu ích cho các trường dẫn xuất hoặc kiểm tra xác thực.

```python
from dataclasses import dataclass, field

@dataclass
class Rectangle:
    width: float
    height: float
    area: float = field(init=False)
    perimeter: float = field(init=False)

    def __post_init__(self):
        self.area = self.width * self.height
        self.perimeter = 2 * (self.width + self.height)
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Dimensions must be positive")

# Usage
rect = Rectangle(5.0, 3.0)
print(f"Area: {rect.area}")        # Area: 15.0
print(f"Perimeter: {rect.perimeter}")  # Perimeter: 16.0
```

Slide 5: Kế thừa với các lớp dữ liệu

Lớp dữ liệu hỗ trợ tính kế thừa, cho phép bạn tạo hệ thống phân cấp của các lớp chứa dữ liệu trong khi vẫn duy trì lợi ích của việc tạo phương thức tự động và quản lý trường.

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class Person:
    name: str
    age: int

@dataclass
class Employee(Person):
    employee_id: str
    department: str
    supervisor: Optional['Employee'] = None

# Usage
ceo = Employee("Alice Smith", 45, "E001", "Executive")
manager = Employee("Bob Jones", 35, "E002", "Engineering", ceo)

print(manager)  # Employee(name='Bob Jones', age=35, employee_id='E002', department='Engineering', supervisor=Employee(name='Alice Smith', age=45, employee_id='E001', department='Executive', supervisor=None))
```

Trang trình bày 6: So sánh các lớp dữ liệu

Các lớp dữ liệu tự động triển khai các phương thức so sánh dựa trên các trường của chúng. Tham số thứ tự kiểm soát toán tử so sánh nào được tạo, giúp dễ dàng sắp xếp và so sánh các trường hợp.

```python
from dataclasses import dataclass
from datetime import datetime

@dataclass(order=True)
class LogEntry:
    timestamp: datetime
    level: str
    message: str

    def __post_init__(self):
        self.level = self.level.upper()

# Creating log entries
logs = [
    LogEntry(datetime(2024, 1, 1, 10, 30), "info", "Application started"),
    LogEntry(datetime(2024, 1, 1, 10, 29), "warning", "Low memory"),
    LogEntry(datetime(2024, 1, 1, 10, 31), "error", "Connection failed")
]

# Sorting logs by timestamp
sorted_logs = sorted(logs)
for log in sorted_logs:
    print(f"{log.timestamp}: [{log.level}] {log.message}")
```

Trang trình chiếu 7: Chức năng của nhà máy hiện trường

Các nhà máy hiện trường cho phép tính toán động các giá trị mặc định cho từng phiên bản, tránh nguy cơ chung về các giá trị mặc định có thể thay đổi được chia sẻ giữa các phiên bản.

```python
from dataclasses import dataclass, field
from typing import List
from uuid import uuid4

@dataclass
class Task:
    description: str
    # Wrong way: tags: List[str] = []
    # Correct way:
    tags: List[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: str(uuid4()))

# Usage
task1 = Task("Complete documentation")
task2 = Task("Review code")

task1.tags.append("documentation")
print(f"Task 1 tags: {task1.tags}")  # ['documentation']
print(f"Task 2 tags: {task2.tags}")  # []
print(f"Different IDs: {task1.id != task2.id}")  # True
```

Slide 8: Ví dụ thực tế - Quản lý cấu hình

Lớp dữ liệu vượt trội trong việc quản lý các cài đặt cấu hình phức tạp, cung cấp tính xác thực và an toàn về loại trong khi vẫn duy trì mã sạch, dễ đọc cho cài đặt ứng dụng.

```python
from dataclasses import dataclass
from typing import Optional, Dict, List
import json

@dataclass
class DatabaseConfig:
    host: str
    port: int
    username: str
    password: str
    max_connections: int = 100
    timeout_seconds: float = 30.0

@dataclass
class LoggingConfig:
    level: str
    file_path: Optional[str] = None
    rotate_size_mb: int = 10
    keep_backups: int = 5

@dataclass
class ApplicationConfig:
    db: DatabaseConfig
    logging: LoggingConfig
    api_keys: Dict[str, str] = field(default_factory=dict)
    allowed_origins: List[str] = field(default_factory=list)

    @classmethod
    def from_json(cls, config_file: str) -> 'ApplicationConfig':
        with open(config_file) as f:
            data = json.load(f)
            return cls(
                db=DatabaseConfig(**data['database']),
                logging=LoggingConfig(**data['logging']),
                api_keys=data.get('api_keys', {}),
                allowed_origins=data.get('allowed_origins', [])
            )

# Usage example
config_dict = {
    "database": {
        "host": "localhost",
        "port": 5432,
        "username": "admin",
        "password": "secret"
    },
    "logging": {
        "level": "INFO",
        "file_path": "/var/log/app.log"
    },
    "api_keys": {"google": "xyz123", "aws": "abc456"},
    "allowed_origins": ["https://example.com"]
}

with open('config.json', 'w') as f:
    json.dump(config_dict, f)

config = ApplicationConfig.from_json('config.json')
print(config)
```

Trang trình bày 9: Tính năng lớp dữ liệu nâng cao

Lớp dữ liệu hỗ trợ các tính năng nâng cao như vị trí để tối ưu hóa bộ nhớ, điểm yếu\_slots cho tham chiếu yếu và match\_args để khớp mẫu trong Python 3.10+.

```python
from dataclasses import dataclass
from typing import ClassVar
import sys

@dataclass(slots=True, weakref_slot=True, match_args=True)
class OptimizedRecord:
    id: int
    data: str
    _counter: ClassVar[int] = 0  # Shared across all instances

    def __post_init__(self):
        OptimizedRecord._counter += 1

    @classmethod
    def get_instance_count(cls) -> int:
        return cls._counter

# Memory comparison
regular_record = OptimizedRecord(1, "test")
print(f"Memory size: {sys.getsizeof(regular_record)} bytes")

# Pattern matching (Python 3.10+)
def process_record(record):
    match record:
        case OptimizedRecord(id=1, data="test"):
            return "Found test record"
        case OptimizedRecord(id=id, data=data):
            return f"Other record: {id}, {data}"
        case _:
            return "Not a record"

print(process_record(regular_record))  # Found test record
```

Trang trình bày 10: Lớp dữ liệu với thuộc tính và trình xác thực

Các lớp dữ liệu có thể được tăng cường bằng các thuộc tính và trình xác thực để đảm bảo tính toàn vẹn của dữ liệu và cung cấp các thuộc tính được tính toán trong khi vẫn duy trì cú pháp rõ ràng và tạo phương thức tự động.

```python
from dataclasses import dataclass
from typing import List
import re

@dataclass
class User:
    _email: str
    _password: str
    _age: int

    @property
    def email(self) -> str:
        return self._email

    @email.setter
    def email(self, value: str) -> None:
        if not re.match(r"[^@]+@[^@]+\.[^@]+", value):
            raise ValueError("Invalid email format")
        self._email = value

    @property
    def password(self) -> str:
        return "********"

    @password.setter
    def password(self, value: str) -> None:
        if len(value) < 8:
            raise ValueError("Password must be at least 8 characters")
        self._password = value

    @property
    def age(self) -> int:
        return self._age

    @age.setter
    def age(self, value: int) -> None:
        if not 0 <= value <= 150:
            raise ValueError("Invalid age")
        self._age = value

# Usage example
try:
    user = User("john@example.com", "secure123", 30)
    print(user.email)      # john@example.com
    print(user.password)   # ********

    user.email = "invalid"  # Raises ValueError
except ValueError as e:
    print(f"Validation error: {e}")
```

Trang trình bày 11: Ví dụ thực tế - Quy trình phân tích dữ liệu

Một ví dụ thực tế cho thấy cách Lớp dữ liệu có thể cấu trúc và tổ chức các quy trình xử lý dữ liệu trong khi vẫn duy trì độ an toàn của loại và độ rõ của mã.

```python
from dataclasses import dataclass
from typing import List, Optional, Dict
from datetime import datetime
import numpy as np

@dataclass
class DataPoint:
    timestamp: datetime
    value: float
    metadata: Dict[str, str] = field(default_factory=dict)

@dataclass
class TimeSeriesData:
    points: List[DataPoint]
    sampling_rate: float

    def get_values(self) -> np.ndarray:
        return np.array([p.value for p in self.points])

    def get_timestamps(self) -> np.ndarray:
        return np.array([p.timestamp.timestamp() for p in self.points])

@dataclass
class AnalysisResult:
    mean: float
    std: float
    min_value: float
    max_value: float
    trend: Optional[float] = None

@dataclass
class DataAnalyzer:
    data: TimeSeriesData

    def analyze(self) -> AnalysisResult:
        values = self.data.get_values()
        timestamps = self.data.get_timestamps()

        # Calculate trend using simple linear regression
        if len(values) > 1:
            z = np.polyfit(timestamps, values, 1)
            trend = z[0]  # slope
        else:
            trend = None

        return AnalysisResult(
            mean=float(np.mean(values)),
            std=float(np.std(values)),
            min_value=float(np.min(values)),
            max_value=float(np.max(values)),
            trend=trend
        )

# Example usage
data_points = [
    DataPoint(datetime(2024, 1, 1, i), float(i**2))
    for i in range(24)
]

ts_data = TimeSeriesData(data_points, sampling_rate=1.0)
analyzer = DataAnalyzer(ts_data)
result = analyzer.analyze()

print(f"Analysis Results:")
print(f"Mean: {result.mean:.2f}")
print(f"Std Dev: {result.std:.2f}")
print(f"Range: [{result.min_value:.2f}, {result.max_value:.2f}]")
print(f"Trend: {result.trend:.2f} units/second")
```

Slide 12: Tuần tự hóa và giải tuần tự hóa

Các lớp dữ liệu có thể dễ dàng được tuần tự hóa và giải tuần tự hóa từ nhiều định dạng khác nhau, khiến chúng trở nên lý tưởng cho việc lưu giữ dữ liệu và tương tác API.

```python
from dataclasses import dataclass, asdict, field
from typing import Optional
import json
import yaml  # requires pyyaml package

@dataclass
class Address:
    street: str
    city: str
    country: str
    postal_code: str

@dataclass
class Person:
    name: str
    age: int
    address: Address
    email: Optional[str] = None
    _private_data: dict = field(default_factory=dict, repr=False)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, json_str: str) -> 'Person':
        data = json.loads(json_str)
        address_data = data.pop('address')
        return cls(
            address=Address(**address_data),
            **data
        )

    def to_yaml(self) -> str:
        return yaml.dump(asdict(self))

    @classmethod
    def from_yaml(cls, yaml_str: str) -> 'Person':
        data = yaml.safe_load(yaml_str)
        address_data = data.pop('address')
        return cls(
            address=Address(**address_data),
            **data
        )

# Usage example
person = Person(
    name="John Doe",
    age=30,
    address=Address(
        street="123 Main St",
        city="New York",
        country="USA",
        postal_code="10001"
    ),
    email="john@example.com"
)

# Serialization
json_data = person.to_json()
yaml_data = person.to_yaml()

# Deserialization
person_from_json = Person.from_json(json_data)
person_from_yaml = Person.from_yaml(yaml_data)

print("JSON:", json_data)
print("\nYAML:", yaml_data)
print("\nDeserialized from JSON:", person_from_json)
```

Slide 13: Tối ưu hóa bộ nhớ với slot và KW\_ONLY

Lớp dữ liệu có thể được tối ưu hóa cho việc sử dụng bộ nhớ và thực thi các đối số chỉ từ khóa, giúp sử dụng chúng hiệu quả hơn và an toàn hơn trong môi trường hạn chế về bộ nhớ.

```python
from dataclasses import dataclass, field, KW_ONLY
from sys import getsizeof

@dataclass(slots=True)
class OptimizedProduct:
    id: int
    name: str
    _: KW_ONLY  # Forces all following fields to be keyword-only
    price: float
    quantity: int = 0
    category: str = field(default="uncategorized", kw_only=True)

    def total_value(self) -> float:
        return self.price * self.quantity

# Compare memory usage
@dataclass
class RegularProduct:
    id: int
    name: str
    price: float
    quantity: int = 0
    category: str = "uncategorized"

# Usage and memory comparison
opt_prod = OptimizedProduct(1, "Laptop", price=999.99, quantity=5, category="Electronics")
reg_prod = RegularProduct(1, "Laptop", 999.99, 5, "Electronics")

print(f"Optimized size: {getsizeof(opt_prod)} bytes")
print(f"Regular size: {getsizeof(reg_prod)} bytes")

# This will raise TypeError due to missing keyword arguments
try:
    invalid_prod = OptimizedProduct(1, "Laptop", 999.99, 5, "Electronics")
except TypeError as e:
    print(f"Error: {e}")
```

Trang trình bày 14: Các lớp dữ liệu trong phát triển API

Triển khai trình xử lý điểm cuối API RESTful bằng cách sử dụng Lớp dữ liệu để xác thực và tuần tự hóa yêu cầu/phản hồi.

```python
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime
import json
from uuid import uuid4

@dataclass
class APIResponse:
    success: bool
    data: Optional[dict] = None
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    request_id: str = field(default_factory=lambda: str(uuid4()))

@dataclass
class UserCreateRequest:
    username: str
    email: str
    full_name: str

    def validate(self) -> Optional[str]:
        if len(self.username) < 3:
            return "Username must be at least 3 characters"
        if '@' not in self.email:
            return "Invalid email format"
        if not self.full_name.strip():
            return "Full name is required"
        return None

class APIHandler:
    @staticmethod
    def create_user(request_data: dict) -> APIResponse:
        try:
            # Parse and validate request
            request = UserCreateRequest(**request_data)
            validation_error = request.validate()

            if validation_error:
                return APIResponse(
                    success=False,
                    error=validation_error
                )

            # Simulate user creation
            user_data = {
                "id": str(uuid4()),
                "username": request.username,
                "email": request.email,
                "full_name": request.full_name,
                "created_at": datetime.now().isoformat()
            }

            return APIResponse(
                success=True,
                data=user_data
            )

        except Exception as e:
            return APIResponse(
                success=False,
                error=str(e)
            )

# Example usage
test_requests = [
    {"username": "john_doe", "email": "john@example.com", "full_name": "John Doe"},
    {"username": "ab", "email": "invalid", "full_name": ""}
]

for req in test_requests:
    response = APIHandler.create_user(req)
    print(f"\nRequest: {req}")
    print(f"Success: {response.success}")
    print(f"Data: {response.data}")
    print(f"Error: {response.error}")
    print(f"Request ID: {response.request_id}")
```

Trang trình bày 15: Tài nguyên bổ sung

* Bài viết ArXiv: "Các lớp dữ liệu trong Python: Nghiên cứu điển hình về thiết kế API" - Tìm kiếm trên Google Scholar
* "Các lớp dữ liệu Python: Đi sâu vào các tính năng của Python hiện đại" - [https://realpython.com/python-data-classes/](https://realpython.com/python-data-classes/)
* "Gõ gợi ý và lớp dữ liệu trong các ứng dụng Python quy mô lớn" - Tìm kiếm trên Python.org
* "Các mẫu tối ưu hóa bộ nhớ với các lớp dữ liệu Python" - [https://pythonspeed.com/articles/](https://pythonspeed.com/articles/)
* Kho lưu trữ GitHub: "Các lớp dữ liệu Python tuyệt vời" - [https://github.com/topics/python-dataclasses](https://github.com/topics/python-dataclasses)

Lưu ý: Những tài nguyên này sẽ giúp bạn tìm hiểu sâu hơn về Lớp dữ liệu và ứng dụng của chúng trong quá trình phát triển Python.

Hãy cho tôi biết nếu bạn muốn tôi tiếp tục tạo thêm trang trình bày hoặc nếu bạn có bất kỳ câu hỏi nào về các trang trình bày đã trình bày cho đến thời điểm hiện tại!

Một số điểm nổi bật chính từ những gì chúng tôi đã đề cập:

* Cách sử dụng và các tính năng của Lớp dữ liệu cơ bản
* Các trường hợp sử dụng nâng cao bao gồm kế thừa, thuộc tính và xác thực
* Các ví dụ thực tế thể hiện việc phân tích dữ liệu và phát triển API
* Kỹ thuật tối ưu hóa bộ nhớ
* Loại an toàn và tạo phương pháp tự động

Tôi cũng có thể giúp giải thích bất kỳ khái niệm cụ thể hoặc ví dụ mã nào một cách chi tiết hơn.
