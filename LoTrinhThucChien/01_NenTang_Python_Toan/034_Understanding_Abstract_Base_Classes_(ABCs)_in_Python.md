## Tìm hiểu các lớp cơ sở trừu tượng (ABC) trong Python
Slide 1: Giới thiệu về các lớp cơ sở trừu tượng

Các lớp cơ sở trừu tượng (ABC) cung cấp một cách để xác định các giao diện trong Python, thực thi một hợp đồng mà các lớp dẫn xuất phải thực hiện. Chúng hoạt động như một kế hoạch chi tiết cho các lớp khác, thiết lập một tập hợp các phương thức và thuộc tính mà việc triển khai cụ thể phải cung cấp.

```python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        pass

    @abstractmethod
    def perimeter(self):
        pass

# This will raise TypeError if instantiated directly
# shape = Shape()  # TypeError: Can't instantiate abstract class
```

Slide 2: Triển khai các lớp trừu tượng

Các lớp trừu tượng xác định một hợp đồng giao diện mà các lớp con phải tuân theo. Khi một lớp kế thừa từ một lớp cơ sở trừu tượng, nó phải triển khai tất cả các phương thức trừu tượng, nếu không nó sẽ gây ra TypeError khi được khởi tạo.

```python
class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height

    def perimeter(self):
        return 2 * (self.width + self.height)

# Valid instantiation
rect = Rectangle(5, 3)
print(f"Area: {rect.area()}")  # Output: Area: 15
```

Slide 3: Thuộc tính và phương thức trừu tượng

Các lớp trừu tượng có thể định nghĩa cả phương thức trừu tượng và thuộc tính trừu tượng, yêu cầu các lớp triển khai phải cung cấp cả giao diện hành vi và dữ liệu. Điều này đảm bảo thực hiện đầy đủ hợp đồng.

```python
class Vehicle(ABC):
    @property
    @abstractmethod
    def fuel_type(self):
        pass

    @abstractmethod
    def start_engine(self):
        pass

class ElectricCar(Vehicle):
    @property
    def fuel_type(self):
        return "electricity"

    def start_engine(self):
        return "Starting electric motor"
```

Trang trình bày 4: Nhiều lớp cơ sở trừu tượng

Python hỗ trợ kế thừa từ nhiều lớp cơ sở trừu tượng, cho phép kết hợp giao diện phức tạp. Điều này cho phép định nghĩa hợp đồng linh hoạt trong khi vẫn duy trì các yêu cầu thực hiện nghiêm ngặt.

```python
class Drawable(ABC):
    @abstractmethod
    def draw(self): pass

class Moveable(ABC):
    @abstractmethod
    def move(self): pass

class GameSprite(Drawable, Moveable):
    def draw(self):
        return "Drawing sprite"

    def move(self):
        return "Moving sprite"
```

Trang trình bày 5: Ví dụ thực tế - Quy trình xử lý dữ liệu

Các lớp cơ sở trừu tượng vượt trội trong việc xác định các quy trình xử lý trong đó các cách triển khai khác nhau có thể xử lý các loại dữ liệu hoặc nguồn khác nhau trong khi vẫn duy trì một giao diện nhất quán.

```python
class DataProcessor(ABC):
    @abstractmethod
    def load_data(self, source):
        pass

    @abstractmethod
    def process(self, data):
        pass

    @abstractmethod
    def save_result(self, result, destination):
        pass

class CSVProcessor(DataProcessor):
    def load_data(self, source):
        return f"Loading CSV from {source}"

    def process(self, data):
        return f"Processing {data}"

    def save_result(self, result, destination):
        return f"Saving to {destination}"
```

Slide 6: Các phương pháp trừu tượng và triển khai

Các lớp trừu tượng có thể cung cấp các triển khai mặc định trong khi vẫn yêu cầu ghi đè phương thức, cung cấp cả tính linh hoạt và hành vi mặc định khi cần.

```python
class DataValidator(ABC):
    @abstractmethod
    def validate(self, data):
        # Default implementation
        if not data:
            return False
        return True

class NumericValidator(DataValidator):
    def validate(self, data):
        # Must call super() to use default implementation
        if not super().validate(data):
            return False
        return isinstance(data, (int, float))

validator = NumericValidator()
print(validator.validate(42))  # Output: True
```

Trang trình bày 7: Các lớp cơ sở trừu tượng với siêu dữ liệu

Hiểu siêu dữ liệu trong các lớp cơ sở trừu tượng cung cấp khả năng kiểm soát sâu hơn đối với việc tạo và xác thực lớp, cho phép hành vi tùy chỉnh trong quá trình định nghĩa lớp.

```python
from abc import ABCMeta

class ValidatorMeta(ABCMeta):
    def __new__(mcls, name, bases, namespace):
        for key, value in namespace.items():
            if getattr(value, "_validation_required", False):
                if not hasattr(value, "validate"):
                    raise TypeError(f"{key} must implement validate()")
        return super().__new__(mcls, name, bases, namespace)

class BaseValidator(metaclass=ValidatorMeta):
    pass
```

Slide 8: Mẫu thiết kế với ABC - Observer Pattern

Các lớp cơ sở trừu tượng là nền tảng trong việc triển khai các mẫu thiết kế. Đây là cách triển khai mẫu Observer bằng cách sử dụng ABC.

```python
class Subject(ABC):
    def __init__(self):
        self._observers = []

    def attach(self, observer):
        self._observers.append(observer)

    def detach(self, observer):
        self._observers.remove(observer)

    @abstractmethod
    def notify(self):
        pass

class Observer(ABC):
    @abstractmethod
    def update(self, subject):
        pass
```

Trang trình bày 9: Mã nguồn để triển khai mẫu quan sát

```python
class ConcreteSubject(Subject):
    def __init__(self):
        super().__init__()
        self._state = None

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        self._state = value
        self.notify()

    def notify(self):
        for observer in self._observers:
            observer.update(self)

class ConcreteObserver(Observer):
    def update(self, subject):
        print(f"Observer updated with state: {subject.state}")

# Usage
subject = ConcreteSubject()
observer = ConcreteObserver()
subject.attach(observer)
subject.state = "New State"  # Output: Observer updated with state: New State
```

Trang trình bày 10: Mẫu phương thức mẫu sử dụng ABC

Mẫu Phương thức mẫu xác định khung của thuật toán trong lớp cơ sở trong khi cho phép các lớp con ghi đè các bước cụ thể mà không thay đổi cấu trúc của thuật toán.

```python
class DataMiner(ABC):
    def mine(self, path):
        raw_data = self._extract(path)
        clean_data = self._transform(raw_data)
        return self._load(clean_data)

    @abstractmethod
    def _extract(self, path):
        pass

    @abstractmethod
    def _transform(self, data):
        pass

    @abstractmethod
    def _load(self, data):
        pass
```

Slide 11: Source Code for Template Method Implementation

```python
class PDFMiner(DataMiner):
    def _extract(self, path):
        return f"Extracting PDF data from {path}"

    def _transform(self, data):
        return f"Transforming PDF data: {data}"

    def _load(self, data):
        return f"Loading transformed PDF data: {data}"

class CSVMiner(DataMiner):
    def _extract(self, path):
        return f"Extracting CSV data from {path}"

    def _transform(self, data):
        return f"Transforming CSV data: {data}"

    def _load(self, data):
        return f"Loading transformed CSV data: {data}"

# Usage
pdf_miner = PDFMiner()
result = pdf_miner.mine("document.pdf")
print(result)  # Output: Loading transformed PDF data: Transforming PDF data: Extracting PDF data from document.pdf
```

Slide 12: Unit Test với ABCs

Các lớp cơ sở trừu tượng cung cấp nền tảng mạnh mẽ cho thử nghiệm đơn vị, cho phép các trường hợp thử nghiệm xác minh rằng việc triển khai cụ thể đáp ứng giao diện được yêu cầu.

```python
import unittest

class TestDataProcessor(unittest.TestCase):
    def test_processor_implementation(self):
        class TestProcessor(DataProcessor):
            def load_data(self, source): return "data"
            def process(self, data): return "processed"
            def save_result(self, result, dest): return "saved"

        processor = TestProcessor()
        self.assertTrue(isinstance(processor, DataProcessor))
        self.assertEqual(processor.load_data("test"), "data")

if __name__ == '__main__':
    unittest.main()
```

Slide 13: Tính năng ABC nâng cao - Thuộc tính lớp trừu tượng

Hiểu các tính năng nâng cao của ABC bao gồm làm việc với các thuộc tính lớp và các phương thức tĩnh trong khi vẫn duy trì hợp đồng trừu tượng.

```python
class PaymentProcessor(ABC):
    @classmethod
    @abstractmethod
    def get_processor_name(cls):
        pass

    @staticmethod
    @abstractmethod
    def validate_currency(currency_code):
        pass

class StripeProcessor(PaymentProcessor):
    @classmethod
    def get_processor_name(cls):
        return "Stripe"

    @staticmethod
    def validate_currency(currency_code):
        return currency_code in ['USD', 'EUR', 'GBP']
```

Trang trình bày 14: Tài nguyên bổ sung

* [https://arxiv.org/abs/1809.03193](https://arxiv.org/abs/1809.03193) - "Các mẫu thiết kế trong Python: Đánh giá tài liệu có hệ thống"
* [https://arxiv.org/abs/2007.08983](https://arxiv.org/abs/2007.08983) - "Phát hiện mẫu thiết kế hướng đối tượng bằng máy học"
* [https://arxiv.org/abs/1906.11678](https://arxiv.org/abs/1906.11678) - "Về tác động của việc trừu tượng hóa ngôn ngữ lập trình"
* [https://arxiv.org/abs/2012.14631](https://arxiv.org/abs/2012.14631) - "Tự động phát hiện mùi mã Python"
