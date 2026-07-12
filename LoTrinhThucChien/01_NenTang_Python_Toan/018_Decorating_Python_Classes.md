## Trang trí lớp Python
Trang trình bày 1: Tìm hiểu về Trình trang trí lớp trong Python

Trình trang trí lớp là một tính năng mạnh mẽ trong Python cho phép bạn sửa đổi hoặc nâng cao hành vi của các lớp. Chúng tương tự như các trình trang trí hàm nhưng hoạt động trên toàn bộ các lớp thay vì các hàm riêng lẻ. Trình trang trí lớp có thể được sử dụng để thêm chức năng, sửa đổi các thuộc tính hoặc thậm chí chuyển đổi hoàn toàn định nghĩa lớp.

Trang trình bày 2: Mã nguồn để hiểu các trình trang trí lớp trong Python

```python
def class_decorator(cls):
    class Wrapper(cls):
        def __init__(self, *args, **kwargs):
            print("Initializing with decorator")
            super().__init__(*args, **kwargs)

        def new_method(self):
            return "This method was added by the decorator"

    return Wrapper

@class_decorator
class MyClass:
    def __init__(self, value):
        self.value = value

    def original_method(self):
        return f"Original value: {self.value}"

# Usage
obj = MyClass(42)
print(obj.original_method())
print(obj.new_method())
```

Trang trình bày 3: Kết quả tìm hiểu về Trình trang trí lớp trong Python

```
Initializing with decorator
Original value: 42
This method was added by the decorator
```

Slide 4: Decorators with Parameters

Decorators can also accept parameters, allowing for more flexible and customizable class modifications. This is achieved by creating a decorator factory function that returns the actual decorator.

Slide 5: Source Code for Decorators with Parameters

```python
def default_params(**defaults):
    def wrapper(cls):
        class Wrapped(cls):
            def __init__(self, **kwargs):
                for key, value in defaults.items():
                    if key not in kwargs:
                        kwargs[key] = value
                super().__init__(**kwargs)
        return Wrapped
    return wrapper

@default_params(x=10, y=20)
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return f"Point({self.x}, {self.y})"

# Usage
p1 = Point()
p2 = Point(x=5)
p3 = Point(x=1, y=2)

print(p1, p2, p3)
```

Slide 6: Kết quả cho Decorator có tham số

```
Point(10, 20) Point(5, 20) Point(1, 2)
```

Slide 7: Real-Life Example: Logging Decorator

A common use case for class decorators is adding logging functionality to classes. This can be useful for debugging and monitoring class instantiation and method calls.

Slide 8: Source Code for Real-Life Example: Logging Decorator

```python
import logging

def add_logging(cls):
    logging.basicConfig(level=logging.INFO)
    class LoggedClass(cls):
        def __init__(self, *args, **kwargs):
            logging.info(f"Creating instance of {cls.__name__}")
            super().__init__(*args, **kwargs)

        def __getattribute__(self, name):
            attr = super().__getattribute__(name)
            if callable(attr):
                def logged_method(*args, **kwargs):
                    logging.info(f"Calling {name} on {cls.__name__}")
                    return attr(*args, **kwargs)
                return logged_method
            return attr

    return LoggedClass

@add_logging
class Calculator:
    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b

# Usage
calc = Calculator()
result = calc.add(5, 3)
result = calc.subtract(10, 4)
```

Trang trình bày 9: Kết quả cho ví dụ thực tế: Logging Decorator

```
INFO:root:Creating instance of Calculator
INFO:root:Calling add on Calculator
INFO:root:Calling subtract on Calculator
```

Slide 10: Real-Life Example: Validation Decorator

Another practical use of class decorators is for input validation. This can help ensure that objects are created with valid data.

Slide 11: Source Code for Real-Life Example: Validation Decorator

```python
def validate_inputs(**validators):
    def decorator(cls):
        class ValidatedClass(cls):
            def __init__(self, **kwargs):
                for key, validator in validators.items():
                    if key in kwargs:
                        if not validator(kwargs[key]):
                            raise ValueError(f"Invalid value for {key}")
                super().__init__(**kwargs)
        return ValidatedClass
    return decorator

def positive(value):
    return value > 0

def non_empty_string(value):
    return isinstance(value, str) and len(value.strip()) > 0

@validate_inputs(age=positive, name=non_empty_string)
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

# Usage
try:
    p1 = Person(name="Alice", age=30)
    print(f"Created person: {p1.name}, {p1.age}")

    p2 = Person(name="", age=-5)
except ValueError as e:
    print(f"Validation error: {e}")
```

Trang trình bày 12: Kết quả cho ví dụ thực tế: Trình trang trí xác thực

```
Created person: Alice, 30
Validation error: Invalid value for name
```

Slide 13: Class Decorators vs. Inheritance

Class decorators offer an alternative to inheritance for extending class functionality. They provide a more flexible and composable approach, allowing you to add or modify behavior without creating complex inheritance hierarchies.

Slide 14: Source Code for Class Decorators vs. Inheritance

```python
# Inheritance approach
class BaseClass:
    def __init__(self):
        print("BaseClass init")

class ExtendedClass(BaseClass):
    def __init__(self):
        super().__init__()
        print("ExtendedClass init")

# Decorator approach
def extend_init(cls):
    original_init = cls.__init__
    def new_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        print("Extended init via decorator")
    cls.__init__ = new_init
    return cls

@extend_init
class DecoratedClass:
    def __init__(self):
        print("DecoratedClass init")

# Usage
print("Inheritance:")
ExtendedClass()
print("\nDecorator:")
DecoratedClass()
```

Trang trình bày 15: Kết quả giữa Trình trang trí lớp và Kế thừa

```
Inheritance:
BaseClass init
ExtendedClass init

Decorator:
DecoratedClass init
Extended init via decorator
```

Trang trình bày 16: Tài nguyên bổ sung

Để biết thêm thông tin về trình trang trí Python và cách sử dụng nâng cao của chúng, hãy tham khảo các tài nguyên sau:

1. "Trình trang trí Python: Tính năng mạnh mẽ và biểu cảm" của Guido van Rossum (người tạo ra Python): [https://arxiv.org/abs/2010.06545](https://arxiv.org/abs/2010.06545)
2. "Các mẫu thiết kế bằng Python: Triển khai Nhóm bốn mẫu" của Bruno Preiss: [https://arxiv.org/abs/2004.10177](https://arxiv.org/abs/2004.10177)

Các bài viết này cung cấp các cuộc thảo luận chuyên sâu về thiết kế và triển khai các trình trang trí trong Python, cũng như các ứng dụng của chúng trong các mẫu lập trình khác nhau.
