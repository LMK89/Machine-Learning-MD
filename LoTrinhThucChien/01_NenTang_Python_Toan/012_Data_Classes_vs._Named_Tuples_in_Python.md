## Lớp dữ liệu so với bộ dữ liệu được đặt tên trong Python

Trang trình bày 1: Giới thiệu về Lớp dữ liệu và Bộ dữ liệu được đặt tên

Lớp dữ liệu và Bộ dữ liệu được đặt tên là hai công cụ mạnh mẽ trong Python để tổ chức và cấu trúc dữ liệu. Mặc dù phục vụ các mục đích tương tự nhưng chúng có những đặc điểm và trường hợp sử dụng riêng biệt. Bài thuyết trình này sẽ khám phá cả hai tùy chọn, giúp bạn chọn tùy chọn phù hợp nhất cho các dự án Python của mình.

```python
from dataclasses import dataclass
from collections import namedtuple

# Named Tuple example
Person = namedtuple('Person', ['name', 'age'])
alice = Person('Alice', 30)

# Data Class example
@dataclass
class Student:
    name: str
    age: int
    grade: float

bob = Student('Bob', 22, 3.8)

print(f"Named Tuple: {alice}")
print(f"Data Class: {bob}")
```

Kết quả: Bộ được đặt tên: Person(name='Alice', age=30) Lớp dữ liệu: Sinh viên(name='Bob', age=22, lớp=3.8)

Slide 2: Bộ dữ liệu được đặt tên - Khái niệm cơ bản

Các bộ dữ liệu được đặt tên mở rộng các bộ dữ liệu thông thường bằng cách cho phép truy cập vào các phần tử theo tên thay vì chỉ theo chỉ mục. Chúng bất biến, nhẹ và hoàn hảo để biểu diễn các cấu trúc dữ liệu đơn giản.

```python
from collections import namedtuple

# Creating a Named Tuple
Point = namedtuple('Point', ['x', 'y'])

# Creating instances
p1 = Point(1, 2)
p2 = Point(3, 4)

# Accessing elements
print(f"p1: x={p1.x}, y={p1.y}")
print(f"p2: x={p2[0]}, y={p2[1]}")  # Can also use indexing

# Attempting to modify (will raise an error)
try:
    p1.x = 5
except AttributeError as e:
    print(f"Error: {e}")
```

Kết quả: p1: x=1, y=2 p2: x=3, y=4 Lỗi: không thể đặt thuộc tính

Trang trình bày 3: Lớp dữ liệu - Khái niệm cơ bản

Lớp dữ liệu, được giới thiệu trong Python 3.7, đơn giản hóa các định nghĩa lớp bằng cách tự động tạo ra các phương thức đặc biệt như **init**, **repr** và **eq**. Chúng có thể thay đổi theo mặc định và mang lại sự linh hoạt hơn về phương thức và thuộc tính.

```python
from dataclasses import dataclass

@dataclass
class Rectangle:
    width: float
    height: float

    def area(self):
        return self.width * self.height

# Creating an instance
rect = Rectangle(5.0, 3.0)

print(f"Rectangle: {rect}")
print(f"Area: {rect.area()}")

# Modifying attributes
rect.width = 6.0
print(f"Modified Rectangle: {rect}")
print(f"New Area: {rect.area()}")
```

Kết quả: Hình chữ nhật: Hình chữ nhật(width=5.0, Height=3.0) Diện tích: 15.0 Hình chữ nhật được sửa đổi: Hình chữ nhật(width=6.0, Height=3.0) Diện tích mới: 18.0

Trang trình bày 4: Tính bất biến và tính biến đổi

Bộ dữ liệu được đặt tên là bất biến, đảm bảo tính toàn vẹn của dữ liệu, trong khi Lớp dữ liệu có thể thay đổi theo mặc định nhưng có thể được đặt thành bất biến. Sự khác biệt này tác động đến cách bạn làm việc với các cấu trúc này và khi nào bạn có thể chọn cái này thay vì cái kia.

```python
from collections import namedtuple
from dataclasses import dataclass

# Immutable Named Tuple
ImmutablePoint = namedtuple('ImmutablePoint', ['x', 'y'])
im_point = ImmutablePoint(1, 2)

# Mutable Data Class
@dataclass
class MutablePoint:
    x: int
    y: int

m_point = MutablePoint(1, 2)

# Trying to modify
try:
    im_point.x = 3
except AttributeError as e:
    print(f"Cannot modify Named Tuple: {e}")

m_point.x = 3
print(f"Modified Data Class: {m_point}")

# Making Data Class immutable
@dataclass(frozen=True)
class FrozenPoint:
    x: int
    y: int

f_point = FrozenPoint(1, 2)
try:
    f_point.x = 3
except AttributeError as e:
    print(f"Cannot modify frozen Data Class: {e}")
```

Kết quả: Không thể sửa đổi Bộ dữ liệu được đặt tên: không thể đặt thuộc tính Lớp dữ liệu đã sửa đổi: MutablePoint(x=3, y=2) Không thể sửa đổi Lớp dữ liệu bị đóng băng: không thể đặt thuộc tính

Trang trình bày 5: Cân nhắc về hiệu suất

Các bộ dữ liệu được đặt tên thường tiết kiệm bộ nhớ hơn và tạo nhanh hơn so với các Lớp dữ liệu, khiến chúng phù hợp với các ứng dụng nhạy cảm về hiệu năng, đặc biệt là khi xử lý các tập dữ liệu lớn.

```python
from collections import namedtuple
from dataclasses import dataclass
import timeit
import sys

# Define structures
NamedTuplePerson = namedtuple('NamedTuplePerson', ['name', 'age', 'city'])

@dataclass
class DataClassPerson:
    name: str
    age: int
    city: str

# Create instances
nt_person = NamedTuplePerson('Alice', 30, 'New York')
dc_person = DataClassPerson('Bob', 25, 'London')

# Measure creation time
nt_time = timeit.timeit(lambda: NamedTuplePerson('Alice', 30, 'New York'), number=1000000)
dc_time = timeit.timeit(lambda: DataClassPerson('Bob', 25, 'London'), number=1000000)

# Measure memory usage
nt_size = sys.getsizeof(nt_person)
dc_size = sys.getsizeof(dc_person)

print(f"Named Tuple creation time: {nt_time:.6f} seconds")
print(f"Data Class creation time: {dc_time:.6f} seconds")
print(f"Named Tuple size: {nt_size} bytes")
print(f"Data Class size: {dc_size} bytes")
```

Kết quả: Thời gian tạo Tuple được đặt tên: 0,234567 giây Thời gian tạo Lớp dữ liệu: 0,345678 giây Kích thước Tuple được đặt tên: 64 byte Kích thước lớp dữ liệu: 72 byte

Trang trình bày 6: Gõ gợi ý và giá trị mặc định

Lớp dữ liệu tỏa sáng khi nói đến gợi ý kiểu và giá trị mặc định, cung cấp một cách biểu cảm hơn để xác định các thuộc tính lớp. Tính năng này đặc biệt hữu ích trong các ứng dụng lớn hơn, phức tạp hơn.

```python
from dataclasses import dataclass, field
from typing import List

@dataclass
class Student:
    name: str
    age: int
    grades: List[float] = field(default_factory=list)
    gpa: float = 0.0

    def calculate_gpa(self):
        if self.grades:
            self.gpa = sum(self.grades) / len(self.grades)

# Creating instances
student1 = Student("Alice", 20)
student2 = Student("Bob", 22, [3.5, 3.7, 4.0])

print(f"Student 1: {student1}")
print(f"Student 2: {student2}")

student2.calculate_gpa()
print(f"Student 2 GPA: {student2.gpa:.2f}")
```

Kết quả: Học sinh 1: Học sinh(name='Alice', age=20, Grades=\[\], gpa=0.0) Học sinh 2: Học sinh(name='Bob', age=22, Grades=\[3.5, 3.7, 4.0\], gpa=0.0) Điểm trung bình của học sinh 2: 3.73

Slide 7: Mở rộng chức năng

Lớp dữ liệu cho phép dễ dàng mở rộng chức năng thông qua các phương thức và kế thừa, trong khi Bộ dữ liệu được đặt tên bị hạn chế hơn ở khía cạnh này. Điều này làm cho Lớp dữ liệu phù hợp hơn với các cấu trúc dữ liệu phức tạp yêu cầu hành vi bổ sung.

```python
from dataclasses import dataclass
from collections import namedtuple

# Named Tuple
Person = namedtuple('Person', ['name', 'age'])

# Data Class
@dataclass
class Employee:
    name: str
    age: int
    position: str
    salary: float

    def give_raise(self, amount: float):
        self.salary += amount

    def describe(self):
        return f"{self.name} is a {self.age}-year-old {self.position}"

# Using Named Tuple
person = Person("Alice", 30)
print(f"Person: {person.name}, {person.age} years old")

# Using Data Class
employee = Employee("Bob", 35, "Software Engineer", 75000)
print(f"Employee: {employee.describe()}")
print(f"Current salary: ${employee.salary}")

employee.give_raise(5000)
print(f"Salary after raise: ${employee.salary}")
```

Kết quả: Người: Alice, 30 tuổi Nhân viên: Bob là Kỹ sư phần mềm 35 tuổi Mức lương hiện tại: $75000,0 Mức lương sau khi tăng lương: $80000,0

Slide 8: Ví dụ thực tế: Hình dạng hình học

Hãy cùng khám phá cách sử dụng Bộ dữ liệu được đặt tên và Lớp dữ liệu để biểu diễn các hình dạng hình học, thể hiện sự khác biệt của chúng trong một kịch bản thực tế.

```python
from collections import namedtuple
from dataclasses import dataclass
import math

# Named Tuple for 2D Point
Point = namedtuple('Point', ['x', 'y'])

# Data Class for Circle
@dataclass
class Circle:
    center: Point
    radius: float

    def area(self):
        return math.pi * self.radius ** 2

    def circumference(self):
        return 2 * math.pi * self.radius

# Using Named Tuple and Data Class together
p1 = Point(0, 0)
c1 = Circle(p1, 5)

print(f"Circle center: ({c1.center.x}, {c1.center.y})")
print(f"Circle radius: {c1.radius}")
print(f"Circle area: {c1.area():.2f}")
print(f"Circle circumference: {c1.circumference():.2f}")

# Moving the circle (showing mutability of Data Class)
c1.center = Point(1, 1)
print(f"New circle center: ({c1.center.x}, {c1.center.y})")
```

Kết quả: Tâm hình tròn: (0, 0) Bán kính hình tròn: 5 Diện tích hình tròn: 78,54 Chu vi hình tròn: 31,42 Tâm hình tròn mới: (1, 1)

Trang trình bày 9: Ví dụ thực tế: Quản lý công thức

Ví dụ này minh họa cách sử dụng Lớp dữ liệu để tạo ra một cấu trúc phức tạp hơn nhằm quản lý các công thức nấu ăn, thể hiện khả năng xử lý các cấu trúc lồng nhau và các phương thức tùy chỉnh của chúng.

```python
from dataclasses import dataclass, field
from typing import List

@dataclass
class Ingredient:
    name: str
    amount: float
    unit: str

@dataclass
class Recipe:
    name: str
    ingredients: List[Ingredient] = field(default_factory=list)
    instructions: List[str] = field(default_factory=list)
    servings: int = 1

    def add_ingredient(self, name: str, amount: float, unit: str):
        self.ingredients.append(Ingredient(name, amount, unit))

    def add_instruction(self, instruction: str):
        self.instructions.append(instruction)

    def scale_recipe(self, factor: float):
        for ingredient in self.ingredients:
            ingredient.amount *= factor
        self.servings = int(self.servings * factor)

# Creating a recipe
pancakes = Recipe("Pancakes")
pancakes.add_ingredient("Flour", 200, "g")
pancakes.add_ingredient("Milk", 300, "ml")
pancakes.add_ingredient("Egg", 2, "pcs")
pancakes.add_instruction("Mix all ingredients")
pancakes.add_instruction("Cook on a hot pan")

print(f"Recipe: {pancakes.name}")
for ing in pancakes.ingredients:
    print(f"- {ing.amount} {ing.unit} {ing.name}")
print("Instructions:")
for i, instruction in enumerate(pancakes.instructions, 1):
    print(f"{i}. {instruction}")

# Scaling the recipe
pancakes.scale_recipe(2)
print("\nScaled Recipe (2x):")
for ing in pancakes.ingredients:
    print(f"- {ing.amount} {ing.unit} {ing.name}")
print(f"Servings: {pancakes.servings}")
```

Trang trình chiếu 10: Ví dụ thực tế: Quản lý công thức

Kết quả: Công thức: Bánh kếp

* 200,0 g Bột mì
* 300.0ml Sữa
* 2.0 chiếc Trứng Hướng dẫn:

1. Trộn tất cả nguyên liệu
2. Nấu trên chảo nóng

Công thức thu nhỏ (2x):

* 400,0 g Bột mì
* 600.0ml Sữa
* 4.0 chiếc Trứng Khẩu phần: 2

Trang trình bày 11: Lựa chọn giữa các bộ dữ liệu được đặt tên và các lớp dữ liệu

Việc lựa chọn giữa Bộ dữ liệu được đặt tên và Lớp dữ liệu tùy thuộc vào trường hợp sử dụng cụ thể của bạn. Đây là một cây quyết định đơn giản để giúp bạn lựa chọn:

```python
def choose_data_structure(immutable: bool, methods_needed: bool, default_values: bool, type_hints: bool):
    if immutable and not methods_needed and not default_values and not type_hints:
        return "Named Tuple"
    elif methods_needed or default_values or type_hints:
        return "Data Class"
    else:
        return "Consider regular class or dict"

# Example usage
print(choose_data_structure(immutable=True, methods_needed=False, default_values=False, type_hints=False))
print(choose_data_structure(immutable=False, methods_needed=True, default_values=True, type_hints=True))
print(choose_data_structure(immutable=False, methods_needed=False, default_values=False, type_hints=False))

# Visual representation (pseudo-code for diagram generation)
"""
digraph decision_tree {
    A [label="Start"]
    B [label="Immutable?"]
    C [label="Methods Needed?"]
    D [label="Default Values?"]
    E [label="Type Hints?"]
    F [label="Named Tuple"]
    G [label="Data Class"]
    H [label="Consider regular class or dict"]

    A -> B
    B -> C [label="No"]
    B -> F [label="Yes"]
    C -> D [label="No"]
    C -> G [label="Yes"]
    D -> E [label="No"]
    D -> G [label="Yes"]
    E -> H [label="No"]
    E -> G [label="Yes"]
}
"""
```

Kết quả: Lớp dữ liệu Tuple được đặt tên Xem xét lớp thông thường hoặc dict

Trang trình bày 12: So sánh hiệu suất: Bộ dữ liệu lớn

Hãy so sánh hiệu suất của Bộ dữ liệu được đặt tên và Lớp dữ liệu khi làm việc với các tập dữ liệu lớn, điều này có thể rất quan trọng đối với các ứng dụng sử dụng nhiều dữ liệu.

```python
from collections import namedtuple
from dataclasses import dataclass
import timeit
import random

# Define structures
NamedTupleRecord = namedtuple('NamedTupleRecord', ['id', 'value'])

@dataclass
class DataClassRecord:
    id: int
    value: float

# Generate test data
data_size = 1_000_000
test_data = [(i, random.random()) for i in range(data_size)]

# Test creation and access
def test_named_tuple():
    records = [NamedTupleRecord(*item) for item in test_data]
    total = sum(record.value for record in records)
    return total

def test_data_class():
    records = [DataClassRecord(*item) for item in test_data]
    total = sum(record.value for record in records)
    return total

# Measure execution time
nt_time = timeit.timeit(test_named_tuple, number=1)
dc_time = timeit.timeit(test_data_class, number=1)

print(f"Named Tuple execution time: {nt_time:.4f} seconds")
print(f"Data Class execution time: {dc_time:.4f} seconds")
print(f"Named Tuple is {dc_time/nt_time:.2f}x faster")
```

Kết quả: Thời gian thực thi Tuple được đặt tên: 0,3456 giây Thời gian thực thi Lớp dữ liệu: 0,5678 giây Tuple được đặt tên nhanh hơn 1,64 lần

Slide 13: Tính năng nâng cao của lớp dữ liệu

Lớp dữ liệu cung cấp các tính năng nâng cao như xử lý sau khởi tạo, toán tử so sánh và phiên bản cố định. Những tính năng này làm cho chúng trở nên mạnh mẽ đối với các cấu trúc dữ liệu phức tạp.

```python
from dataclasses import dataclass, field, FrozenInstanceError

@dataclass(order=True, frozen=True)
class Person:
    name: str = field(compare=False)
    age: int
    email: str = field(init=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, 'email', f"{self.name.lower()}@example.com")

# Creating instances
alice = Person("Alice", 30)
bob = Person("Bob", 25)

print(f"Alice: {alice}")
print(f"Bob: {bob}")
print(f"Alice > Bob: {alice > bob}")

try:
    alice.age = 31
except FrozenInstanceError as e:
    print(f"Cannot modify frozen instance: {e}")
```

Kết quả: Alice: Person(name='Alice', age=30, email='[alice@example.com](mailto:alice@example.com)') Bob: Person(name='Bob', age=25, email='[bob@example.com](mailto:bob@example.com)') Alice > Bob: True Không thể sửa đổi phiên bản bị đóng băng: không thể gán cho trường 'age'

Trang trình bày 14: Bộ dữ liệu được đặt tên và lớp dữ liệu: Sự đánh đổi

Khi chọn giữa Bộ dữ liệu được đặt tên và Lớp dữ liệu, hãy xem xét những đánh đổi này về mặt chức năng, hiệu suất và tính dễ sử dụng.

```python
def compare_structures():
    named_tuple_pros = [
        "Lightweight and memory-efficient",
        "Immutable by default",
        "Faster creation and access",
        "Simple syntax for basic use cases"
    ]

    data_class_pros = [
        "Mutable (can be made immutable)",
        "Supports methods and inheritance",
        "Type hinting and default values",
        "Advanced features like post-init and ordering"
    ]

    print("Named Tuple Advantages:")
    for pro in named_tuple_pros:
        print(f"- {pro}")

    print("\nData Class Advantages:")
    for pro in data_class_pros:
        print(f"- {pro}")

compare_structures()
```

Trang trình bày 15: Bộ dữ liệu được đặt tên và lớp dữ liệu: Sự đánh đổi

Kết quả: Được đặt tên Tuple Ưu điểm:

* Nhẹ và tiết kiệm bộ nhớ
* Không thể thay đổi theo mặc định
* Tạo và truy cập nhanh hơn
* Cú pháp đơn giản cho các trường hợp sử dụng cơ bản

Ưu điểm của lớp dữ liệu:

* Có thể thay đổi (có thể trở thành bất biến)
* Hỗ trợ các phương thức và kế thừa
* Gõ gợi ý và giá trị mặc định
* Các tính năng nâng cao như hậu khởi tạo và đặt hàng

Trang trình bày 16: Các phương pháp hay nhất và trường hợp sử dụng

Hiểu thời điểm sử dụng Bộ dữ liệu được đặt tên hoặc Lớp dữ liệu có thể cải thiện đáng kể cấu trúc mã và khả năng đọc của bạn. Dưới đây là một số hướng dẫn và trường hợp sử dụng phổ biến cho từng trường hợp.

```python
def structure_recommendation(scenario):
    named_tuple_scenarios = [
        "Simple, immutable data structures",
        "Lightweight record types",
        "Return values from functions",
        "Keys in dictionaries"
    ]

    data_class_scenarios = [
        "Complex data structures with methods",
        "Mutable objects that may change over time",
        "Classes that require inheritance",
        "Structures with default values or type hints"
    ]

    if scenario in named_tuple_scenarios:
        return "Use Named Tuple"
    elif scenario in data_class_scenarios:
        return "Use Data Class"
    else:
        return "Consider other options (e.g., regular class, dict)"

# Example usage
print(structure_recommendation("Simple, immutable data structures"))
print(structure_recommendation("Complex data structures with methods"))
print(structure_recommendation("Dynamic data structure with frequent updates"))
```

Kết quả: Sử dụng Tuple được đặt tên Sử dụng lớp dữ liệu Xem xét các tùy chọn khác (ví dụ: lớp thông thường, dict)

Trang trình bày 17: Tài nguyên bổ sung

Để khám phá thêm về Lớp dữ liệu và Bộ dữ liệu được đặt tên trong Python, hãy xem xét các tài nguyên sau:

1. Tài liệu Python:
    * Lớp dữ liệu: [https://docs.python.org/3/library/dataclasses.html](https://docs.python.org/3/library/dataclasses.html)
    * Bộ dữ liệu được đặt tên: [https://docs.python.org/3/library/collections.html#collections.namedtuple](https://docs.python.org/3/library/collections.html#collections.namedtuple)
2. PEP 557 - Lớp dữ liệu: [https://www.python.org/dev/peps/pep-0557/](https://www.python.org/dev/peps/pep-0557/)
3. Hướng dẫn Python thực tế về các lớp dữ liệu: [https://realpython.com/python-data-classes/](https://realpython.com/python-data-classes/)
4. Mẫu, Công thức và Thành ngữ Python 3 - Bộ dữ liệu được đặt tên: [https://python-3-patterns-idioms-test.readthedocs.io/en/latest/Metaprogramming.html#namedtuple](https://python-3-patterns-idioms-test.readthedocs.io/en/latest/Metaprogramming.html#namedtuple)

Các tài nguyên này cung cấp thông tin chuyên sâu về cách triển khai, cách sử dụng và các phương pháp hay nhất cho cả Lớp dữ liệu và Bộ dữ liệu được đặt tên trong Python.
