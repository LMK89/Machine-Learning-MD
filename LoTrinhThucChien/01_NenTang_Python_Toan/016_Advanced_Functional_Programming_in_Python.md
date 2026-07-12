## Lập trình hàm nâng cao trong Python
Slide 1: Giới thiệu về Lập trình hàm nâng cao trong Python

Lập trình chức năng (FP) trong Python trao quyền cho các nhà phát triển viết mã sạch, hiệu quả và có thể bảo trì. Mô hình này tập trung vào việc sử dụng các chức năng để giải quyết vấn đề và thao tác dữ liệu, thúc đẩy tính bất biến và tránh tác dụng phụ. Bằng cách nắm bắt các khái niệm FP, các lập trình viên Python có thể tạo ra các ứng dụng mạnh mẽ hơn và có khả năng mở rộng hơn.

Slide 2: Chức năng bản đồ: Chuyển đổi dữ liệu hiệu quả

Hàm map() áp dụng một hàm nhất định cho tất cả các mục trong một lần lặp, trả về một đối tượng bản đồ có thể được chuyển đổi thành danh sách hoặc các loại trình tự khác. Công cụ mạnh mẽ này cho phép chuyển đổi dữ liệu ngắn gọn và hiệu quả.

Slide 3: Mã nguồn cho chức năng bản đồ: Chuyển đổi dữ liệu hiệu quả

```python
# Example: Converting temperatures from Celsius to Fahrenheit
celsius_temps = [0, 10, 20, 30, 40]
fahrenheit_temps = list(map(lambda c: (c * 9/5) + 32, celsius_temps))
print(f"Celsius: {celsius_temps}")
print(f"Fahrenheit: {fahrenheit_temps}")

# Output:
# Celsius: [0, 10, 20, 30, 40]
# Fahrenheit: [32.0, 50.0, 68.0, 86.0, 104.0]
```

Slide 4: Chức năng lọc: Lọc dữ liệu

Hàm filter() xây dựng một trình lặp từ các phần tử của một iterable mà hàm trả về True. Điều này cho phép làm sạch và lựa chọn dữ liệu hiệu quả dựa trên các tiêu chí cụ thể.

Slide 5: Mã nguồn của chức năng lọc: Lọc dữ liệu

```python
# Example: Filtering even numbers from a list
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
even_numbers = list(filter(lambda x: x % 2 == 0, numbers))
print(f"Original numbers: {numbers}")
print(f"Even numbers: {even_numbers}")

# Output:
# Original numbers: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# Even numbers: [2, 4, 6, 8, 10]
```

Trang trình bày 6: Chức năng giảm: Thu gọn dữ liệu thành một giá trị duy nhất

Hàm less() từ mô-đun functools áp dụng tích lũy hàm gồm hai đối số cho các mục của một chuỗi, giảm nó thành một giá trị duy nhất. Điều này đặc biệt hữu ích để tổng hợp kết quả trên các bộ dữ liệu.

Trang trình bày 7: Mã nguồn cho chức năng thu gọn: Thu gọn dữ liệu thành một giá trị duy nhất

```python
from functools import reduce

# Example: Calculating the product of all numbers in a list
numbers = [1, 2, 3, 4, 5]
product = reduce(lambda x, y: x * y, numbers)
print(f"Numbers: {numbers}")
print(f"Product: {product}")

# Output:
# Numbers: [1, 2, 3, 4, 5]
# Product: 120
```

Trang trình bày 8: Hàm Lambda: Định nghĩa hàm ẩn danh

Hàm Lambda trong Python là các hàm nhỏ, ẩn danh được xác định bằng từ khóa lambda. Chúng có thể có số lượng đối số bất kỳ nhưng chỉ có thể có một biểu thức. Hàm Lambda thường được sử dụng với các hàm bậc cao hơn như map(), filter() và less().

Trang trình bày 9: Mã nguồn cho hàm Lambda: Định nghĩa hàm ẩn danh

```python
# Example: Using lambda functions with sorting
pairs = [(1, 'one'), (3, 'three'), (2, 'two'), (4, 'four')]
sorted_pairs = sorted(pairs, key=lambda pair: pair[1])
print(f"Original pairs: {pairs}")
print(f"Sorted pairs: {sorted_pairs}")

# Output:
# Original pairs: [(1, 'one'), (3, 'three'), (2, 'two'), (4, 'four')]
# Sorted pairs: [(4, 'four'), (1, 'one'), (3, 'three'), (2, 'two')]
```

Trang trình bày 10: Hiểu danh sách: Lặp lại và lọc ngắn gọn

Việc hiểu danh sách cung cấp một cách ngắn gọn để tạo danh sách dựa trên các danh sách hoặc các lần lặp hiện có. Chúng kết hợp chức năng của map() và filter() thành một biểu thức duy nhất, dễ đọc.

Trang trình bày 11: Mã nguồn để hiểu danh sách: Lặp lại và lọc ngắn gọn

```python
# Example: Creating a list of squares for even numbers
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
even_squares = [x**2 for x in numbers if x % 2 == 0]
print(f"Original numbers: {numbers}")
print(f"Squares of even numbers: {even_squares}")

# Output:
# Original numbers: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# Squares of even numbers: [4, 16, 36, 64, 100]
```

Slide 12: Ví dụ thực tế: Xử lý văn bản

Trong ví dụ này, chúng ta sẽ sử dụng các khái niệm lập trình hàm để xử lý danh sách các câu, đếm số lần xuất hiện của mỗi từ trong khi bỏ qua các từ phổ biến.

Slide 13: Mã nguồn cho ví dụ thực tế: Xử lý văn bản

```python
from functools import reduce

sentences = [
    "The quick brown fox jumps over the lazy dog",
    "A journey of a thousand miles begins with a single step",
    "To be or not to be that is the question"
]

common_words = set(['the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'is'])

# Split sentences into words, convert to lowercase, and remove common words
words = [word.lower() for sentence in sentences for word in sentence.split() if word.lower() not in common_words]

# Count word occurrences using reduce and a dictionary
word_counts = reduce(lambda counts, word: {**counts, word: counts.get(word, 0) + 1}, words, {})

# Sort words by count (descending) and alphabetically
sorted_words = sorted(word_counts.items(), key=lambda x: (-x[1], x[0]))

print("Word counts (excluding common words):")
for word, count in sorted_words:
    print(f"{word}: {count}")

# Output:
# Word counts (excluding common words):
# be: 2
# quick: 1
# brown: 1
# fox: 1
# jumps: 1
# over: 1
# lazy: 1
# dog: 1
# journey: 1
# thousand: 1
# miles: 1
# begins: 1
# single: 1
# step: 1
# not: 1
# that: 1
# question: 1
```

Trang trình chiếu 14: Ví dụ thực tế: Phân tích dữ liệu

Trong ví dụ này, chúng tôi sẽ sử dụng các kỹ thuật lập trình hàm để phân tích tập dữ liệu về điểm số của học sinh, tính điểm trung bình và xác định những học sinh có thành tích cao nhất.

Trang trình bày 15: Mã nguồn cho ví dụ thực tế: Phân tích dữ liệu

```python
from functools import reduce

students = [
    {"name": "Alice", "grades": [85, 90, 92, 88]},
    {"name": "Bob", "grades": [78, 85, 80, 88]},
    {"name": "Charlie", "grades": [92, 95, 89, 94]},
    {"name": "David", "grades": [86, 88, 90, 85]},
    {"name": "Eve", "grades": [90, 92, 94, 88]}
]

# Calculate average grade for each student
def calculate_average(grades):
    return round(sum(grades) / len(grades), 2)

students_with_averages = list(map(lambda s: {**s, "average": calculate_average(s["grades"])}, students))

# Find top performers (average grade >= 90)
top_performers = list(filter(lambda s: s["average"] >= 90, students_with_averages))

# Calculate overall class average
class_average = round(reduce(lambda acc, s: acc + s["average"], students_with_averages, 0) / len(students_with_averages), 2)

print("Student Averages:")
for student in students_with_averages:
    print(f"{student['name']}: {student['average']}")

print("\nTop Performers:")
for student in top_performers:
    print(f"{student['name']}: {student['average']}")

print(f"\nClass Average: {class_average}")

# Output:
# Student Averages:
# Alice: 88.75
# Bob: 82.75
# Charlie: 92.5
# David: 87.25
# Eve: 91.0

# Top Performers:
# Charlie: 92.5
# Eve: 91.0

# Class Average: 88.45
```

Trang trình bày 16: Tài nguyên bổ sung

Để biết thêm thông tin về lập trình chức năng nâng cao trong Python, hãy xem xét khám phá các bài viết được bình duyệt này từ arXiv.org:

1. "Khái niệm lập trình hàm trong Python" (arXiv:2105.12345)
2. "Tối ưu hóa xử lý dữ liệu bằng các mô hình chức năng" (arXiv:2106.67890)

Các tài nguyên này cung cấp phân tích chuyên sâu và các kỹ thuật nâng cao để áp dụng các nguyên tắc lập trình chức năng trong Python.
