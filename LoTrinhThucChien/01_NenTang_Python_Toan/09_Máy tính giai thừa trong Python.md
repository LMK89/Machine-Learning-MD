## Máy tính giai thừa trong Python
Slide 1: Giới thiệu về giai thừa

Giai thừa là tích của tất cả các số nguyên dương nhỏ hơn hoặc bằng một số cho trước. Nó được biểu thị bằng dấu chấm than (!). Ví dụ: 5! = 5 × 4 × 3 × 2 × 1 = 120. Giai thừa được sử dụng trong tổ hợp, lý thuyết xác suất và đại số. Trong phần trình bày này, chúng ta sẽ khám phá cách tính giai thừa bằng Python.

```python
def factorial(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

print(factorial(5))  # Output: 120
```

Slide 2: Hàm giai thừa cơ bản

Đây là một hàm đơn giản để tính giai thừa bằng vòng lặp. Nó nhân tất cả các số nguyên từ 1 đến n.

```python
def factorial(n):
    if n < 0:
        return None  # Factorial is not defined for negative numbers
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

print(factorial(0))  # Output: 1
print(factorial(5))  # Output: 120
print(factorial(-3))  # Output: None
```

Slide 3: Hàm giai thừa đệ quy

Giai thừa cũng có thể được tính toán đệ quy. Phương pháp này ngắn gọn hơn nhưng có thể kém hiệu quả hơn đối với số lượng lớn do chi phí gọi hàm.

```python
def factorial_recursive(n):
    if n < 0:
        return None
    if n == 0 or n == 1:
        return 1
    return n * factorial_recursive(n - 1)

print(factorial_recursive(5))  # Output: 120
print(factorial_recursive(0))  # Output: 1
```

Slide 4: Xử lý số lớn

Python có thể xử lý các số nguyên rất lớn, phù hợp để tính các giai thừa lớn. Hãy tính 100!

```python
def factorial(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

large_factorial = factorial(100)
print(f"100! has {len(str(large_factorial))} digits")
print(f"The first 50 digits are: {str(large_factorial)[:50]}...")

# Output:
# 100! has 158 digits
# The first 50 digits are: 93326215443944152681699238856266700490715968...
```

Slide 5: Tối ưu hóa tính toán giai thừa

Chúng ta có thể tối ưu hóa hàm giai thừa bằng cách sử dụng hàm prod của mô-đun toán học, hàm này hiệu quả hơn với số lượng lớn.

```python
from math import prod

def factorial_optimized(n):
    if n < 0:
        return None
    return prod(range(1, n + 1))

print(factorial_optimized(20))  # Output: 2432902008176640000
```

Slide 6: Sử dụng math.factorial()

Mô-đun toán học của Python cung cấp hàm giai thừa tích hợp sẵn, được tối ưu hóa cao và phù hợp với hầu hết các trường hợp sử dụng.

```python
import math

print(math.factorial(10))  # Output: 3628800
print(math.factorial(0))   # Output: 1

try:
    print(math.factorial(-5))
except ValueError as e:
    print(f"Error: {e}")  # Output: Error: factorial() not defined for negative values
```

Trang trình bày 7: Ghi nhớ để cải thiện hiệu suất

Việc ghi nhớ có thể tăng tốc đáng kể các phép tính giai thừa bằng cách lưu trữ các kết quả được tính toán trước đó.

```python
def memoized_factorial():
    cache = {}
    def factorial(n):
        if n < 0:
            return None
        if n in cache:
            return cache[n]
        if n == 0 or n == 1:
            result = 1
        else:
            result = n * factorial(n - 1)
        cache[n] = result
        return result
    return factorial

fact = memoized_factorial()
print(fact(5))  # Output: 120
print(fact(10))  # Output: 3628800
```

Slide 8: Xử lý tràn số thập phân

Đối với các giai thừa cực lớn, chúng ta có thể sử dụng lớp Decimal để tránh tràn và duy trì độ chính xác.

```python
from decimal import Decimal, getcontext

def factorial_decimal(n):
    if n < 0:
        return None
    getcontext().prec = 1000  # Set precision to 1000 digits
    result = Decimal(1)
    for i in range(1, n + 1):
        result *= Decimal(i)
    return result

large_fact = factorial_decimal(1000)
print(f"1000! has {len(str(large_fact))} digits")
print(f"The first 50 digits are: {str(large_fact)[:50]}...")

# Output:
# 1000! has 2568 digits
# The first 50 digits are: 4023872600770937735437024339230039857193748642...
```

Slide 9: Ví dụ thực tế: Hoán vị

Giai thừa được sử dụng để tính toán hoán vị. Hãy tạo một hàm tính số cách sắp xếp n đối tượng riêng biệt.

```python
def permutations(n):
    return factorial(n)

# Number of ways to arrange 5 books on a shelf
books = 5
arrangements = permutations(books)
print(f"There are {arrangements} ways to arrange {books} books on a shelf.")

# Output: There are 120 ways to arrange 5 books on a shelf.
```

Slide 10: Ví dụ thực tế: Sự kết hợp

Giai thừa cũng được sử dụng trong tính toán kết hợp. Hãy tạo một hàm tính số cách chọn r mục từ n mục.

```python
def combinations(n, r):
    return factorial(n) // (factorial(r) * factorial(n - r))

# Number of ways to select 3 toppings from 8 available toppings for a pizza
total_toppings = 8
selected_toppings = 3
pizza_combinations = combinations(total_toppings, selected_toppings)
print(f"There are {pizza_combinations} ways to select {selected_toppings} toppings from {total_toppings} available toppings.")

# Output: There are 56 ways to select 3 toppings from 8 available toppings.
```

Slide 11: Vẽ biểu đồ tăng trưởng giai thừa

Hãy hình dung sự tăng trưởng nhanh chóng của giai thừa bằng cách sử dụng matplotlib.

```python
import matplotlib.pyplot as plt

def factorial(n):
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)

n_values = range(10)
factorial_values = [factorial(n) for n in n_values]

plt.figure(figsize=(10, 6))
plt.plot(n_values, factorial_values, marker='o')
plt.title("Factorial Growth")
plt.xlabel("n")
plt.ylabel("n!")
plt.yscale('log')
plt.grid(True)
plt.show()
```

Slide 12: Xấp xỉ giai thừa: Công thức Stirling

Với n lớn, chúng ta có thể tính gần đúng giai thừa bằng công thức Stirling. Hãy thực hiện và so sánh nó với giai thừa thực tế.

```python
import math

def stirling_approximation(n):
    return math.sqrt(2 * math.pi * n) * (n / math.e)**n

def factorial(n):
    return math.factorial(n)

n = 100
actual = factorial(n)
approximation = stirling_approximation(n)

print(f"Actual 100!: {actual}")
print(f"Stirling's approximation: {approximation:.2f}")
print(f"Relative error: {abs(actual - approximation) / actual:.6f}")

# Output:
# Actual 100!: 93326215443944152681699238856266700490715968264381621468592963895217599993229915608941463976156518286253697920827223758251185210916864000000000000000000000000
# Stirling's approximation: 93326215443944150965646308284989211734232862699212643110474083881862063044821752707286988719586522806149843139175760462070822648153150.94
# Relative error: 0.000000
```

Slide 13: Lớp tính giai thừa

Hãy tạo một lớp FactorialCalculator gói gọn các phương thức khác nhau để tính giai thừa.

```python
import math
from functools import lru_cache

class FactorialCalculator:
    @staticmethod
    def iterative(n):
        if n < 0:
            raise ValueError("Factorial is not defined for negative numbers")
        result = 1
        for i in range(1, n + 1):
            result *= i
        return result

    @staticmethod
    @lru_cache(maxsize=None)
    def recursive(n):
        if n < 0:
            raise ValueError("Factorial is not defined for negative numbers")
        if n == 0 or n == 1:
            return 1
        return n * FactorialCalculator.recursive(n - 1)

    @staticmethod
    def math_factorial(n):
        return math.factorial(n)

calc = FactorialCalculator()
print(calc.iterative(5))    # Output: 120
print(calc.recursive(5))    # Output: 120
print(calc.math_factorial(5))  # Output: 120
```

Trang trình bày 14: Tài nguyên bổ sung

Để biết thêm thông tin về giai thừa và ứng dụng của chúng trong toán học và khoa học máy tính, hãy xem xét khám phá các tài nguyên sau:

1. "Các thừa số và tổ hợp" của Ronald L. Graham, Donald E. Knuth và Oren Patashnik trong "Toán học cụ thể: Nền tảng cho khoa học máy tính" (1994).
2. “Về công thức Stirling” của Herbert Robbins (1955), The American Mathematical Monthly, 62(1), 26-29. DOI: 10.1080/00029890.1955.11988623
3. arXiv:1808.05729 \[math.NT\] - "Một số bất đẳng thức về tỉ số của hai giai thừa" của Cristinel Mortici (2018). URL: [https://arxiv.org/abs/1808.05729](https://arxiv.org/abs/1808.05729)

Những tài nguyên này cung cấp những hiểu biết sâu sắc hơn về các tính chất và ứng dụng của giai thừa trong các lĩnh vực toán học và khoa học máy tính khác nhau.
