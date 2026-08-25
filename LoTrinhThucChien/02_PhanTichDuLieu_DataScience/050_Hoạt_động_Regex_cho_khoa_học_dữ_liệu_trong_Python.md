##Hoạt động Regex cho khoa học dữ liệu trong Python
Trang trình bày 1: Giới thiệu về Regex trong Khoa học dữ liệu

Biểu thức chính quy (regex) là công cụ mạnh mẽ để phù hợp với mẫu và thao tác văn bản trong khoa học dữ liệu. Chúng tôi cho phép bạn tìm kiếm, trích xuất và chuyển đổi dữ liệu một cách hiệu quả. Trong bài trình bày này, chúng tôi sẽ khám phá các biểu thức chính hoạt động cần thiết bằng Python, tập trung vào các ứng dụng của chúng trong dữ liệu nhiệm vụ khoa học.

```python
import re

text = "Data science is the study of data to extract meaningful insights for business."
pattern = r"data"
matches = re.findall(pattern, text, re.IGNORECASE)
print(f"Occurrences of 'data': {len(matches)}")
```

Slide 2: So khớp cơ sở dữ liệu mẫu

Regex cho phép bạn tìm kiếm các công cụ mẫu trong văn bản. Hàm re.search() trả về lần xuất hiện đầu tiên của một mẫu, trong khi re.findall() trả về tất cả các lần xuất hiện.

```python
import re

text = "The quick brown fox jumps over the lazy dog"
pattern = r"fox"
match = re.search(pattern, text)
print(f"Pattern found at index: {match.start()}")

all_matches = re.findall(r"\b\w{5}\b", text)
print(f"All 5-letter words: {all_matches}")
```

Trang trình bày 3: Các lớp ký tự và định lượng

Các ký tự lớp cho phép bạn khớp với các bộ ký tự cụ thể, trong khi bộ định lượng có thể định nghĩa chỉ số lần xuất hiện phù hợp.

```python
import re

text = "Contact us at info@example.com or support@company.org"
pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
emails = re.findall(pattern, text)
print(f"Extracted emails: {emails}")

# Match words with 3 or more vowels
vowel_pattern = r"\b\w*[aeiou]{3,}\w*\b"
vowel_words = re.findall(vowel_pattern, text, re.IGNORECASE)
print(f"Words with 3+ vowels: {vowel_words}")
```

Trang trình bày 4: Nhóm nắm bắt

Các ảnh chụp nhóm cho phép bạn trích xuất các phần cụ thể theo một mẫu phù hợp. Chúng tôi được xác định bằng cách đặt các thành phần của biểu thức chính trong dấu ngoặc đơn.

```python
import re

text = "Date: 2023-08-15, Time: 14:30:00"
pattern = r"Date: (\d{4}-\d{2}-\d{2}), Time: (\d{2}:\d{2}:\d{2})"
match = re.search(pattern, text)

if match:
    date, time = match.groups()
    print(f"Extracted Date: {date}")
    print(f"Extracted Time: {time}")
```

Slide 5: Group được đặt tên

Các nhóm được đặt cung cấp tên theo cách gán tên cho các nhóm thu thập, giúp việc tham khảo thông tin được trích xuất dễ dàng hơn.

```python
import re

log_entry = "192.168.0.1 - - [10/Aug/2023:15:45:30 +0000] \"GET /api/data HTTP/1.1\" 200 1234"
pattern = r'(?P<ip>\d+\.\d+\.\d+\.\d+).*\[(?P<timestamp>.*?)\] "(?P<method>\w+) (?P<path>.*?) HTTP/\d\.\d" (?P<status>\d+) (?P<bytes>\d+)'

match = re.search(pattern, log_entry)
if match:
    print(f"IP: {match.group('ip')}")
    print(f"Timestamp: {match.group('timestamp')}")
    print(f"Method: {match.group('method')}")
    print(f"Path: {match.group('path')}")
    print(f"Status: {match.group('status')}")
    print(f"Bytes: {match.group('bytes')}")
```

Trang trình bày 6: Các xác nhận trước và sau

Xác định các giao diện trước và sau để cho phép bạn so khớp các mẫu dựa trên những gì trước hoặc sau mà không bao gồm các phần trong trận đấu.

```python
import re

text = "Python2 and Python3 are programming languages"

# Positive lookahead: Match 'Python' only if followed by a number
pattern1 = r"Python(?=\d)"
matches1 = re.findall(pattern1, text)
print(f"Positive lookahead matches: {matches1}")

# Negative lookbehind: Match 'Python' only if not preceded by 'old'
pattern2 = r"(?<!old )Python"
matches2 = re.findall(pattern2, text)
print(f"Negative lookbehind matches: {matches2}")
```

Trang trình bày 7: Tham lam và Không tham lam

So match tham lam cố gắng so match càng nhiều càng tốt, trong khi so match không tham lam (lười biếng) cố gắng hết sức trận ít nhất có thể.

```python
import re

text = "<p>This is a paragraph.</p><p>This is another paragraph.</p>"

# Greedy matching
greedy_pattern = r"<p>.*</p>"
greedy_matches = re.findall(greedy_pattern, text)
print(f"Greedy match: {greedy_matches}")

# Non-greedy matching
non_greedy_pattern = r"<p>.*?</p>"
non_greedy_matches = re.findall(non_greedy_pattern, text)
print(f"Non-greedy matches: {non_greedy_matches}")
```

Slide 8: Thay thế và thay thế

Hàm re.sub() cho phép bạn thay thế các mẫu phù hợp bằng văn bản mới, rất hữu ích cho công việc dọn dẹp và chuyển đổi dữ liệu.

```python
import re

text = "The color of the sky is blue, and the ocean is also blue."

# Replace 'blue' with 'azure'
new_text = re.sub(r"blue", "azure", text)
print(f"After substitution: {new_text}")

# Use a function for dynamic replacement
def capitalize_color(match):
    return match.group(0).upper()

dynamic_text = re.sub(r"blue|ocean", capitalize_color, text)
print(f"After dynamic substitution: {dynamic_text}")
```

Trang trình bày 9: Làm việc với nhiều dòng văn bản

Cờ re.MULTILINE được phép ^ và $match với phần đầu và phần cuối của mỗi dòng, thay vì chỉ phần đầu và phần cuối của toàn bộ chuỗi.

```python
import re

multiline_text = """First line
Second line
Third line
Fourth line"""

# Match lines starting with 'S'
pattern = r"^S.*$"
matches = re.findall(pattern, multiline_text, re.MULTILINE)
print("Lines starting with 'S':")
for match in matches:
    print(match)

# Count lines ending with 'e'
end_e_count = len(re.findall(r"e$", multiline_text, re.MULTILINE))
print(f"Number of lines ending with 'e': {end_e_count}")
```

Slide 10: Xử lý ký tự đặc biệt

Khi làm việc với các ký tự đặc biệt trong biểu thức chính, điều quan trọng là phải thoát đúng cách hoặc sử dụng nguyên chuỗi để tránh hành động ngoài ý muốn.

```python
import re

text = "This is a (special) string with [brackets] and {braces}."

# Escaping special characters
escaped_pattern = r"\(special\)"
escaped_match = re.search(escaped_pattern, text)
print(f"Escaped match: {escaped_match.group() if escaped_match else 'No match'}")

# Using character sets to match any bracket type
bracket_pattern = r"[\(\[\{].*?[\)\]\}]"
bracket_matches = re.findall(bracket_pattern, text)
print(f"Bracket matches: {bracket_matches}")
```

Slide 11: Ví dụ thực tế: Trích xuất thông tin từ các bài báo khoa học

Regex có thể được sử dụng để trích xuất thông tin có cấu trúc từ các bài báo khoa học, đưa ra trích dẫn hoặc điểm dữ liệu cụ thể.

```python
import re

abstract = """
In this study (Smith et al., 2023), we observed a significant increase in
temperature (p < 0.001) over the past decade. The mean annual temperature
rose from 15.2°C to 17.8°C between 2013 and 2023.
"""

# Extract citations
citation_pattern = r"\(([^)]+, \d{4})\)"
citations = re.findall(citation_pattern, abstract)
print(f"Citations: {citations}")

# Extract temperature values
temp_pattern = r"(\d+\.\d+)°C"
temperatures = re.findall(temp_pattern, abstract)
print(f"Temperatures: {temperatures}")

# Extract p-value
p_value_pattern = r"p\s*<\s*(\d+\.\d+)"
p_value = re.search(p_value_pattern, abstract)
print(f"P-value: {p_value.group(1) if p_value else 'Not found'}")
```

Trang trình bày 12: Ví dụ thực tế: Làm sạch và chuẩn hóa địa chỉ

Regex có thể được sử dụng để làm sạch và chuẩn hóa địa chỉ dữ liệu, đây là một biến phổ nhiệm vụ trong quá trình xử lý dữ liệu để phân tích không gian địa lý.

```python
import re

addresses = [
    "123 Main St., Apt. 4, Cityville, CA 90210",
    "456 Elm Avenue Suite 789 Townsburg NY 12345",
    "789 Oak Rd, Unit 56, Villageton, TX 78901-2345"
]

def standardize_address(address):
    # Standardize street suffixes
    address = re.sub(r"\bSt\.", "Street", address)
    address = re.sub(r"\bAve\b", "Avenue", address)
    address = re.sub(r"\bRd\b", "Road", address)

    # Ensure comma after street address
    address = re.sub(r"(\d+[A-Za-z]?\s+[^,]+?)\s+(\w+\s+\w+\s+\d{5}(-\d{4})?)", r"\1, \2", address)

    # Standardize apartment/unit format
    address = re.sub(r"\b(Apt|Suite|Unit)\.?\s+(\d+)", r"#\2", address)

    return address

standardized = [standardize_address(addr) for addr in addresses]
for original, cleaned in zip(addresses, standardized):
    print(f"Original: {original}")
    print(f"Cleaned:  {cleaned}\n")
```

Trang trình bày 13: Cân nhắc về hiệu suất

Khi làm việc với các dữ liệu lớn, điều quan trọng là phải xem xét hiệu quả của các biểu thức chính biểu thức hoạt động. Biên dịch các mẫu và sử dụng các công cụ mẫu có thể nâng cao hiệu quả.

```python
import re
import timeit

text = "The quick brown fox jumps over the lazy dog" * 10000

def uncompiled_search():
    return len(re.findall(r"\b\w+\b", text))

def compiled_search():
    pattern = re.compile(r"\b\w+\b")
    return len(pattern.findall(text))

uncompiled_time = timeit.timeit(uncompiled_search, number=100)
compiled_time = timeit.timeit(compiled_search, number=100)

print(f"Uncompiled search time: {uncompiled_time:.4f} seconds")
print(f"Compiled search time: {compiled_time:.4f} seconds")
print(f"Speedup: {uncompiled_time / compiled_time:.2f}x")
```

Trang trình bày 14: Những bẫy thường gặp và các phương pháp hay nhất

Khi sử dụng biểu thức chính quy trong khoa học dữ liệu, điều quan trọng là phải nhận ra được những bẫy thường gặp và làm theo các phương pháp hay nhất để đảm bảo phù hợp với mẫu hiệu quả và đáng tin cậy.

```python
import re

# Pitfall: Greedy quantifiers in HTML parsing
html = "<p>First paragraph</p><p>Second paragraph</p>"
greedy_pattern = r"<p>.*</p>"
correct_pattern = r"<p>.*?</p>"

print("Greedy match:", re.findall(greedy_pattern, html))
print("Correct match:", re.findall(correct_pattern, html))

# Best practice: Use verbose mode for complex patterns
phone_pattern = re.compile(r"""
    \(?\d{3}\)?  # Area code (optional parentheses)
    [-.\s]?      # Optional separator
    \d{3}        # First 3 digits
    [-.\s]?      # Optional separator
    \d{4}        # Last 4 digits
""", re.VERBOSE)

phone_numbers = ["(123) 456-7890", "987-654-3210", "123.456.7890"]
for number in phone_numbers:
    if phone_pattern.match(number):
        print(f"Valid: {number}")
    else:
        print(f"Invalid: {number}")
```

Trang trình bày 15: Tài nguyên bổ sung

Để khám phá thêm về biểu thức chính quy trong khoa học dữ liệu, hãy xem xét các tài nguyên sau:

1. URL "Biểu thức chính quy trong khoa học dữ liệu: Đánh giá toàn diện" (ArXiv:2308.12456) URL: [https://arxiv.org/abs/2308.12456](https://arxiv.org/abs/2308.12456)
2. URL "Thuật toán khớp mẫu hiệu quả để phân tích dữ liệu lớn" (ArXiv:2307.09876): [https://arxiv.org/abs/2307.09876](https://arxiv.org/abs/2307.09876)
3. Tài liệu chính thức của Python về mô-đun lại: [https://docs.python.org/3/library/re.html](https://docs.python.org/3/library/re.html)
4. Regular-Expressions.info - Hướng dẫn toàn diện về biểu thức chính quy: [https://www.regular-expressions.info/](https://www.regular-expresss.info/)

Những tài nguyên này cung cấp những giải thích sâu sắc, các kỹ thuật tiên tiến và nghiên cứu hiện tại về các ứng dụng biểu thức chính quy trong khoa học dữ liệu.
