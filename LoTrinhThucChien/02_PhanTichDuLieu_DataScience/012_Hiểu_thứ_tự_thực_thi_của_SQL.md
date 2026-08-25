## Tìm hiểu thực thi thứ tự của SQL

Trang trình bày 1: SQL là ngôn ngữ khai báo

SQL hoạt động theo nguyên tắc mô tả những gì bạn muốn, thay vì đánh vần từng bước tính toán. Triết lý thiết kế này làm cho SQL trở nên độc lập trong số các ngôn ngữ lập trình - bạn khai báo kết quả mong muốn của mình và công cụ SQL sẽ xác định đường dẫn hiệu quả nhất để đạt được kết quả đó.

```python
# Example showing declarative vs imperative approach
# Declarative (SQL-like) approach in Python
data = [
    {'name': 'Alice', 'age': 25},
    {'name': 'Bob', 'age': 30},
    {'name': 'Charlie', 'age': 35}
]

# Using list comprehension (declarative-style)
adults = [person for person in data if person['age'] >= 30]

# Imperative approach
adults = []
for person in data:
    if person['age'] >= 30:
        adults.append(person)
```

Slide 2: Truy vấn cấu trúc

Truy vấn SQL phong phú theo cấu trúc logic trong đó các mệnh đề được sắp xếp theo một thứ tự cụ thể. Khi được viết theo một trình tự, công việc thực thi sẽ đi theo một đường dẫn khác, hiệu suất tối ưu và toàn bộ dữ liệu.

```python
def demonstrate_query_structure():
    query = {
        'select': ['column1', 'column2'],
        'from': 'table_name',
        'where': 'condition',
        'group_by': 'column1',
        'having': 'group_condition',
        'order_by': 'column1',
        'limit': 10
    }
    return query
```

Slide 3: FROM and JOIN Operations

The first step in query execution involves identifying and combining data sources. This forms the foundation of all subsequent operations.

```python
def demonstrate_join():
    table1 = [('A', 1), ('B', 2), ('C', 3)]
    table2 = [(1, 'X'), (2, 'Y'), (3, 'Z')]

    # Simulating an INNER JOIN
    joined_data = []
    for t1 in table1:
        for t2 in table2:
            if t1[1] == t2[0]:  # Join condition
                joined_data.append((t1[0], t1[1], t2[1]))
    return joined_data
```

Trang trình bày 4: Xử lý mệnh đề WHERE

Sau khi các nguồn dữ liệu được kết hợp, quá trình lọc diễn ra thông qua mệnh đề WHERE. Bước này sẽ loại bỏ các hàng không đáp ứng các điều kiện được chỉ định.

```python
def filter_data(data, condition):
    # Simulating WHERE clause
    return [
        row for row in data
        if eval(f"row[condition['column']] {condition['operator']} {condition['value']}")
    ]

# Example usage
data = [{'age': 25}, {'age': 30}, {'age': 35}]
condition = {'column': 'age', 'operator': '>', 'value': 30}
filtered = filter_data(data, condition)
```

Slide 5: Thực hiện NHÓM THEO

Thao tác GROUP BY tổng hợp các hàng có giá trị chung, tạo nền tảng cho tổng hợp các hàm.

```python
from collections import defaultdict

def group_data(data, group_column):
    groups = defaultdict(list)
    for row in data:
        key = row[group_column]
        groups[key].append(row)
    return dict(groups)

# Example data
data = [
    {'category': 'A', 'value': 1},
    {'category': 'B', 'value': 2},
    {'category': 'A', 'value': 3}
]
grouped = group_data(data, 'category')
```
Slide 5: GROUP BY Implementation

GROUP BY transforms individual rows into grouped sets based on specified columns, preparing data for aggregate operations like counting or averaging values.

```python
def simple_group_by(data):
    # Sample data representing colors and their occurrences
    colors = ['red', 'blue', 'red', 'green', 'blue', 'red']

    # Dictionary to store grouped counts
    grouped_data = {}

    # Group and count occurrences
    for color in colors:
        if color in grouped_data:
            grouped_data[color] += 1
        else:
            grouped_data[color] = 1

    return grouped_data
```

Trang trình bày 6: Kết quả phát triển khai GROUP BY

```python
# Output of simple_group_by():
{
    'red': 3,
    'blue': 2,
    'green': 1
}
```

Slide 7: HAVING Clause

The HAVING clause filters grouped data based on aggregate conditions, operating after GROUP BY has formed the groups.

```python
def apply_having(grouped_data, min_count):
    # Filter groups based on count threshold
    filtered_groups = {
        color: count
        for color, count in grouped_data.items()
        if count >= min_count
    }
    return filtered_groups

# Usage example with minimum count of 2
result = apply_having({'red': 3, 'blue': 2, 'green': 1}, 2)
```

Slide 8: CHỌN Xử lý

CHỌN cột xác định nào xuất hiện trong kết quả cuối cùng, có thể bao gồm các giá trị được tính toán hoặc tổng hợp.

```python
def process_select(data, columns):
    # Sample data processing with SELECT-like behavior
    selected_data = []

    for record in data:
        selected_record = {}
        for col in columns:
            if col in record:
                selected_record[col] = record[col]
        selected_data.append(selected_record)

    return selected_data
```

Slide 9: ORDER BY Implementation

ORDER BY sorts the final result set based on specified columns and sort directions.

```python
def custom_sort(data, sort_key, ascending=True):
    # Implementation of basic sorting mechanism
    sorted_data = sorted(
        data,
        key=lambda x: x[sort_key],
        reverse=not ascending
    )
    return sorted_data
```

Slide 10: LIMIT hoạt động

LIMIT check Kiểm soát số lượng hàng ở đầu ra cuối cùng, hữu ích cho việc phân trang và giảm khối lượng dữ liệu.

```python
def apply_limit(data, limit_value):
    # Simple implementation of LIMIT
    return data[:limit_value] if limit_value > 0 else data
```

Slide 11: Real-Life Example - Student Records

This example demonstrates a complete query execution flow using student attendance records.

```python
def process_student_records():
    # Sample student attendance data
    records = [
        {'student': 'Alice', 'subject': 'Math', 'attendance': 90},
        {'student': 'Bob', 'subject': 'Math', 'attendance': 85},
        {'student': 'Alice', 'subject': 'Science', 'attendance': 95}
    ]

    # Group by student
    grouped = {}
    for record in records:
        student = record['student']
        if student not in grouped:
            grouped[student] = []
        grouped[student].append(record)

    # Calculate average attendance per student
    averages = {
        student: sum(r['attendance'] for r in records) / len(records)
        for student, records in grouped.items()
    }

    return averages
```

Slide 12: Ví dụ thực tế - Phân tích dữ liệu thời gian

Ví dụ này cho thấy cách xử lý và phân tích các chỉ số nhiệt độ.

```python
def analyze_temperature_readings():
    # Sample temperature readings throughout a day
    readings = [
        {'hour': 1, 'temp': 20}, {'hour': 2, 'temp': 19},
        {'hour': 3, 'temp': 18}, {'hour': 4, 'temp': 20}
    ]

    # Group by temperature value
    temp_groups = {}
    for reading in readings:
        temp = reading['temp']
        if temp not in temp_groups:
            temp_groups[temp] = []
        temp_groups[temp].append(reading['hour'])

    # Find most frequent temperature
    most_frequent = max(temp_groups.items(), key=lambda x: len(x[1]))

    return {
        'temp': most_frequent[0],
        'occurrences': len(most_frequent[1]),
        'at_hours': most_frequent[1]
    }
```

Slide 13: Additional Resources

For deeper understanding of SQL query execution and optimization, refer to:

*   "Query Optimization Techniques in Database Systems" (arXiv:1911.03834)
*   "A Survey of Query Execution Engine and Query Optimization" (arXiv:2111.02668)
