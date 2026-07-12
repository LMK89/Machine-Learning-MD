## Làm chủ SQL bằng Python với PandaSQL
Trang trình bày 1: Giới thiệu về PandaSQL

PandaSQL là một thư viện mạnh mẽ giúp thu hẹp khoảng cách giữa SQL và DataFrames của gấu trúc trong Python. Nó cho phép người dùng viết các truy vấn SQL trực tiếp trên DataFrames của gấu trúc, kết hợp sự quen thuộc của SQL với tính linh hoạt của gấu trúc.

```python
import pandas as pd
import pandasql as ps

# Create a sample DataFrame
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'city': ['New York', 'London', 'Paris']
})

# Run a SQL query on the DataFrame
query = "SELECT * FROM df WHERE age > 28"
result = ps.sqldf(query, locals())
print(result)
```

Trang trình bày 2: Thiết lập PandaSQL

Để bắt đầu với PandaSQL, bạn cần cài đặt nó bằng pip. Sau khi cài đặt, bạn có thể nhập nó cùng với gấu trúc để bắt đầu truy vấn DataFrames của mình.

```python
# Install PandaSQL
!pip install pandasql

# Import necessary libraries
import pandas as pd
import pandasql as ps

# Create a sample DataFrame
df = pd.DataFrame({
    'product': ['A', 'B', 'C', 'A', 'B'],
    'quantity': [10, 20, 15, 5, 25],
    'price': [100, 200, 150, 100, 180]
})

print(df)
```

Slide 3: Truy vấn SQL cơ bản với PandaSQL

PandaSQL cho phép bạn viết các truy vấn SQL dưới dạng chuỗi và thực thi chúng trên DataFrames của gấu trúc. Hãy bắt đầu với một truy vấn CHỌN đơn giản để truy xuất tất cả các hàng từ DataFrame của chúng tôi.

```python
# Basic SELECT query
query = "SELECT * FROM df"
result = ps.sqldf(query, locals())
print(result)
```

Slide 4: Filtering Data with WHERE Clause

You can use the WHERE clause in your SQL queries to filter data based on specific conditions. This is equivalent to using boolean indexing in pandas.

```python
# Filtering data with WHERE clause
query = "SELECT * FROM df WHERE quantity > 15"
result = ps.sqldf(query, locals())
print(result)

# Equivalent pandas operation
pandas_result = df[df['quantity'] > 15]
print("\nPandas equivalent:")
print(pandas_result)
```

Slide 5: Tổng hợp dữ liệu với GROUP BY

PandaSQL hỗ trợ các tập hợp SQL bằng cách sử dụng GROUP BY, tương tự như phương thức groupby() của gấu trúc, theo sau là các hàm tổng hợp.

```python
# Aggregating data with GROUP BY
query = """
SELECT product, SUM(quantity) as total_quantity, AVG(price) as avg_price
FROM df
GROUP BY product
"""
result = ps.sqldf(query, locals())
print(result)

# Equivalent pandas operation
pandas_result = df.groupby('product').agg({'quantity': 'sum', 'price': 'mean'})
pandas_result.columns = ['total_quantity', 'avg_price']
print("\nPandas equivalent:")
print(pandas_result)
```

Trang trình bày 6: Tham gia DataFrames

PandaSQL cho phép bạn tham gia nhiều DataFrames bằng cú pháp SQL JOIN, cú pháp này có thể trực quan hơn đối với những người quen thuộc với SQL so với hàm pandas merge().

```python
# Create two sample DataFrames
df1 = pd.DataFrame({'id': [1, 2, 3], 'name': ['Alice', 'Bob', 'Charlie']})
df2 = pd.DataFrame({'id': [2, 3, 4], 'city': ['London', 'Paris', 'Berlin']})

# Join DataFrames using SQL
query = """
SELECT df1.id, df1.name, df2.city
FROM df1
LEFT JOIN df2 ON df1.id = df2.id
"""
result = ps.sqldf(query, locals())
print(result)

# Equivalent pandas operation
pandas_result = pd.merge(df1, df2, on='id', how='left')
print("\nPandas equivalent:")
print(pandas_result)
```

Trang trình bày 7: Truy vấn con và các thao tác phức tạp

PandaSQL hỗ trợ các truy vấn con và các hoạt động SQL phức tạp, đôi khi có thể đơn giản hơn các hoạt động của gấu trúc lồng nhau.

```python
# Create a sample DataFrame
df = pd.DataFrame({
    'department': ['A', 'A', 'B', 'B', 'C'],
    'employee': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'salary': [50000, 60000, 55000, 65000, 70000]
})

# Use a subquery to find employees with above-average salary
query = """
SELECT department, employee, salary
FROM df
WHERE salary > (SELECT AVG(salary) FROM df)
"""
result = ps.sqldf(query, locals())
print(result)

# Equivalent pandas operation
avg_salary = df['salary'].mean()
pandas_result = df[df['salary'] > avg_salary]
print("\nPandas equivalent:")
print(pandas_result)
```

Trang trình bày 8: Các hàm cửa sổ trong PandaSQL

PandaSQL hỗ trợ các chức năng cửa sổ, có thể được sử dụng cho các hoạt động như tính tổng hoặc xếp hạng. Chúng tương tự như các phương thức mở rộng() và xếp hạng() của gấu trúc.

```python
# Create a sample DataFrame
df = pd.DataFrame({
    'date': pd.date_range(start='2023-01-01', periods=5),
    'sales': [100, 150, 200, 120, 180]
})

# Use window function for cumulative sum
query = """
SELECT date, sales,
       SUM(sales) OVER (ORDER BY date) as cumulative_sales
FROM df
"""
result = ps.sqldf(query, locals())
print(result)

# Equivalent pandas operation
df['cumulative_sales'] = df['sales'].cumsum()
print("\nPandas equivalent:")
print(df)
```

Trang trình bày 9: Ví dụ thực tế: Phân tích kết quả học tập của sinh viên

Hãy sử dụng PandaSQL để phân tích dữ liệu hiệu suất của học sinh, chứng minh cách sử dụng dữ liệu đó trong bối cảnh giáo dục.

```python
# Create a sample DataFrame of student scores
students_df = pd.DataFrame({
    'student_id': range(1, 11),
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve',
             'Frank', 'Grace', 'Henry', 'Ivy', 'Jack'],
    'math_score': [85, 92, 78, 95, 88, 72, 90, 83, 79, 94],
    'science_score': [92, 88, 75, 89, 95, 80, 85, 88, 92, 86],
    'literature_score': [78, 85, 90, 82, 87, 88, 91, 76, 84, 89]
})

# Calculate average scores and rank students
query = """
SELECT name,
       (math_score + science_score + literature_score) / 3.0 as avg_score,
       RANK() OVER (ORDER BY (math_score + science_score + literature_score) DESC) as rank
FROM students_df
ORDER BY avg_score DESC
"""
result = ps.sqldf(query, locals())
print(result)
```

Trang trình bày 10: Ví dụ thực tế: Phân tích dữ liệu cảm biến

Trong ví dụ này, chúng tôi sẽ sử dụng PandaSQL để phân tích dữ liệu cảm biến, thể hiện ứng dụng của nó trong các tình huống giám sát môi trường và IoT.

```python
# Create a sample DataFrame of sensor readings
import numpy as np

np.random.seed(42)
dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='H')
sensor_df = pd.DataFrame({
    'timestamp': dates,
    'temperature': np.random.normal(20, 5, len(dates)),
    'humidity': np.random.normal(60, 10, len(dates)),
    'air_quality': np.random.normal(50, 20, len(dates))
})

# Analyze daily averages and flag unusual readings
query = """
SELECT
    DATE(timestamp) as date,
    AVG(temperature) as avg_temp,
    AVG(humidity) as avg_humidity,
    AVG(air_quality) as avg_air_quality,
    CASE
        WHEN AVG(temperature) > 25 OR AVG(humidity) > 70 OR AVG(air_quality) > 100
        THEN 'Alert'
        ELSE 'Normal'
    END as status
FROM sensor_df
GROUP BY DATE(timestamp)
HAVING status = 'Alert'
ORDER BY date
"""
result = ps.sqldf(query, locals())
print(result)
```

Trang trình bày 11: Cân nhắc về hiệu suất

Mặc dù PandaSQL cung cấp giao diện SQL quen thuộc nhưng điều quan trọng là phải xem xét các tác động về hiệu suất, đặc biệt đối với các tập dữ liệu lớn hoặc các truy vấn phức tạp.

```python
import time

# Create a larger DataFrame
large_df = pd.DataFrame({
    'id': range(100000),
    'value': np.random.randn(100000)
})

# Measure time for PandaSQL query
start_time = time.time()
query = "SELECT * FROM large_df WHERE value > 0"
result = ps.sqldf(query, locals())
pandasql_time = time.time() - start_time

# Measure time for equivalent pandas operation
start_time = time.time()
pandas_result = large_df[large_df['value'] > 0]
pandas_time = time.time() - start_time

print(f"PandaSQL time: {pandasql_time:.4f} seconds")
print(f"Pandas time: {pandas_time:.4f} seconds")
```

Trang trình bày 12: Các phương pháp và mẹo hay nhất

Khi sử dụng PandaSQL, hãy xem xét các phương pháp hay nhất sau để tối ưu hóa quy trình làm việc và hiệu suất truy vấn của bạn:

1. Sử dụng PandaSQL cho các truy vấn phức tạp trong đó cú pháp SQL trực quan hơn.
2. Đối với các thao tác đơn giản, hãy sử dụng các phương pháp gấu trúc bản địa để có hiệu suất tốt hơn.
3. Tận dụng sự hỗ trợ của PandaSQL cho các chức năng và truy vấn phụ của cửa sổ khi thích hợp.
4. Hãy chú ý đến việc sử dụng bộ nhớ, đặc biệt là với các tập dữ liệu lớn.
5. Sử dụng lập chỉ mục thích hợp trong DataFrames gấu trúc của bạn để tăng tốc các truy vấn PandaSQL.
6. Luôn so sánh hiệu suất của các truy vấn PandaSQL với các hoạt động tương đương của gấu trúc đối với các tác vụ quan trọng.

```python
# Example of using appropriate indexing
df = pd.DataFrame({
    'id': range(1000000),
    'value': np.random.randn(1000000)
})
df.set_index('id', inplace=True)

# PandaSQL query using the index
query = "SELECT * FROM df WHERE id BETWEEN 500000 AND 500010"
result = ps.sqldf(query, locals())
print(result)
```

Slide 13: Kết luận và định hướng tương lai

PandaSQL thu hẹp khoảng cách giữa SQL và pandas, cung cấp một công cụ mạnh mẽ để phân tích dữ liệu bằng Python. Nó đặc biệt hữu ích cho những người chuyển từ SQL sang pandas hoặc làm việc trong môi trường mà SQL là ngôn ngữ truy vấn chính. Khi nhu cầu xử lý dữ liệu phát triển, các thư viện như PandaSQL có thể tiếp tục thích ứng, có khả năng kết hợp các tính năng như:

1. Hỗ trợ các tính năng SQL nâng cao hơn
2. Cải thiện tối ưu hóa hiệu suất
3. Tích hợp với công nghệ dữ liệu lớn

Hãy theo dõi dự án PandaSQL để biết các bản cập nhật và cải tiến trong tương lai.

```python
# Example of a more advanced query combining multiple features
query = """
WITH ranked_data AS (
    SELECT *,
           RANK() OVER (PARTITION BY department ORDER BY salary DESC) as salary_rank
    FROM df
)
SELECT department, employee, salary, salary_rank
FROM ranked_data
WHERE salary_rank <= 2
ORDER BY department, salary_rank
"""
result = ps.sqldf(query, locals())
print(result)
```

Slide 14: Additional Resources

For those interested in diving deeper into PandaSQL and related topics, here are some valuable resources:

1.  PandaSQL GitHub Repository: [https://github.com/yhat/pandasql](https://github.com/yhat/pandasql)
2.  "Pandas: Powerful Python Data Analysis Toolkit" by Wes McKinney (ArXiv:1402.1726): [https://arxiv.org/abs/1402.1726](https://arxiv.org/abs/1402.1726)
3.  "SQL for Data Scientists: A Beginner's Guide for Building Datasets for Analysis" by Renee M. P. Teate
4.  Pandas Documentation: [https://pandas.pydata.org/docs/](https://pandas.pydata.org/docs/)
5.  SQLite Documentation (PandaSQL uses SQLite under the hood): [https://www.sqlite.org/docs.html](https://www.sqlite.org/docs.html)

These resources will help you further explore the capabilities of PandaSQL and enhance your data analysis skills in Python.
