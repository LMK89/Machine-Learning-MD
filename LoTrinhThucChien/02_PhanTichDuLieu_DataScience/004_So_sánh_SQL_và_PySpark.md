## So sánh SQL và PySpark
Slide 1: Giới thiệu về SQL và PySpark

Thao tác và phân tích dữ liệu có thể được thực hiện bằng cả SQL và PySpark. Công nghệ này phục vụ các mục tiêu tương tự nhưng hoạt động khác nhau. SQL là ngôn ngữ tiêu chuẩn cho hệ thống cơ sở dữ liệu, trong khi PySpark là API Python cho Apache Spark, được thiết kế để xử lý dữ liệu lớn.

```python
# SQL Example
sql_query = """
SELECT name, age
FROM users
WHERE age > 25"""

# PySpark Equivalent
from pyspark.sql import SparkSession
spark_df.select("name", "age").filter("age > 25")
```

Trang trình bày 2: Tạo bảng và dữ liệu khung

SQL tạo các bảng trong hệ thống cơ sở dữ liệu, trong khi PySpark tạo các phân tán DataFrames trong bộ nhớ.

```python
# SQL
create_table = """
CREATE TABLE employees (
    id INT,
    name VARCHAR(50),
    department VARCHAR(50)
)"""

# PySpark
from pyspark.sql.types import *
schema = StructType([
    StructField("id", IntegerType(), True),
    StructField("name", StringType(), True),
    StructField("department", StringType(), True)
])
df = spark.createDataFrame([], schema)
```

Slide 3: Lựa chọn dữ liệu

Cả SQL và PySpark đều cung cấp các cách để chọn các cột cụ thể và lọc dữ liệu. Cú pháp khác nhau nhưng khái niệm vẫn tương tự.

```python
# SQL
sql_query = """
SELECT name, age
FROM employees
WHERE department = 'IT'"""

# PySpark
df.select("name", "age").filter(col("department") == "IT")
```

Slide 4: Tập hợp

Thực hiện các nhóm hoạt động và tổng hợp là cơ sở trong dữ liệu phân tích. Cả hai công nghệ đều cung cấp khả năng tổng hợp mạnh mẽ.

```python
# SQL
sql_query = """
SELECT department, COUNT(*) as count, AVG(salary) as avg_salary
FROM employees
GROUP BY department"""

# PySpark
df.groupBy("department").agg(
    count("*").alias("count"),
    avg("salary").alias("avg_salary")
)
```

Slide 5: Ví dụ thực tế - Thời tiết

Phân tích Phân tích các nhiệt độ chỉ số từ nhiều trạm thời gian ở các thành phố khác nhau.

```python
# SQL
sql_query = """
SELECT city,
       AVG(temperature) as avg_temp,
       COUNT(*) as readings
FROM weather_readings
WHERE year = 2023
GROUP BY city
HAVING COUNT(*) > 100"""

# PySpark
weather_df.filter(col("year") == 2023)\
    .groupBy("city")\
    .agg(
        avg("temperature").alias("avg_temp"),
        count("*").alias("readings")
    )\
    .filter(col("readings") > 100)
```

Slide 6: Ví dụ thực tế - Sinh viên

Phân tích thành tích Phân tích số học của sinh viên trong các môn học khác nhau và tính toán chỉ số thành tích.

```python
# SQL
sql_query = """
SELECT subject,
       AVG(score) as avg_score,
       COUNT(DISTINCT student_id) as student_count
FROM exam_results
GROUP BY subject
HAVING AVG(score) < 75"""

# PySpark
exam_df.groupBy("subject")\
    .agg(
        avg("score").alias("avg_score"),
        countDistinct("student_id").alias("student_count")
    )\
    .filter(col("avg_score") < 75)
```

Trang trình bày 7: Tham gia SQL và PySpark

Cả hai nền tảng đều hỗ trợ nhiều loại kết nối khác nhau để kết hợp dữ liệu từ nhiều nguồn.

```python
# SQL
sql_query = """
SELECT s.name, c.course_name
FROM students s
LEFT JOIN courses c
ON s.course_id = c.id"""

# PySpark
students_df.join(
    courses_df,
    students_df.course_id == courses_df.id,
    "left"
)
```

Slide 8: Chức năng của cửa sổ

Cửa sổ chức năng cho phép tính toán trên một tập hợp các liên kết đến hiện tại hàng.

```python
# SQL
sql_query = """
SELECT name,
       score,
       AVG(score) OVER (PARTITION BY subject) as avg_subject_score
FROM exam_results"""

# PySpark
from pyspark.sql.window import Window
window_spec = Window.partitionBy("subject")
exam_df.withColumn(
    "avg_subject_score",
    avg("score").over(window_spec)
)
```

Slide 9: Thiếu xử lý

Các cách tiếp cận khác có giá trị để xử lý giá trị null trong cả SQL và PySpark.

```python
# SQL
sql_query = """
SELECT name,
       COALESCE(age, 0) as age,
       NULLIF(department, 'Unknown') as dept
FROM employees"""

# PySpark
df.na.fill({"age": 0})\
    .withColumn(
        "dept",
        when(col("department") == "Unknown", None)\
        .otherwise(col("department"))
    )
```

Slide 10: Thao tác trên chuỗi

Cả SQL và PySpark đều cung cấp các hàm để thao tác chuỗi.

```python
# SQL
sql_query = """
SELECT UPPER(name) as upper_name,
       SUBSTRING(description, 1, 10) as short_desc
FROM products"""

# PySpark
from pyspark.sql.functions import upper, substring
df.select(
    upper("name").alias("upper_name"),
    substring("description", 1, 10).alias("short_desc")
)
```

Slide 11: Các loại tạp dữ liệu phức tạp

Xử lý mảng và cấu hình trong nền hai nền.

```python
# SQL
sql_query = """
SELECT name,
       tags[1] as first_tag,
       metadata->>'city' as city
FROM products"""

# PySpark
df.select(
    "name",
    col("tags").getItem(0).alias("first_tag"),
    col("metadata.city")
)
```

Trang trình bày 12: Hiệu suất tối ưu

Cả SQL và PySpark đều cung cấp các cách để tối ưu hóa hiệu suất truy vấn.

```python
# SQL with indexing
sql_query = """
CREATE INDEX idx_department
ON employees(department)
WHERE department IS NOT NULL"""

# PySpark with caching
df.cache()  # Cache DataFrame in memory
df.repartition(10)  # Optimize partitioning
```

Trang trình bày 13: Tài nguyên bổ sung

Để biết thêm thông tin chi tiết về tích hợp SQL và PySpark, hãy tham khảo:

* "Máy tính phân giải với PySpark SQL: Một nghiên cứu so sánh" (arXiv:2103.07538)
* "Phân tích hiệu suất của SparkSQL so với truyền thống SQL" (arXiv:1906.04516)

Những bài viết này cung cấp những tính năng so sánh và phân tích hiệu suất của cả hai công nghệ.
