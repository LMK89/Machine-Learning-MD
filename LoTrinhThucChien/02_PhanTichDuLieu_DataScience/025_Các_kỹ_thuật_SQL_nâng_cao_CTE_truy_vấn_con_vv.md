## Cao kỹ thuật SQL nâng cao! CTE, truy vấn phụ và hơn thế nữa
Trang trình bày 1: Chung biểu thức bảng (CTE)

Chung biểu thức bảng (CTE)

CTE là tệp kết quả được đặt tạm thời tồn tại trong phạm vi của một lệnh SQL. Họ đơn giản hóa các truy vấn phức tạp bằng cách chia chúng thành các phần nhỏ hơn, dễ quản lý hơn.

Mã số:

```sql
WITH sales_summary AS (
    SELECT
        product_id,
        SUM(quantity) AS total_quantity,
        SUM(price * quantity) AS total_revenue
    FROM sales
    GROUP BY product_id
)
SELECT
    p.product_name,
    s.total_quantity,
    s.total_revenue
FROM products p
JOIN sales_summary s ON p.product_id = s.product_id
ORDER BY s.total_revenue DESC
LIMIT 10;
```

Slide 2: Subqueries

Subqueries

Subqueries are nested queries within a larger SQL statement. They can be used in various parts of a query, such as SELECT, FROM, WHERE, and HAVING clauses.

Code:

```sql
SELECT
    employee_name,
    salary
FROM employees
WHERE salary > (
    SELECT AVG(salary)
    FROM employees
    WHERE department = 'Sales'
)
ORDER BY salary DESC;
```

Slide 3: Tự tham gia

Auto tham gia

Tự động kết nối được sử dụng khi một bảng cần kết nối với bảng chính, thường là để so sánh các hàng trong cùng một bảng hoặc để thiết lập mối quan hệ quan hệ phân cấp.

Mã số:

```sql
SELECT
    e.employee_name AS employee,
    m.employee_name AS manager
FROM employees e
LEFT JOIN employees m ON e.manager_id = m.employee_id
ORDER BY e.employee_name;
```

Slide 4: Window Functions

Window Functions

Window functions perform calculations across a set of table rows that are related to the current row, allowing for complex analytical queries.

Code:

```sql
SELECT
    employee_name,
    department,
    salary,
    AVG(salary) OVER (PARTITION BY department) AS dept_avg_salary,
    salary - AVG(salary) OVER (PARTITION BY department) AS salary_diff_from_avg
FROM employees
ORDER BY department, salary DESC;
```

Slide 5: Công đoàn

Công đoàn

UNION hợp nhất hai kết quả hoặc nhiều lệnh CHỌN, loại bỏ các hàng trùng lặp theo mặc định. UNION ALL giữ lại tất cả các hàng, kể cả các hàng trùng lặp vòng lặp.

Mã số:

```sql
SELECT product_name, 'In Stock' AS status
FROM products
WHERE stock_quantity > 0

UNION

SELECT product_name, 'Out of Stock' AS status
FROM products
WHERE stock_quantity = 0

ORDER BY product_name;
```

Slide 6: Thao tác ngày tháng

Thao tác ngày

SQL cung cấp nhiều chức năng khác nhau để làm việc theo ngày, cho phép tính toán và lọc dựa trên phức tạp ngày.

Mã số:

```sql
SELECT
    order_id,
    order_date,
    delivery_date,
    DATEDIFF(delivery_date, order_date) AS days_to_deliver,
    DATE_ADD(order_date, INTERVAL 7 DAY) AS expected_delivery,
    CASE
        WHEN delivery_date <= DATE_ADD(order_date, INTERVAL 7 DAY) THEN 'On Time'
        ELSE 'Delayed'
    END AS delivery_status
FROM orders
WHERE YEAR(order_date) = YEAR(CURDATE())
ORDER BY order_date;
```

Slide 7: Pivoting Techniques

Pivoting Techniques

Pivoting transforms rows into columns, useful for creating summary reports or transforming data for analysis.

Code:

```sql
SELECT
    product_category,
    SUM(CASE WHEN MONTH(order_date) = 1 THEN total_amount ELSE 0 END) AS Jan_sales,
    SUM(CASE WHEN MONTH(order_date) = 2 THEN total_amount ELSE 0 END) AS Feb_sales,
    SUM(CASE WHEN MONTH(order_date) = 3 THEN total_amount ELSE 0 END) AS Mar_sales
FROM sales
WHERE YEAR(order_date) = YEAR(CURDATE())
GROUP BY product_category
ORDER BY product_category;
```

Trang trình bày 8: Kỹ thuật không xoay vòng

Kỹ thuật không xoay vòng

Bỏ qua việc chuyển đổi các cột thành hàng, hữu ích cho việc chuẩn hóa dữ liệu hoặc chuẩn bị dữ liệu để phân tích.

Mã số:

```sql
SELECT
    product_id,
    'Jan_sales' AS month,
    Jan_sales AS sales_amount
FROM monthly_sales
UNION ALL
SELECT
    product_id,
    'Feb_sales' AS month,
    Feb_sales AS sales_amount
FROM monthly_sales
UNION ALL
SELECT
    product_id,
    'Mar_sales' AS month,
    Mar_sales AS sales_amount
FROM monthly_sales
ORDER BY product_id, month;
```

Slide 9: Data Modeling and Table Relationships

Data Modeling and Table Relationships

Data modeling involves designing the structure of a database, including tables and their relationships. Common relationship types include one-to-one, one-to-many, and many-to-many.

Code:

```sql
-- One-to-Many relationship example
CREATE TABLE departments (
    department_id INT PRIMARY KEY,
    department_name VARCHAR(50) NOT NULL
);

CREATE TABLE employees (
    employee_id INT PRIMARY KEY,
    employee_name VARCHAR(100) NOT NULL,
    department_id INT,
    FOREIGN KEY (department_id) REFERENCES departments(department_id)
);

-- Many-to-Many relationship example
CREATE TABLE students (
    student_id INT PRIMARY KEY,
    student_name VARCHAR(100) NOT NULL
);

CREATE TABLE courses (
    course_id INT PRIMARY KEY,
    course_name VARCHAR(100) NOT NULL
);

CREATE TABLE enrollments (
    student_id INT,
    course_id INT,
    enrollment_date DATE,
    PRIMARY KEY (student_id, course_id),
    FOREIGN KEY (student_id) REFERENCES students(student_id),
    FOREIGN KEY (course_id) REFERENCES courses(course_id)
);
```

Trang trình bày 10: Truyền đạt mã của bạn

Truyền đạt mã hoá của bạn

Việc truyền đạt mã SQL rõ ràng là rất quan trọng đối với hoạt động và bảo trì. Sử dụng dạng nhận dạng, định dạng quân đội và tên có ý nghĩa cho bảng, cột và bí danh.

Mã số:

```sql
-- Calculate the average order value per customer
-- for orders placed in the last 30 days
WITH recent_orders AS (
    SELECT
        customer_id,
        order_id,
        total_amount
    FROM orders
    WHERE order_date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)
)
SELECT
    c.customer_name,
    COUNT(ro.order_id) AS order_count,
    AVG(ro.total_amount) AS avg_order_value
FROM customers c
LEFT JOIN recent_orders ro ON c.customer_id = ro.customer_id
GROUP BY c.customer_id, c.customer_name
HAVING order_count > 0
ORDER BY avg_order_value DESC
LIMIT 10;
```

Slide 11: Turning Business Problems into Code

Turning Business Problems into Code

Translating business requirements into SQL involves understanding the problem, identifying relevant data, and breaking down the solution into logical steps.

Code:

```sql
-- Business Problem: Find top 5 products with the highest revenue growth
-- compared to the same month last year
WITH monthly_revenue AS (
    SELECT
        p.product_id,
        p.product_name,
        EXTRACT(YEAR_MONTH FROM s.sale_date) AS year_month,
        SUM(s.quantity * s.unit_price) AS revenue
    FROM sales s
    JOIN products p ON s.product_id = p.product_id
    WHERE s.sale_date >= DATE_SUB(CURDATE(), INTERVAL 13 MONTH)
    GROUP BY p.product_id, p.product_name, year_month
),
revenue_growth AS (
    SELECT
        cur.product_id,
        cur.product_name,
        cur.year_month,
        cur.revenue AS current_revenue,
        prev.revenue AS previous_revenue,
        (cur.revenue - prev.revenue) / prev.revenue * 100 AS growth_percentage
    FROM monthly_revenue cur
    JOIN monthly_revenue prev ON
        cur.product_id = prev.product_id AND
        cur.year_month = prev.year_month + 100
    WHERE cur.year_month = EXTRACT(YEAR_MONTH FROM CURDATE())
)
SELECT
    product_name,
    current_revenue,
    previous_revenue,
    growth_percentage
FROM revenue_growth
ORDER BY growth_percentage DESC
LIMIT 5;
```

Trang trình bày 12: Truy vấn tối ưu

Truy vấn mức độ ưu tiên

Liên kết truy vấn ưu tiên tối ưu hóa để cải thiện hiệu suất của các truy vấn SQL. Các kỹ thuật bao gồm các mục thiết lập thích hợp, tránh truy vấn phụ khi có thể và sử dụng GIẢI THÍCH để phân tích truy vấn kế hoạch.

Mã số:

```sql
-- Before optimization
SELECT
    c.customer_name,
    COUNT(o.order_id) AS order_count
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
WHERE o.order_date >= DATE_SUB(CURDATE(), INTERVAL 1 YEAR)
GROUP BY c.customer_id, c.customer_name
HAVING order_count > 10
ORDER BY order_count DESC;

-- After optimization
SELECT
    c.customer_name,
    COUNT(o.order_id) AS order_count
FROM customers c
INNER JOIN (
    SELECT customer_id, order_id
    FROM orders
    WHERE order_date >= DATE_SUB(CURDATE(), INTERVAL 1 YEAR)
) o ON c.customer_id = o.customer_id
GROUP BY c.customer_id, c.customer_name
HAVING order_count > 10
ORDER BY order_count DESC;

-- Add index to improve performance
CREATE INDEX idx_orders_customer_date ON orders (customer_id, order_date);
```

Trang hiển thị 13: QAing data

Dữ liệu QAing

Đảm bảo chất lượng (QA) trong SQL liên quan đến tính xác thực, tính năng tối ưu và độ chính xác của dữ liệu. Điều chỉnh này bao gồm việc kiểm tra các giá trị null, lặp lại bản ghi và đảm bảo dữ liệu đáp ứng các quy tắc kinh doanh.

Mã số:

```sql
-- Check for null values in important columns
SELECT
    COUNT(*) AS total_rows,
    COUNT(*) - COUNT(customer_id) AS null_customer_id,
    COUNT(*) - COUNT(order_date) AS null_order_date,
    COUNT(*) - COUNT(total_amount) AS null_total_amount
FROM orders;

-- Identify duplicate orders
SELECT
    order_id,
    customer_id,
    order_date,
    COUNT(*) AS duplicate_count
FROM orders
GROUP BY order_id, customer_id, order_date
HAVING COUNT(*) > 1;

-- Ensure all products have a valid category
SELECT
    p.product_id,
    p.product_name,
    p.category_id
FROM products p
LEFT JOIN categories c ON p.category_id = c.category_id
WHERE c.category_id IS NULL;
```

Trang trình bày 14: Tài nguyên bổ sung

Tài nguyên bổ sung

Để nâng cao hơn nữa các kỹ năng SQL của bạn, hãy xem xét khám phá các tài nguyên sau:

1. "Xử lý truy vấn hiệu quả cho công việc khoa học khối lượng dữ liệu trên nhiều CPU lõi" của Orestis Polychroniou và cộng đồng. (2019) URL ArXiv: [https://arxiv.org/abs/1906.01560](https://arxiv.org/abs/1906.01560)
2. "Tự động hóa quá trình phát triển cơ sở dữ liệu" của Isak Karlsson và cộng đồng. (2020) URL ArXiv: [https://arxiv.org/abs/2010.05761](https://arxiv.org/abs/2010.05761)
3. "Xử lý truy vấn để phân tích đồ thị" của Angela Bonifati et al. (2020) URL ArXiv: [https://arxiv.org/abs/2012.06889](https://arxiv.org/abs/2012.06889)

Các bài viết này cung cấp cái nhìn sâu sắc về nâng cao kỹ thuật SQL, cơ sở dữ liệu tối ưu hóa và các xu hướng mới nổi trong quản lý dữ liệu.
