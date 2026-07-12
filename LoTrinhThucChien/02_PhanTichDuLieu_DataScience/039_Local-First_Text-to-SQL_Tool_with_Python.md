## Công cụ chuyển văn bản sang SQL cục bộ đầu tiên với Python
Trang trình bày 1: Giới thiệu về Local-First Text-to-SQL

Chuyển văn bản thành SQL cục bộ đầu tiên là một phương pháp tập trung vào việc xử lý các truy vấn ngôn ngữ tự nhiên thành các câu lệnh SQL trực tiếp trên thiết bị của người dùng. Phương pháp này tăng cường quyền riêng tư, giảm độ trễ và cho phép hoạt động ngoại tuyến. Hãy khám phá cách triển khai điều này bằng Python.

```python
import sqlite3
import nltk
from nltk.tokenize import word_tokenize

# Initialize SQLite database
conn = sqlite3.connect('local_database.db')
cursor = conn.cursor()

# Create a sample table
cursor.execute('''CREATE TABLE IF NOT EXISTS employees
                  (id INTEGER PRIMARY KEY, name TEXT, department TEXT)''')

# Function to convert natural language to SQL
def natural_language_to_sql(query):
    tokens = word_tokenize(query.lower())
    if 'show' in tokens and 'all' in tokens and 'employees' in tokens:
        return "SELECT * FROM employees"
    return "Invalid query"

# Example usage
user_query = "Show all employees"
sql_query = natural_language_to_sql(user_query)
print(f"Generated SQL: {sql_query}")

# Execute the query
cursor.execute(sql_query)
results = cursor.fetchall()
print(f"Results: {results}")
```

Slide 2: Thiết lập môi trường

Để bắt đầu với tính năng Chuyển văn bản sang SQL cục bộ đầu tiên, chúng ta cần thiết lập môi trường Python với các thư viện cần thiết. Chúng tôi sẽ sử dụng SQLite cho cơ sở dữ liệu cục bộ và NLTK để xử lý ngôn ngữ tự nhiên.

```python
# Install required libraries
!pip install nltk

# Import necessary modules
import sqlite3
import nltk
from nltk.tokenize import word_tokenize

# Download NLTK data
nltk.download('punkt')

# Initialize SQLite database
conn = sqlite3.connect('local_database.db')
cursor = conn.cursor()

print("Environment setup complete!")
```

Trang trình bày 3: Tạo cơ sở dữ liệu cục bộ

Hãy tạo một cơ sở dữ liệu SQLite cục bộ với một bảng mẫu để làm việc. Điều này sẽ đóng vai trò là nguồn dữ liệu của chúng tôi cho các truy vấn Chuyển văn bản sang SQL.

```python
# Create a sample table
cursor.execute('''CREATE TABLE IF NOT EXISTS books
                  (id INTEGER PRIMARY KEY, title TEXT, author TEXT, year INTEGER)''')

# Insert sample data
sample_data = [
    ('The Great Gatsby', 'F. Scott Fitzgerald', 1925),
    ('To Kill a Mockingbird', 'Harper Lee', 1960),
    ('1984', 'George Orwell', 1949)
]

cursor.executemany('INSERT INTO books (title, author, year) VALUES (?, ?, ?)', sample_data)
conn.commit()

print("Sample database created and populated!")
```

Trang trình bày 4: Chuyển đổi văn bản sang SQL cơ bản

Chúng ta sẽ bắt đầu với một hàm đơn giản giúp chuyển đổi các truy vấn ngôn ngữ tự nhiên cơ bản thành câu lệnh SQL. Hàm này sẽ xử lý các yêu cầu đơn giản như "Hiển thị tất cả sách".

```python
def basic_text_to_sql(query):
    tokens = word_tokenize(query.lower())

    if 'show' in tokens and 'all' in tokens and 'books' in tokens:
        return "SELECT * FROM books"
    elif 'count' in tokens and 'books' in tokens:
        return "SELECT COUNT(*) FROM books"
    else:
        return "Invalid query"

# Test the function
test_queries = [
    "Show all books",
    "Count books",
    "List authors"
]

for query in test_queries:
    sql = basic_text_to_sql(query)
    print(f"Query: {query}\nSQL: {sql}\n")
```

Trang trình bày 5: Xử lý các truy vấn phức tạp hơn

Hãy nâng cao chức năng Chuyển văn bản thành SQL của chúng tôi để xử lý các truy vấn phức tạp hơn, bao gồm các hoạt động lọc và sắp xếp.

```python
def advanced_text_to_sql(query):
    tokens = word_tokenize(query.lower())

    if 'show' in tokens and 'books' in tokens:
        sql = "SELECT * FROM books"
        if 'by' in tokens and tokens.index('by') + 1 < len(tokens):
            author = tokens[tokens.index('by') + 1]
            sql += f" WHERE author LIKE '%{author}%'"
        if 'after' in tokens and tokens.index('after') + 1 < len(tokens):
            year = tokens[tokens.index('after') + 1]
            sql += f" WHERE year > {year}"
        if 'order' in tokens and 'by' in tokens:
            if 'year' in tokens:
                sql += " ORDER BY year"
            elif 'title' in tokens:
                sql += " ORDER BY title"
        return sql
    return "Invalid query"

# Test the function
test_queries = [
    "Show books by Orwell",
    "Show books after 1950",
    "Show books order by year"
]

for query in test_queries:
    sql = advanced_text_to_sql(query)
    print(f"Query: {query}\nSQL: {sql}\n")
```

Trang trình bày 6: Triển khai Trình thực thi truy vấn

Bây giờ chúng ta có thể chuyển đổi văn bản sang SQL, hãy tạo một hàm để thực hiện các truy vấn này và trả về kết quả.

```python
def execute_query(query):
    try:
        sql = advanced_text_to_sql(query)
        cursor.execute(sql)
        results = cursor.fetchall()
        return results
    except sqlite3.Error as e:
        return f"An error occurred: {e}"

# Test the executor
test_queries = [
    "Show all books",
    "Show books by Orwell",
    "Show books after 1950 order by year"
]

for query in test_queries:
    results = execute_query(query)
    print(f"Query: {query}\nResults: {results}\n")
```

Trang trình bày 7: Xử lý sự mơ hồ và phản hồi của người dùng

Trong các tình huống thực tế, các truy vấn của người dùng có thể không rõ ràng. Hãy triển khai một hệ thống để xử lý sự mơ hồ và yêu cầu người dùng làm rõ.

```python
def handle_ambiguity(query):
    tokens = word_tokenize(query.lower())
    ambiguities = []

    if 'show' in tokens and 'books' in tokens:
        if 'author' not in tokens and 'year' not in tokens:
            ambiguities.append("Did you want to filter by author or year?")
        if 'order' not in tokens:
            ambiguities.append("Do you want to order the results?")

    if ambiguities:
        print("Your query is ambiguous. Please clarify:")
        for i, amb in enumerate(ambiguities, 1):
            print(f"{i}. {amb}")
        clarification = input("Enter the number of the clarification you'd like to address: ")
        return int(clarification)
    return 0

# Test the ambiguity handler
test_query = "Show books"
clarification_needed = handle_ambiguity(test_query)
print(f"Clarification needed: {clarification_needed}")
```

Trang trình bày 8: Triển khai hiểu ngôn ngữ tự nhiên (NLU)

Để làm cho tính năng Chuyển văn bản thành SQL của chúng tôi trở nên mạnh mẽ hơn, hãy triển khai Hiểu ngôn ngữ tự nhiên cơ bản bằng cách sử dụng tính năng gắn thẻ một phần giọng nói và nhận dạng thực thể được đặt tên của NLTK.

```python
from nltk import pos_tag, ne_chunk
from nltk.chunk import tree2conlltags

nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

def extract_entities(query):
    tokens = word_tokenize(query)
    pos_tags = pos_tag(tokens)
    ne_tree = ne_chunk(pos_tags)
    iob_tags = tree2conlltags(ne_tree)

    entities = {
        'PERSON': [],
        'DATE': [],
        'ORGANIZATION': []
    }

    for word, pos, ne in iob_tags:
        if ne != 'O':
            entity_type = ne.split('-')[1]
            if entity_type in entities:
                entities[entity_type].append(word)

    return entities

# Test the entity extractor
test_query = "Show books by George Orwell published after 1945"
entities = extract_entities(test_query)
print(f"Extracted entities: {entities}")
```

Trang trình bày 9: Tích hợp NLU với Text-to-SQL

Bây giờ, hãy tích hợp các khả năng NLU của chúng tôi vào quá trình chuyển đổi Văn bản sang SQL để xử lý các truy vấn ngôn ngữ tự nhiên hơn.

```python
def nlu_text_to_sql(query):
    entities = extract_entities(query)
    tokens = word_tokenize(query.lower())

    sql = "SELECT * FROM books"
    conditions = []

    if entities['PERSON']:
        author = ' '.join(entities['PERSON'])
        conditions.append(f"author LIKE '%{author}%'")

    if entities['DATE']:
        year = entities['DATE'][0]
        if 'after' in tokens or 'since' in tokens:
            conditions.append(f"year > {year}")
        elif 'before' in tokens:
            conditions.append(f"year < {year}")
        else:
            conditions.append(f"year = {year}")

    if conditions:
        sql += " WHERE " + " AND ".join(conditions)

    if 'order' in tokens and 'by' in tokens:
        if 'year' in tokens:
            sql += " ORDER BY year"
        elif 'title' in tokens:
            sql += " ORDER BY title"

    return sql

# Test the NLU-enhanced Text-to-SQL
test_queries = [
    "Show books by George Orwell",
    "Find books published after 1950",
    "List books by Harper Lee ordered by year"
]

for query in test_queries:
    sql = nlu_text_to_sql(query)
    print(f"Query: {query}\nSQL: {sql}\n")
```

Slide 10: Xử lý lỗi và các trường hợp khó khăn

Để làm cho công cụ Chuyển văn bản thành SQL đầu tiên cục bộ của chúng tôi trở nên mạnh mẽ hơn, hãy triển khai xử lý lỗi và quản lý các trường hợp khó khăn.

```python
def safe_text_to_sql(query):
    try:
        sql = nlu_text_to_sql(query)
        # Validate SQL to prevent injection
        if any(keyword in sql.lower() for keyword in ['drop', 'delete', 'update', 'insert']):
            raise ValueError("Potentially harmful SQL detected")
        return sql
    except Exception as e:
        return f"Error: {str(e)}"

def safe_execute_query(query):
    try:
        sql = safe_text_to_sql(query)
        if sql.startswith("Error:"):
            return sql
        cursor.execute(sql)
        results = cursor.fetchall()
        return results
    except sqlite3.Error as e:
        return f"Database error: {str(e)}"

# Test error handling
test_queries = [
    "Show all books",
    "Delete all books",  # Potentially harmful
    "Show books published in abcdef"  # Invalid year
]

for query in test_queries:
    result = safe_execute_query(query)
    print(f"Query: {query}\nResult: {result}\n")
```

Trang trình bày 11: Triển khai giao diện người dùng đơn giản

Hãy tạo một giao diện dòng lệnh đơn giản cho công cụ Chuyển văn bản thành SQL đầu tiên cục bộ của chúng ta.

```python
def text_to_sql_interface():
    print("Welcome to the Local-First Text-to-SQL Tool")
    print("Enter your queries in natural language, or type 'exit' to quit.")

    while True:
        query = input("\nEnter your query: ")
        if query.lower() == 'exit':
            break

        result = safe_execute_query(query)
        if isinstance(result, str):
            print(result)
        else:
            print("Results:")
            for row in result:
                print(row)

# Run the interface
text_to_sql_interface()
```

Trang trình bày 12: Tối ưu hóa hiệu suất

Để đảm bảo công cụ ưu tiên cục bộ của chúng tôi vẫn phản hồi nhanh, hãy triển khai một số tối ưu hóa hiệu suất cơ bản.

```python
import time
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_text_to_sql(query):
    return safe_text_to_sql(query)

def measure_performance(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Query executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper

@measure_performance
def optimized_execute_query(query):
    sql = cached_text_to_sql(query)
    if sql.startswith("Error:"):
        return sql
    cursor.execute(sql)
    return cursor.fetchall()

# Test performance
test_queries = [
    "Show all books",
    "Show all books",  # Should be faster due to caching
    "Show books by Orwell"
]

for query in test_queries:
    result = optimized_execute_query(query)
    print(f"Query: {query}\nResult: {result}\n")
```

Slide 13: Ví dụ thực tế: Hệ thống quản lý thư viện

Hãy áp dụng công cụ Chuyển văn bản thành SQL đầu tiên cục bộ của chúng tôi vào kịch bản hệ thống quản lý thư viện.

```python
# Create a more complex database schema
cursor.execute('''CREATE TABLE IF NOT EXISTS library_books
                  (id INTEGER PRIMARY KEY, title TEXT, author TEXT,
                   isbn TEXT, publication_year INTEGER, available BOOLEAN)''')

# Insert sample data
sample_books = [
    ('The Catcher in the Rye', 'J.D. Salinger', '9780316769174', 1951, True),
    ('Pride and Prejudice', 'Jane Austen', '9780141439518', 1813, False),
    ('The Hobbit', 'J.R.R. Tolkien', '9780547928227', 1937, True)
]

cursor.executemany('''INSERT INTO library_books
                      (title, author, isbn, publication_year, available)
                      VALUES (?, ?, ?, ?, ?)''', sample_books)
conn.commit()

def library_query_to_sql(query):
    tokens = word_tokenize(query.lower())
    sql = "SELECT * FROM library_books"
    conditions = []

    if 'available' in tokens:
        conditions.append("available = 1")
    if 'author' in tokens and tokens.index('author') + 1 < len(tokens):
        author = tokens[tokens.index('author') + 1]
        conditions.append(f"author LIKE '%{author}%'")
    if 'year' in tokens and tokens.index('year') + 1 < len(tokens):
        year = tokens[tokens.index('year') + 1]
        conditions.append(f"publication_year = {year}")

    if conditions:
        sql += " WHERE " + " AND ".join(conditions)

    return sql

# Test library queries
library_queries = [
    "Show all available books",
    "Find books by author Tolkien",
    "List books published in year 1813"
]

for query in library_queries:
    sql = library_query_to_sql(query)
    print(f"Query: {query}\nSQL: {sql}")
    cursor.execute(sql)
    results = cursor.fetchall()
    print(f"Results: {results}\n")
```

Trang trình chiếu 14: Ví dụ thực tế: Cơ sở dữ liệu công thức nấu ăn

Hãy cùng khám phá một ví dụ thực tế khác bằng cách sử dụng cơ sở dữ liệu công thức nấu ăn để chứng minh tính linh hoạt của công cụ Chuyển văn bản sang SQL đầu tiên cục bộ của chúng tôi.

```python
# Create a recipe database
cursor.execute('''CREATE TABLE IF NOT EXISTS recipes
                  (id INTEGER PRIMARY KEY, name TEXT, cuisine TEXT,
                   prep_time INTEGER, difficulty TEXT)''')

# Insert sample data
sample_recipes = [
    ('Spaghetti Carbonara', 'Italian', 30, 'Easy'),
    ('Sushi Rolls', 'Japanese', 60, 'Intermediate'),
    ('Chicken Tikka Masala', 'Indian', 45, 'Intermediate'),
    ('Caesar Salad', 'American', 15, 'Easy')
]

cursor.executemany('''INSERT INTO recipes
                      (name, cuisine, prep_time, difficulty)
                      VALUES (?, ?, ?, ?)''', sample_recipes)
conn.commit()

def recipe_query_to_sql(query):
    tokens = word_tokenize(query.lower())
    sql = "SELECT * FROM recipes"
    conditions = []

    if 'cuisine' in tokens and tokens.index('cuisine') + 1 < len(tokens):
        cuisine = tokens[tokens.index('cuisine') + 1]
        conditions.append(f"cuisine LIKE '%{cuisine}%'")
    if 'easy' in tokens:
        conditions.append("difficulty = 'Easy'")
    if 'quick' in tokens or 'fast' in tokens:
        conditions.append("prep_time <= 30")

    if conditions:
        sql += " WHERE " + " AND ".join(conditions)

    return sql

# Test recipe queries
recipe_queries = [
    "Show all Italian cuisine recipes",
    "Find easy recipes",
    "List quick Japanese dishes"
]

for query in recipe_queries:
    sql = recipe_query_to_sql(query)
    print(f"Query: {query}\nSQL: {sql}")
    cursor.execute(sql)
    results = cursor.fetchall()
    print(f"Results: {results}\n")
```

Trang trình bày 15: Nâng cao trải nghiệm người dùng với kết hợp mờ

Để cải thiện trải nghiệm người dùng, hãy triển khai kết hợp mờ cho tên công thức và món ăn.

```python
from fuzzywuzzy import process

def fuzzy_match(query, choices, threshold=80):
    return process.extractOne(query, choices, score_cutoff=threshold)

def enhanced_recipe_query_to_sql(query):
    tokens = word_tokenize(query.lower())
    sql = "SELECT * FROM recipes"
    conditions = []

    cuisines = ['Italian', 'Japanese', 'Indian', 'American']
    if 'cuisine' in tokens:
        cuisine_query = ' '.join(tokens[tokens.index('cuisine')+1:])
        matched_cuisine = fuzzy_match(cuisine_query, cuisines)
        if matched_cuisine:
            conditions.append(f"cuisine = '{matched_cuisine[0]}'")

    if 'recipe' in tokens:
        recipe_query = ' '.join(tokens[tokens.index('recipe')+1:])
        cursor.execute("SELECT name FROM recipes")
        recipe_names = [row[0] for row in cursor.fetchall()]
        matched_recipe = fuzzy_match(recipe_query, recipe_names)
        if matched_recipe:
            conditions.append(f"name = '{matched_recipe[0]}'")

    if conditions:
        sql += " WHERE " + " AND ".join(conditions)

    return sql

# Test enhanced recipe queries
enhanced_queries = [
    "Find Italian cuisine recipes",
    "Show me the spageti carbonara recipe",
    "List dishes from Indian cuisine"
]

for query in enhanced_queries:
    sql = enhanced_recipe_query_to_sql(query)
    print(f"Query: {query}\nSQL: {sql}")
    cursor.execute(sql)
    results = cursor.fetchall()
    print(f"Results: {results}\n")
```

Trang trình bày 16: Tài nguyên bổ sung

Đối với những người quan tâm đến việc tìm hiểu sâu hơn về triển khai Chuyển văn bản thành SQL đầu tiên cục bộ và xử lý ngôn ngữ tự nhiên, đây là một số tài nguyên có giá trị:

1. "Tạo văn bản sang SQL thần kinh cho các câu hỏi phụ thuộc vào bối cảnh tên miền chéo" của Zhichu Lu et al. (2022) ArXiv: [https://arxiv.org/abs/2201.10094](https://arxiv.org/abs/2201.10094)
2. "Cải tiến phương pháp đánh giá văn bản sang SQL" của Catherine Finegan-Dollak và cộng sự. (2018) ArXiv: [https://arxiv.org/abs/1806.09029](https://arxiv.org/abs/1806.09029)
3. "Kết nối dữ liệu văn bản và dạng bảng để phân tích cú pháp ngữ nghĩa từ văn bản sang SQL trên nhiều miền" của Xi Victoria Lin và cộng sự. (2020) ArXiv: [https://arxiv.org/abs/2012.12627](https://arxiv.org/abs/2012.12627)

Các bài viết này cung cấp thông tin chi tiết về các kỹ thuật nâng cao để tạo và đánh giá Chuyển văn bản thành SQL, có thể được điều chỉnh cho việc triển khai cục bộ đầu tiên.
