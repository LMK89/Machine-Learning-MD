## Quái vật tranh chấp dữ liệu trong thế giới thực

Slide 1: Thực tế của dữ liệu trong thế giới thực

Việc sắp xếp dữ liệu trong các tình huống thực tế thực sự khó khăn hơn so với làm việc với các bộ dữ liệu sạch, được xử lý trước như iris hoặc mtcars. Dữ liệu trong thế giới thực thường có sự không nhất quán, thiếu giá trị và các định dạng không mong muốn đòi hỏi nỗ lực đáng kể để làm sạch và chuẩn bị cho phân tích. Tuy nhiên, việc mô tả quá trình này là "thuần hóa một con thú" có thể là một sự cường điệu hóa. Mặc dù việc sắp xếp dữ liệu có thể phức tạp nhưng đây là một phần thiết yếu và dễ quản lý trong quy trình khoa học dữ liệu có thể được tiếp cận một cách có hệ thống.

```python
# Example of real-world data inconsistencies
raw_data = [
    {'name': 'John Doe', 'age': '35', 'income': '$50,000'},
    {'name': 'Jane Smith', 'age': 'N/A', 'income': '45000'},
    {'name': 'Bob Johnson', 'age': '42', 'income': None},
    {'name': 'Alice Brown', 'age': '28', 'income': '55,000'}
]

# Demonstrating inconsistencies
for entry in raw_data:
    print(f"Name: {entry['name']}")
    print(f"Age: {entry['age']} (type: {type(entry['age']).__name__})")
    print(f"Income: {entry['income']} (type: {type(entry['income']).__name__})")
    print()
```

Slide 2: Xử lý các giá trị bị thiếu

Các giá trị bị thiếu là hiện tượng phổ biến trong các bộ dữ liệu trong thế giới thực và có thể tác động đáng kể đến việc phân tích nếu không được xử lý đúng cách. Có một số chiến lược để xử lý dữ liệu bị thiếu, bao gồm áp đặt (điền các giá trị bị thiếu), nội suy hoặc xóa bản ghi có giá trị bị thiếu. Sự lựa chọn phụ thuộc vào bản chất của dữ liệu và các yêu cầu cụ thể của việc phân tích.

```python
def handle_missing_values(data, strategy='mean'):
    clean_data = []
    if strategy == 'mean':
        # Calculate mean age, excluding 'N/A'
        valid_ages = [int(entry['age']) for entry in data if entry['age'] != 'N/A' and entry['age'] is not None]
        mean_age = sum(valid_ages) / len(valid_ages)

        for entry in data:
            new_entry = entry.copy()
            if entry['age'] == 'N/A' or entry['age'] is None:
                new_entry['age'] = mean_age
            clean_data.append(new_entry)
    return clean_data

cleaned_data = handle_missing_values(raw_data)
for entry in cleaned_data:
    print(f"Name: {entry['name']}, Age: {entry['age']}")
```

Trang trình bày 3: Phát hiện và loại bỏ ngoại lệ

Các ngoại lệ có thể làm sai lệch đáng kể các phân tích thống kê và mô hình học máy. Việc xác định và xử lý thích hợp các giá trị ngoại lệ là rất quan trọng để duy trì tính toàn vẹn của dữ liệu và đảm bảo kết quả chính xác. Các phương pháp phổ biến để phát hiện ngoại lệ bao gồm các kỹ thuật thống kê như điểm Z và Phạm vi liên tứ phân vị (IQR), cũng như các phương pháp trực quan như biểu đồ hình hộp.

```python
import statistics

def detect_outliers(data, feature, threshold=2):
    values = [float(entry[feature]) for entry in data if entry[feature] is not None]
    mean = statistics.mean(values)
    std_dev = statistics.stdev(values)

    outliers = []
    for entry in data:
        if entry[feature] is not None:
            z_score = (float(entry[feature]) - mean) / std_dev
            if abs(z_score) > threshold:
                outliers.append(entry)

    return outliers

# Assuming we've cleaned the 'age' data to be numeric
cleaned_data = [{'name': 'John', 'age': 35}, {'name': 'Jane', 'age': 28},
                {'name': 'Bob', 'age': 42}, {'name': 'Alice', 'age': 90}]

outliers = detect_outliers(cleaned_data, 'age', threshold=2)
print("Detected outliers:")
for outlier in outliers:
    print(f"Name: {outlier['name']}, Age: {outlier['age']}")
```

Slide 4: Chuyển đổi dữ liệu

Chuyển đổi dữ liệu thường là cần thiết để chuẩn bị dữ liệu cho việc phân tích hoặc lập mô hình. Điều này có thể liên quan đến việc chuẩn hóa hoặc chuẩn hóa các đặc tính số, mã hóa các biến phân loại hoặc áp dụng các phép biến đổi toán học để đạt được phân bố mong muốn. Việc chuyển đổi dữ liệu thích hợp đảm bảo rằng tất cả các tính năng đều đóng góp phù hợp cho quá trình phân tích.

```python
def standardize_feature(data, feature):
    values = [entry[feature] for entry in data]
    mean = sum(values) / len(values)
    std_dev = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5

    for entry in data:
        entry[f'{feature}_standardized'] = (entry[feature] - mean) / std_dev

    return data

# Example usage
numeric_data = [{'value': 10}, {'value': 20}, {'value': 30}, {'value': 40}, {'value': 50}]
standardized_data = standardize_feature(numeric_data, 'value')

for entry in standardized_data:
    print(f"Original: {entry['value']}, Standardized: {entry['value_standardized']:.2f}")
```

Trang trình bày 5: Kỹ thuật tính năng

Kỹ thuật tính năng là quá trình tạo ra các tính năng mới từ dữ liệu hiện có để cải thiện hiệu suất của mô hình. Điều này có thể liên quan đến việc kết hợp các tính năng hiện có, trích xuất thông tin từ các loại dữ liệu phức tạp hoặc áp dụng kiến ​​thức về miền để tạo ra nhiều biến thông tin hơn. Kỹ thuật tính năng hiệu quả thường đòi hỏi sự hiểu biết sâu sắc về miền vấn đề và tư duy sáng tạo.

```python
def engineer_features(data):
    for entry in data:
        # Create a new feature: age group
        if entry['age'] < 18:
            entry['age_group'] = 'minor'
        elif 18 <= entry['age'] < 65:
            entry['age_group'] = 'adult'
        else:
            entry['age_group'] = 'senior'

        # Create a feature for name length
        entry['name_length'] = len(entry['name'])

    return data

# Example usage
sample_data = [
    {'name': 'John Doe', 'age': 35},
    {'name': 'Jane Smith', 'age': 17},
    {'name': 'Bob Johnson', 'age': 70}
]

engineered_data = engineer_features(sample_data)
for entry in engineered_data:
    print(f"Name: {entry['name']}, Age: {entry['age']}, "
          f"Age Group: {entry['age_group']}, Name Length: {entry['name_length']}")
```

Slide 6: Mã hóa dữ liệu phân loại

Nhiều thuật toán học máy yêu cầu đầu vào bằng số, đòi hỏi phải chuyển đổi dữ liệu phân loại thành định dạng số. Các kỹ thuật mã hóa phổ biến bao gồm mã hóa một lần cho các danh mục danh nghĩa và mã hóa nhãn cho các danh mục thứ tự. Việc lựa chọn phương pháp mã hóa có thể tác động đáng kể đến hiệu suất và khả năng diễn giải của mô hình.

```python
def one_hot_encode(data, feature):
    # Get unique categories
    categories = set(entry[feature] for entry in data)

    for entry in data:
        for category in categories:
            entry[f'{feature}_{category}'] = 1 if entry[feature] == category else 0

    return data

# Example usage
categorical_data = [
    {'color': 'red'},
    {'color': 'blue'},
    {'color': 'green'},
    {'color': 'red'}
]

encoded_data = one_hot_encode(categorical_data, 'color')
for entry in encoded_data:
    print(entry)
```

Trang trình bày 7: Ví dụ thực tế: Phân tích dữ liệu thời tiết

Hãy xem xét một ví dụ thực tế về dữ liệu thời tiết khó hiểu. Bộ dữ liệu thời tiết thường gặp nhiều thách thức khác nhau, bao gồm thiếu giá trị, đơn vị đo lường khác nhau và nhu cầu về kỹ thuật tính năng để rút ra những hiểu biết có ý nghĩa.

```python
raw_weather_data = [
    {'date': '2023-01-01', 'temperature': '72F', 'humidity': '65%', 'precipitation': '0.1in'},
    {'date': '2023-01-02', 'temperature': '68F', 'humidity': 'N/A', 'precipitation': '0'},
    {'date': '2023-01-03', 'temperature': '18C', 'humidity': '70%', 'precipitation': '5mm'},
    {'date': '2023-01-04', 'temperature': '65F', 'humidity': '60%', 'precipitation': None}
]

def clean_weather_data(data):
    cleaned_data = []
    for entry in data:
        new_entry = {}
        # Convert date to datetime object
        new_entry['date'] = entry['date']  # In practice, use datetime.strptime()

        # Handle temperature: convert all to Celsius
        if 'F' in entry['temperature']:
            temp = float(entry['temperature'].rstrip('F'))
            new_entry['temperature_celsius'] = (temp - 32) * 5/9
        else:
            new_entry['temperature_celsius'] = float(entry['temperature'].rstrip('C'))

        # Handle humidity: convert to float and fill missing values
        new_entry['humidity'] = float(entry['humidity'].rstrip('%')) if entry['humidity'] != 'N/A' else None

        # Handle precipitation: convert all to mm and handle missing values
        if entry['precipitation'] is None or entry['precipitation'] == '0':
            new_entry['precipitation_mm'] = 0
        elif 'in' in entry['precipitation']:
            precip = float(entry['precipitation'].rstrip('in'))
            new_entry['precipitation_mm'] = precip * 25.4
        else:
            new_entry['precipitation_mm'] = float(entry['precipitation'].rstrip('mm'))

        cleaned_data.append(new_entry)

    return cleaned_data

cleaned_weather = clean_weather_data(raw_weather_data)
for entry in cleaned_weather:
    print(entry)
```

Trang trình bày 8: Kết quả cho: Ví dụ thực tế: Phân tích dữ liệu thời tiết

```
{'date': '2023-01-01', 'temperature_celsius': 22.22222222222222, 'humidity': 65.0, 'precipitation_mm': 2.54}
{'date': '2023-01-02', 'temperature_celsius': 20.0, 'humidity': None, 'precipitation_mm': 0}
{'date': '2023-01-03', 'temperature_celsius': 18.0, 'humidity': 70.0, 'precipitation_mm': 5.0}
{'date': '2023-01-04', 'temperature_celsius': 18.333333333333332, 'humidity': 60.0, 'precipitation_mm': 0}
```

Slide 9: Real-Life Example: Text Data Processing

Text data is another common type of real-world data that often requires extensive wrangling. This can include tasks such as tokenization, removing stop words, stemming or lemmatization, and handling special characters or formatting issues.

```python
import re
from collections import Counter

def process_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenize
    tokens = text.split()

    # Remove stop words (a very basic list for demonstration)
    stop_words = set(['the', 'a', 'an', 'in', 'on', 'at', 'for', 'to', 'of'])
    tokens = [token for token in tokens if token not in stop_words]

    # Count word frequencies
    word_freq = Counter(tokens)

    return word_freq

# Example usage
sample_text = """
The quick brown fox jumps over the lazy dog.
The dog barks, but the fox is too quick!
In the end, both animals are tired.
"""

processed_text = process_text(sample_text)
print("Word frequencies:")
for word, freq in processed_text.most_common(5):
    print(f"{word}: {freq}")
```

Trang trình bày 10: Kết quả cho: Ví dụ thực tế: Xử lý dữ liệu văn bản

```
Word frequencies:
quick: 2
fox: 2
dog: 2
jumps: 1
over: 1
```

Slide 11: Challenges and Best Practices

While data wrangling can be complex, it's a crucial step in the data science process. Some best practices include:

1.  Understanding your data sources and potential issues before starting.
2.  Documenting all data cleaning and transformation steps for reproducibility.
3.  Regularly validating your data throughout the wrangling process.
4.  Using version control for your data and code.
5.  Collaborating with domain experts to ensure appropriate handling of field-specific data.

```python
def data_quality_check(data, expected_columns):
    issues = []

    # Check for missing columns
    missing_columns = set(expected_columns) - set(data[0].keys())
    if missing_columns:
        issues.append(f"Missing columns: {', '.join(missing_columns)}")

    # Check for missing values
    for entry in data:
        for column in expected_columns:
            if column in entry and entry[column] is None:
                issues.append(f"Missing value in column '{column}' for entry: {entry}")

    # Check for data type consistency
    for column in expected_columns:
        column_types = set(type(entry[column]) for entry in data if column in entry)
        if len(column_types) > 1:
            issues.append(f"Inconsistent data types in column '{column}': {column_types}")

    return issues

# Example usage
sample_data = [
    {'name': 'John', 'age': 30, 'city': 'New York'},
    {'name': 'Jane', 'age': '25', 'city': None},
    {'name': 'Bob', 'city': 'Chicago'}
]

expected_columns = ['name', 'age', 'city']
quality_issues = data_quality_check(sample_data, expected_columns)

print("Data quality issues:")
for issue in quality_issues:
    print(f"- {issue}")
```

Slide 12: Tự động sắp xếp dữ liệu

Khi các tập dữ liệu ngày càng lớn hơn và phức tạp hơn, việc tự động hóa các phần của quy trình sắp xếp dữ liệu ngày càng trở nên quan trọng. Mặc dù thường không thể tự động hóa hoàn toàn do đặc điểm riêng của từng tập dữ liệu nhưng một số tác vụ nhất định có thể được chuẩn hóa và tự động hóa để nâng cao hiệu quả.

```python
class DataWrangler:
    def __init__(self, data):
        self.data = data

    def remove_duplicates(self):
        self.data = list({tuple(d.items()) for d in self.data})
        return self

    def fill_missing_values(self, column, strategy='mean'):
        if strategy == 'mean':
            values = [entry[column] for entry in self.data if entry[column] is not None]
            mean_value = sum(values) / len(values)
            for entry in self.data:
                if entry[column] is None:
                    entry[column] = mean_value
        return self

    def standardize_column(self, column):
        values = [entry[column] for entry in self.data]
        mean = sum(values) / len(values)
        std_dev = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5

        for entry in self.data:
            entry[f'{column}_standardized'] = (entry[column] - mean) / std_dev
        return self

    def get_cleaned_data(self):
        return self.data

# Example usage
raw_data = [
    {'id': 1, 'value': 10},
    {'id': 2, 'value': None},
    {'id': 3, 'value': 20},
    {'id': 1, 'value': 10},  # Duplicate
    {'id': 4, 'value': 30}
]

wrangler = DataWrangler(raw_data)
cleaned_data = (wrangler
                .remove_duplicates()
                .fill_missing_values('value')
                .standardize_column('value')
                .get_cleaned_data())

for entry in cleaned_data:
    print(entry)
```

Trang trình bày 13: Cải tiến liên tục trong việc sắp xếp dữ liệu

Sắp xếp dữ liệu là một quá trình lặp đi lặp lại đòi hỏi phải sàng lọc và thích ứng liên tục. Khi bạn làm việc với các bộ dữ liệu đa dạng và gặp phải những thách thức mới, điều quan trọng là phải cập nhật các kỹ thuật và công cụ xử lý tranh chấp của bạn. Sự cải tiến liên tục này bao gồm việc học hỏi từ những kinh nghiệm trong quá khứ, luôn cập nhật các phương pháp mới và cải tiến cách tiếp cận của bạn dựa trên nhu cầu cụ thể của từng dự án.

```python
class AdaptiveDataWrangler:
    def __init__(self):
        self.techniques = {}
        self.performance_log = {}

    def add_technique(self, name, function):
        self.techniques[name] = function
        self.performance_log[name] = {'uses': 0, 'success_rate': 0}

    def apply_technique(self, name, data):
        if name not in self.techniques:
            raise ValueError(f"Technique '{name}' not found")

        result = self.techniques[name](data)
        self.performance_log[name]['uses'] += 1

        # In a real scenario, you'd implement a way to measure success
        success = self.evaluate_success(result)

        current_success_rate = self.performance_log[name]['success_rate']
        total_uses = self.performance_log[name]['uses']
        self.performance_log[name]['success_rate'] = (
            (current_success_rate * (total_uses - 1) + success) / total_uses
        )

        return result

    def evaluate_success(self, result):
        # Placeholder for success evaluation logic
        return 1  # Assume success for this example

    def get_best_technique(self):
        return max(self.performance_log, key=lambda x: self.performance_log[x]['success_rate'])

# Example usage
wrangler = AdaptiveDataWrangler()
wrangler.add_technique('remove_nulls', lambda data: [d for d in data if all(d.values())])
wrangler.add_technique('fill_mean', lambda data: [{**d, 'value': sum(d['value'] for d in data if d['value']) / len(data) if d['value'] is None else d['value']} for d in data])

sample_data = [{'id': 1, 'value': 10}, {'id': 2, 'value': None}, {'id': 3, 'value': 20}]

print("Applying techniques:")
print(wrangler.apply_technique('remove_nulls', sample_data))
print(wrangler.apply_technique('fill_mean', sample_data))

print("\nBest technique:", wrangler.get_best_technique())
```

Trang trình bày 14: Những cân nhắc về mặt đạo đức trong việc sắp xếp dữ liệu

Khi làm việc với dữ liệu trong thế giới thực, điều quan trọng là phải xem xét các tác động về mặt đạo đức. Điều này bao gồm việc đảm bảo quyền riêng tư của dữ liệu, tránh sai lệch trong việc làm sạch và chuyển đổi dữ liệu cũng như minh bạch về các phương pháp được sử dụng. Các biện pháp xử lý dữ liệu có đạo đức giúp duy trì tính toàn vẹn của phân tích và bảo vệ các cá nhân có trong tập dữ liệu của bạn.

```python
def anonymize_data(data, sensitive_fields):
    anonymized_data = []
    for entry in data:
        anonymized_entry = {}
        for key, value in entry.items():
            if key in sensitive_fields:
                anonymized_entry[key] = hash(str(value))  # Simple hashing for demonstration
            else:
                anonymized_entry[key] = value
        anonymized_data.append(anonymized_entry)
    return anonymized_data

def check_data_bias(data, protected_attribute, target_attribute):
    groups = {}
    for entry in data:
        group = entry[protected_attribute]
        if group not in groups:
            groups[group] = {'count': 0, 'sum': 0}
        groups[group]['count'] += 1
        groups[group]['sum'] += entry[target_attribute]

    for group, stats in groups.items():
        stats['average'] = stats['sum'] / stats['count']

    return groups

# Example usage
sample_data = [
    {'id': 1, 'name': 'Alice', 'age': 30, 'salary': 50000, 'gender': 'F'},
    {'id': 2, 'name': 'Bob', 'age': 35, 'salary': 60000, 'gender': 'M'},
    {'id': 3, 'name': 'Charlie', 'age': 40, 'salary': 70000, 'gender': 'M'},
    {'id': 4, 'name': 'Diana', 'age': 38, 'salary': 65000, 'gender': 'F'}
]

anonymized_data = anonymize_data(sample_data, ['name', 'id'])
print("Anonymized data:")
for entry in anonymized_data:
    print(entry)

bias_check = check_data_bias(sample_data, 'gender', 'salary')
print("\nPotential bias check:")
for group, stats in bias_check.items():
    print(f"{group}: Average salary = {stats['average']}")
```

Trang trình bày 15: Tài nguyên bổ sung

Đối với những người muốn hiểu sâu hơn về các kỹ thuật sắp xếp dữ liệu và các phương pháp hay nhất, dưới đây là một số tài nguyên có giá trị:

1. Bài viết ArXiv: "Khảo sát về thu thập dữ liệu cho Machine Learning: Dữ liệu lớn - Quan điểm tích hợp AI" của Yuji Roh, Geon Heo, Steven Euijong Whang (2019). ArXiv:1811.03402 \[cs.LG\]
2. Bài viết ArXiv: "Tự động hóa xác minh chất lượng dữ liệu quy mô lớn" của Sebastian Schelter, Dustin Lange, Philipp Schmidt, Meltem Celikel, Felix Biessmann, Andreas Grafberger (2018). ArXiv:1801.07900 \[cs.DB\]
3. Bài báo ArXiv: "Hướng tới việc làm sạch dữ liệu tự động: Phương pháp thống kê" của Sanjay Krishnan, Jiannan Wang, Eugene Wu, Michael J. Franklin, Ken Goldberg (2016). ArXiv:1603.08248 \[cs.DB\]

Các bài viết này cung cấp các cuộc thảo luận chuyên sâu về các khía cạnh khác nhau của việc sắp xếp dữ liệu, từ thu thập đến xác minh và làm sạch, đồng thời có thể đóng vai trò là điểm khởi đầu tuyệt vời để khám phá thêm chủ đề.
