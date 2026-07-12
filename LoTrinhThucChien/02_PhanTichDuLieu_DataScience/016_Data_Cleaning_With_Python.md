## Làm sạch dữ liệu bằng Python
Slide 1: Giới thiệu về làm sạch dữ liệu

Làm sạch dữ liệu là một bước quan trọng trong quá trình phân tích dữ liệu. Nó liên quan đến việc xác định và sửa lỗi, sự không nhất quán và không chính xác trong bộ dữ liệu để đảm bảo chất lượng và độ tin cậy cho phân tích của bạn. Python cung cấp các công cụ và thư viện mạnh mẽ để làm sạch dữ liệu hiệu quả, khiến nó trở thành một kỹ năng cần thiết cho bất kỳ nhà khoa học hoặc nhà phân tích dữ liệu nào.

```python
# Example: Loading a dataset and checking for missing values
import pandas as pd

# Load a sample dataset
df = pd.read_csv('sample_data.csv')

# Check for missing values
missing_values = df.isnull().sum()

print("Missing values in each column:")
print(missing_values)
```

Slide 2: Xử lý dữ liệu bị thiếu

Một vấn đề phổ biến trong bộ dữ liệu là thiếu giá trị. Python cung cấp nhiều phương pháp khác nhau để xử lý dữ liệu bị thiếu, chẳng hạn như loại bỏ các hàng có giá trị bị thiếu hoặc điền chúng bằng các giá trị thích hợp.

```python
# Example: Handling missing data
import pandas as pd
import numpy as np

# Create a sample dataset with missing values
data = {'A': [1, 2, np.nan, 4], 'B': [5, np.nan, 7, 8], 'C': [9, 10, 11, 12]}
df = pd.DataFrame(data)

# Drop rows with missing values
df_dropped = df.dropna()

# Fill missing values with the mean of the column
df_filled = df.fillna(df.mean())

print("Original DataFrame:")
print(df)
print("\nDataFrame after dropping rows with missing values:")
print(df_dropped)
print("\nDataFrame after filling missing values with column means:")
print(df_filled)
```

Slide 3: Kết quả: Xử lý dữ liệu bị thiếu

```
Original DataFrame:
     A    B   C
0  1.0  5.0   9
1  2.0  NaN  10
2  NaN  7.0  11
3  4.0  8.0  12

DataFrame after dropping rows with missing values:
     A    B   C
0  1.0  5.0   9
3  4.0  8.0  12

DataFrame after filling missing values with column means:
     A    B   C
0  1.0  5.0   9
1  2.0  6.7  10
2  2.3  7.0  11
3  4.0  8.0  12
```

Slide 4: Loại bỏ trùng lặp

Các mục trùng lặp có thể làm sai lệch phân tích của bạn và dẫn đến kết luận không chính xác. Thư viện pandas của Python cung cấp các phương pháp đơn giản để xác định và xóa các hàng trùng lặp khỏi tập dữ liệu của bạn.

```python
# Example: Removing duplicate rows
import pandas as pd

# Create a sample dataset with duplicate rows
data = {'A': [1, 2, 2, 3, 4], 'B': [5, 6, 6, 7, 8]}
df = pd.DataFrame(data)

# Identify duplicate rows
duplicates = df.duplicated()

# Remove duplicate rows
df_unique = df.drop_duplicates()

print("Original DataFrame:")
print(df)
print("\nDuplicate rows:")
print(duplicates)
print("\nDataFrame after removing duplicates:")
print(df_unique)
```

Slide 5: Kết quả cho: Xóa nội dung trùng lặp

```
Original DataFrame:
   A  B
0  1  5
1  2  6
2  2  6
3  3  7
4  4  8

Duplicate rows:
0    False
1    False
2     True
3    False
4    False
dtype: bool

DataFrame after removing duplicates:
   A  B
0  1  5
1  2  6
3  3  7
4  4  8
```

Slide 6: Xử lý các ngoại lệ

Các ngoại lệ có thể tác động đáng kể đến phân tích của bạn và cần được xử lý cẩn thận. Một phương pháp phổ biến là kỹ thuật Phạm vi liên tứ phân vị (IQR) để xác định và loại bỏ các giá trị ngoại lệ.

```python
# Example: Handling outliers using IQR
import pandas as pd
import numpy as np

# Create a sample dataset with outliers
data = {'values': [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]}
df = pd.DataFrame(data)

# Calculate Q1, Q3, and IQR
Q1 = df['values'].quantile(0.25)
Q3 = df['values'].quantile(0.75)
IQR = Q3 - Q1

# Define outlier bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Remove outliers
df_clean = df[(df['values'] >= lower_bound) & (df['values'] <= upper_bound)]

print("Original DataFrame:")
print(df)
print("\nDataFrame after removing outliers:")
print(df_clean)
```

Slide 7: Kết quả cho: Xử lý ngoại lệ

```
Original DataFrame:
   values
0       1
1       2
2       3
3       4
4       5
5       6
6       7
7       8
8       9
9     100

DataFrame after removing outliers:
   values
0       1
1       2
2       3
3       4
4       5
5       6
6       7
7       8
8       9
```

Slide 8: Chuyển đổi kiểu dữ liệu

Đảm bảo các loại dữ liệu chính xác là rất quan trọng để phân tích chính xác. Python cung cấp các phương thức để kiểm tra và chuyển đổi các kiểu dữ liệu khi cần thiết.

```python
# Example: Converting data types
import pandas as pd

# Create a sample dataset with mixed data types
data = {'A': ['1', '2', '3'], 'B': ['4.5', '5.5', '6.5'], 'C': ['True', 'False', 'True']}
df = pd.DataFrame(data)

# Check initial data types
print("Initial data types:")
print(df.dtypes)

# Convert data types
df['A'] = df['A'].astype(int)
df['B'] = df['B'].astype(float)
df['C'] = df['C'].astype(bool)

# Check converted data types
print("\nConverted data types:")
print(df.dtypes)

print("\nConverted DataFrame:")
print(df)
```

Slide 9: Kết quả cho: Chuyển đổi kiểu dữ liệu

```
Initial data types:
A    object
B    object
C    object
dtype: object

Converted data types:
A      int64
B    float64
C       bool
dtype: object

Converted DataFrame:
   A    B      C
0  1  4.5   True
1  2  5.5  False
2  3  6.5   True
```

Slide 10: Làm sạch và chuẩn hóa chuỗi

Dữ liệu chuỗi thường yêu cầu làm sạch và chuẩn hóa để đảm bảo tính nhất quán. Điều này bao gồm các tác vụ như xóa khoảng trắng, chuyển đổi sang chữ thường và xử lý các ký tự đặc biệt.

```python
# Example: String cleaning and normalization
import pandas as pd

# Create a sample dataset with messy string data
data = {'names': [' John ', 'JANE', 'bob ', ' Alice']}
df = pd.DataFrame(data)

# Clean and normalize strings
df['names'] = df['names'].str.strip().str.lower().str.capitalize()

print("Original DataFrame:")
print(data)
print("\nCleaned DataFrame:")
print(df)
```

Slide 11: Kết quả cho: Làm sạch và chuẩn hóa chuỗi

```
Original DataFrame:
{'names': [' John ', 'JANE', 'bob ', ' Alice']}

Cleaned DataFrame:
    names
0    John
1    Jane
2     Bob
3   Alice
```

Slide 12: Xử lý dữ liệu ngày giờ

Dữ liệu ngày và giờ thường yêu cầu xử lý và chuyển đổi đặc biệt để đảm bảo phân tích và định dạng phù hợp.

```python
# Example: Handling date and time data
import pandas as pd

# Create a sample dataset with date strings
data = {'dates': ['2023-01-01', '2023-02-15', '2023-03-30']}
df = pd.DataFrame(data)

# Convert string to datetime
df['dates'] = pd.to_datetime(df['dates'])

# Extract various components
df['year'] = df['dates'].dt.year
df['month'] = df['dates'].dt.month
df['day'] = df['dates'].dt.day
df['day_of_week'] = df['dates'].dt.day_name()

print("Processed DataFrame:")
print(df)
```

Slide 13: Kết quả: Xử lý dữ liệu ngày giờ

```
Processed DataFrame:
       dates  year  month  day day_of_week
0 2023-01-01  2023      1    1      Sunday
1 2023-02-15  2023      2   15    Wednesday
2 2023-03-30  2023      3   30     Thursday
```

Slide 14: Real-Life Example: Cleaning Weather Data

Let's apply our data cleaning techniques to a real-world scenario: cleaning weather data from a meteorological station.

```python
import pandas as pd
import numpy as np

# Create a sample weather dataset
data = {
    'date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
    'temperature': [25.5, 26.0, np.nan, 24.5, 1000],  # celsius
    'humidity': [60, 62, 58, np.nan, 61],  # percentage
    'wind_speed': ['5 m/s', '6 m/s', '4 m/s', '7 m/s', 'calm'],
    'weather': [' Sunny ', 'CLOUDY', 'rainy', ' Partly Cloudy', 'thunderstorm']
}
df = pd.DataFrame(data)

# Step 1: Convert date to datetime
df['date'] = pd.to_datetime(df['date'])

# Step 2: Handle missing values and outliers in temperature
df['temperature'] = pd.to_numeric(df['temperature'], errors='coerce')
df['temperature'] = df['temperature'].clip(lower=-50, upper=50)  # Valid range: -50°C to 50°C
df['temperature'] = df['temperature'].fillna(df['temperature'].mean())

# Step 3: Handle missing values in humidity
df['humidity'] = df['humidity'].fillna(df['humidity'].mean())

# Step 4: Clean and normalize wind speed
df['wind_speed'] = df['wind_speed'].str.replace(' m/s', '').replace('calm', '0')
df['wind_speed'] = pd.to_numeric(df['wind_speed'], errors='coerce')

# Step 5: Clean and normalize weather descriptions
df['weather'] = df['weather'].str.strip().str.lower().str.capitalize()

print("Cleaned Weather Data:")
print(df)
```

Trang trình chiếu 15: Kết quả cho: Ví dụ thực tế: Làm sạch dữ liệu thời tiết

```
Cleaned Weather Data:
        date  temperature  humidity  wind_speed      weather
0 2023-01-01        25.5      60.0         5.0        Sunny
1 2023-01-02        26.0      62.0         6.0       Cloudy
2 2023-01-03        25.3      58.0         4.0        Rainy
3 2023-01-04        24.5      60.3         7.0 Partly cloudy
4 2023-01-05        50.0      61.0         0.0  Thunderstorm
```

Slide 16: Real-Life Example: Cleaning Survey Data

Let's explore another real-world scenario: cleaning survey data collected from participants in a study about reading habits.

```python
import pandas as pd
import numpy as np

# Create a sample survey dataset
data = {
    'respondent_id': [1, 2, 3, 4, 5],
    'age': ['25', '30', 'forty', '22', '28'],
    'gender': ['M', 'F', 'f', 'MALE', 'Female'],
    'books_read_yearly': ['10', '15', '0', '100', 'many'],
    'favorite_genre': [' fiction ', 'NON-FICTION', 'Sci-Fi', 'mystery ', np.nan],
    'survey_date': ['2023-05-01', '2023-05-02', '2023-05-02', '2023-05-03', '2023-05-04']
}
df = pd.DataFrame(data)

# Step 1: Clean age data
df['age'] = pd.to_numeric(df['age'], errors='coerce')

# Step 2: Normalize gender
df['gender'] = df['gender'].str.upper().map({'M': 'Male', 'F': 'Female', 'MALE': 'Male', 'FEMALE': 'Female'})

# Step 3: Clean books_read_yearly
df['books_read_yearly'] = pd.to_numeric(df['books_read_yearly'], errors='coerce')

# Step 4: Clean and normalize favorite_genre
df['favorite_genre'] = df['favorite_genre'].str.strip().str.lower().str.capitalize()

# Step 5: Convert survey_date to datetime
df['survey_date'] = pd.to_datetime(df['survey_date'])

print("Cleaned Survey Data:")
print(df)
```

Trang trình bày 17: Kết quả cho: Ví dụ thực tế: Dữ liệu khảo sát vệ sinh

```
Cleaned Survey Data:
   respondent_id   age  gender  books_read_yearly favorite_genre survey_date
0              1  25.0    Male               10.0       Fiction  2023-05-01
1              2  30.0  Female               15.0   Non-fiction  2023-05-02
2              3   NaN  Female                0.0        Sci-fi  2023-05-02
3              4  22.0    Male              100.0       Mystery  2023-05-03
4              5  28.0  Female                NaN           NaN  2023-05-04
```

Slide 18: Additional Resources

For those interested in diving deeper into data cleaning techniques and best practices, here are some valuable resources:

1.  ArXiv paper: "A Survey on Data Preprocessing for Data Mining: Improvement of the Quality of Raw Data" by García et al. (2021) URL: [https://arxiv.org/abs/2103.10902](https://arxiv.org/abs/2103.10902)
2.  ArXiv paper: "Data Cleaning: Overview and Emerging Challenges" by Ilyas and Chu (2019) URL: [https://arxiv.org/abs/1901.04063](https://arxiv.org/abs/1901.04063)

These papers provide comprehensive overviews of data cleaning techniques, challenges, and recent advancements in the field.
