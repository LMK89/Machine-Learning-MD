## Làm chủ việc chuyển đổi kiểu dữ liệu trong Pandas

Trang trình bày 1: Tìm hiểu các kiểu dữ liệu trong Pandas

Các kiểu dữ liệu (dtype) trong Pandas xác định cách dữ liệu được lưu trữ và xử lý trong DataFrames và Series. Chúng đóng một vai trò quan trọng trong việc sử dụng bộ nhớ và hiệu suất. Pandas hỗ trợ nhiều loại dtype khác nhau, bao gồm các loại số (int64, float64), boolean, object, datetime và phân loại. Hãy cùng khám phá những loại này bằng một ví dụ thực tế.

```python
import pandas as pd

# Create a sample DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'Height': [1.65, 1.80, 1.75],
    'Is_Student': [True, False, True],
    'Birthdate': ['1998-03-15', '1993-07-22', '1988-11-30']
}

df = pd.DataFrame(data)

# Display DataFrame and dtypes
print(df)
print("\nData Types:")
print(df.dtypes)
```

Trang trình bày 2: Kết quả cho: Tìm hiểu các kiểu dữ liệu trong Pandas

```
     Name  Age  Height  Is_Student   Birthdate
0   Alice   25    1.65        True  1998-03-15
1     Bob   30    1.80       False  1993-07-22
2  Charlie  35    1.75        True  1988-11-30

Data Types:
Name          object
Age            int64
Height       float64
Is_Student      bool
Birthdate     object
dtype: object
```

Trang trình bày 3: Các kiểu số trong Pandas

Pandas hỗ trợ nhiều loại số khác nhau, bao gồm số nguyên và số dấu phẩy động. Phổ biến nhất là int64 và float64. Hãy cùng khám phá cách làm việc với những loại này và tác động của chúng đến việc sử dụng bộ nhớ.

```python
import pandas as pd
import numpy as np

# Create a DataFrame with different numeric types
df = pd.DataFrame({
    'int32': np.array([1, 2, 3], dtype=np.int32),
    'int64': np.array([1, 2, 3], dtype=np.int64),
    'float32': np.array([1.0, 2.0, 3.0], dtype=np.float32),
    'float64': np.array([1.0, 2.0, 3.0], dtype=np.float64)
})

# Display DataFrame and memory usage
print(df)
print("\nData Types:")
print(df.dtypes)
print("\nMemory Usage:")
print(df.memory_usage(deep=True))
```

Trang trình bày 4: Kết quả cho: Các kiểu số trong Pandas

```
   int32  int64  float32  float64
0      1      1      1.0      1.0
1      2      2      2.0      2.0
2      3      3      3.0      3.0

Data Types:
int32      int32
int64      int64
float32  float32
float64  float64
dtype: object

Memory Usage:
Index       128
int32        12
int64        24
float32      12
float64      24
dtype: int64
```

Trang trình bày 5: Boolean và các loại đối tượng

Các kiểu Boolean và object rất cần thiết để xử lý các giá trị logic và các kiểu dữ liệu hỗn hợp. Hãy xem xét cách các loại này hoạt động trong Pandas và ý nghĩa bộ nhớ của chúng.

```python
import pandas as pd

# Create a DataFrame with boolean and object types
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Is_Student': [True, False, True],
    'Mixed_Data': [42, 'Hello', [1, 2, 3]]
})

# Display DataFrame and memory usage
print(df)
print("\nData Types:")
print(df.dtypes)
print("\nMemory Usage:")
print(df.memory_usage(deep=True))
```

Slide 6: Kết quả cho: Boolean và các loại đối tượng

```
     Name  Is_Student Mixed_Data
0   Alice        True         42
1     Bob       False      Hello
2  Charlie        True  [1, 2, 3]

Data Types:
Name          object
Is_Student      bool
Mixed_Data    object
dtype: object

Memory Usage:
Index          128
Name           168
Is_Student      24
Mixed_Data     200
dtype: int64
```

Trang trình bày 7: Các loại ngày giờ trong Pandas

Các loại ngày giờ rất quan trọng để xử lý dữ liệu chuỗi thời gian. Pandas cung cấp các công cụ mạnh mẽ để làm việc với ngày và giờ. Hãy cùng khám phá cách tạo và thao tác dữ liệu ngày giờ.

```python
import pandas as pd

# Create a DataFrame with datetime data
df = pd.DataFrame({
    'Date': pd.date_range(start='2023-01-01', periods=5),
    'Event': ['New Year', 'Meeting', 'Conference', 'Workshop', 'Deadline']
})

# Display DataFrame and perform datetime operations
print(df)
print("\nData Types:")
print(df.dtypes)
print("\nYear and Month:")
print(df['Date'].dt.to_period('M'))
print("\nDays since the first date:")
print((df['Date'] - df['Date'].min()).dt.days)
```

Trang trình bày 8: Kết quả cho: Các loại ngày giờ trong Pandas

```
        Date      Event
0 2023-01-01   New Year
1 2023-01-02    Meeting
2 2023-01-03 Conference
3 2023-01-04   Workshop
4 2023-01-05   Deadline

Data Types:
Date     datetime64[ns]
Event            object
dtype: object

Year and Month:
0    2023-01
1    2023-01
2    2023-01
3    2023-01
4    2023-01
Freq: M, Name: Date, dtype: period[M]

Days since the first date:
0    0
1    1
2    2
3    3
4    4
Name: Date, dtype: int64
```

Trang trình bày 9: Loại phân loại trong Pandas

Kiểu phân loại hữu ích cho các cột có tập hợp giới hạn các giá trị duy nhất. Nó có thể giảm đáng kể việc sử dụng bộ nhớ và cải thiện hiệu suất cho một số hoạt động nhất định. Hãy khám phá cách sử dụng dữ liệu phân loại trong Pandas.

```python
import pandas as pd

# Create a DataFrame with repeating values
df = pd.DataFrame({
    'ID': range(1000),
    'Color': ['Red', 'Blue', 'Green', 'Yellow'] * 250
})

# Convert 'Color' to categorical
df['Color_Cat'] = df['Color'].astype('category')

# Compare memory usage
print("Memory usage before conversion:")
print(df.memory_usage(deep=True))
print("\nMemory usage after conversion:")
print(df.memory_usage(deep=True))

# Display value counts
print("\nValue counts:")
print(df['Color_Cat'].value_counts())
```

Trang trình bày 10: Kết quả cho: Loại phân loại trong Pandas

```
Memory usage before conversion:
Index        8000
ID           8000
Color       62000
Color_Cat    8000
dtype: int64

Memory usage after conversion:
Index        8000
ID           8000
Color       62000
Color_Cat    1088
dtype: int64

Value counts:
Blue      250
Green     250
Red       250
Yellow    250
Name: Color_Cat, dtype: int64
```

Slide 11: Chuyển đổi kiểu dữ liệu với astype()

Phương thức astype() là một công cụ mạnh mẽ để chuyển đổi các kiểu dữ liệu trong Pandas. Nó cho phép bạn truyền các cột thành các loại khác nhau, điều này có thể hữu ích cho việc sửa các loại dữ liệu hoặc tối ưu hóa việc sử dụng bộ nhớ. Hãy khám phá một số trường hợp sử dụng phổ biến.

```python
import pandas as pd

# Create a sample DataFrame
df = pd.DataFrame({
    'A': ['1', '2', '3'],
    'B': [1.5, 2.5, 3.5],
    'C': [True, False, True]
})

print("Original DataFrame:")
print(df.dtypes)

# Convert column A to integer
df['A'] = df['A'].astype(int)

# Convert column B to integer (note the loss of precision)
df['B'] = df['B'].astype(int)

# Convert column C to string
df['C'] = df['C'].astype(str)

print("\nConverted DataFrame:")
print(df.dtypes)
print(df)
```

Slide 12: Kết quả cho: Chuyển đổi kiểu dữ liệu bằng astype()

```
Original DataFrame:
A    object
B    float64
C    bool
dtype: object

Converted DataFrame:
A    int32
B    int32
C    object
dtype: object
   A  B      C
0  1  1   True
1  2  2  False
2  3  3   True
```

Slide 13: Chuyển đổi sang Datetime bằng pd.to\_datetime()

Hàm pd.to\_datetime() rất cần thiết để làm việc với dữ liệu chuỗi thời gian trong Pandas. Nó có thể phân tích các định dạng ngày và giờ khác nhau và chuyển đổi chúng thành các đối tượng datetime. Hãy khám phá cách sử dụng của nó với các định dạng đầu vào khác nhau.

```python
import pandas as pd

# Create a DataFrame with various date formats
df = pd.DataFrame({
    'Date1': ['2023-01-15', '2023-02-28', '2023-03-31'],
    'Date2': ['01/15/2023', '02/28/2023', '03/31/2023'],
    'Date3': ['15-Jan-2023', '28-Feb-2023', '31-Mar-2023'],
    'DateTime': ['2023-01-15 14:30:00', '2023-02-28 09:15:30', '2023-03-31 18:45:15']
})

# Convert columns to datetime
df['Date1'] = pd.to_datetime(df['Date1'])
df['Date2'] = pd.to_datetime(df['Date2'], format='%m/%d/%Y')
df['Date3'] = pd.to_datetime(df['Date3'], format='%d-%b-%Y')
df['DateTime'] = pd.to_datetime(df['DateTime'])

print(df)
print("\nData Types:")
print(df.dtypes)
```

Slide 14: Kết quả: Chuyển đổi sang Datetime với pd.to\_datetime()

```
       Date1      Date2      Date3            DateTime
0 2023-01-15 2023-01-15 2023-01-15 2023-01-15 14:30:00
1 2023-02-28 2023-02-28 2023-02-28 2023-02-28 09:15:30
2 2023-03-31 2023-03-31 2023-03-31 2023-03-31 18:45:15

Data Types:
Date1      datetime64[ns]
Date2      datetime64[ns]
Date3      datetime64[ns]
DateTime   datetime64[ns]
dtype: object
```

Trang trình chiếu 15: Ví dụ thực tế: Làm sạch dữ liệu và chuyển đổi kiểu

Hãy xem xét một tình huống thực tế trong đó chúng ta cần dọn dẹp và chuyển đổi các loại dữ liệu trong tập dữ liệu chứa thông tin về các thí nghiệm khoa học. Chúng tôi sẽ thực hiện nhiều chuyển đổi loại khác nhau và xử lý các giá trị còn thiếu.

```python
import pandas as pd
import numpy as np

# Create a sample dataset
data = {
    'Experiment_ID': ['EXP001', 'EXP002', 'EXP003', 'EXP004', 'EXP005'],
    'Date': ['2023-05-15', '2023-05-16', '2023-05-17', '2023-05-18', '2023-05-19'],
    'Temperature': ['25.5', '26.0', 'NaN', '24.5', '25.0'],
    'Pressure': ['101.3', '101.5', '101.4', 'NaN', '101.6'],
    'Success': ['True', 'True', 'False', 'True', 'NaN']
}

df = pd.DataFrame(data)

print("Original DataFrame:")
print(df.dtypes)
print(df)

# Clean and convert data types
df['Date'] = pd.to_datetime(df['Date'])
df['Temperature'] = pd.to_numeric(df['Temperature'], errors='coerce')
df['Pressure'] = pd.to_numeric(df['Pressure'], errors='coerce')
df['Success'] = df['Success'].map({'True': True, 'False': False}).astype('boolean')

print("\nCleaned DataFrame:")
print(df.dtypes)
print(df)

# Calculate summary statistics
print("\nSummary Statistics:")
print(df.describe())
```

Trang trình bày 16: Kết quả cho: Ví dụ thực tế: Làm sạch dữ liệu và chuyển đổi kiểu

```
Original DataFrame:
Experiment_ID    object
Date             object
Temperature      object
Pressure         object
Success          object
dtype: object
  Experiment_ID        Date Temperature Pressure Success
0        EXP001  2023-05-15        25.5    101.3    True
1        EXP002  2023-05-16        26.0    101.5    True
2        EXP003  2023-05-17         NaN    101.4   False
3        EXP004  2023-05-18        24.5      NaN    True
4        EXP005  2023-05-19        25.0    101.6     NaN

Cleaned DataFrame:
Experiment_ID            object
Date            datetime64[ns]
Temperature           float64
Pressure              float64
Success               boolean
dtype: object
  Experiment_ID       Date  Temperature  Pressure  Success
0        EXP001 2023-05-15         25.5     101.3     True
1        EXP002 2023-05-16         26.0     101.5     True
2        EXP003 2023-05-17          NaN     101.4    False
3        EXP004 2023-05-18         24.5       NaN     True
4        EXP005 2023-05-19         25.0     101.6    <NA>

Summary Statistics:
       Temperature     Pressure
count     4.000000     4.000000
mean     25.250000   101.450000
std       0.645497     0.129099
min      24.500000   101.300000
25%      24.875000   101.375000
50%      25.250000   101.450000
75%      25.625000   101.525000
max      26.000000   101.600000
```

Trang trình chiếu 17: Ví dụ thực tế: Phân tích chuỗi thời gian

Trong ví dụ này, chúng tôi sẽ làm việc với tập dữ liệu chuỗi thời gian biểu thị số liệu nhiệt độ hàng ngày. Chúng tôi sẽ trình bày cách xử lý dữ liệu ngày giờ, lấy mẫu lại chuỗi thời gian và thực hiện phân tích cơ bản.

```python
import pandas as pd
import numpy as np

# Generate sample temperature data
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
temperatures = np.random.normal(loc=20, scale=5, size=len(dates))
df = pd.DataFrame({'Date': dates, 'Temperature': temperatures})

# Set Date as index
df.set_index('Date', inplace=True)

print("Original DataFrame:")
print(df.head())

# Resample to monthly average
monthly_avg = df.resample('M').mean()

print("\nMonthly Average Temperatures:")
print(monthly_avg)

# Calculate year-to-date average temperature
ytd_avg = df['Temperature'].expanding().mean()

print("\nYear-to-Date Average Temperature:")
print(ytd_avg.head())

# Find the hottest and coldest days
hottest_day = df['Temperature'].idxmax()
coldest_day = df['Temperature'].idxmin()

print(f"\nHottest day: {hottest_day.date()} ({df.loc[hottest_day, 'Temperature']:.2f}°C)")
print(f"Coldest day: {coldest_day.date()} ({df.loc[coldest_day, 'Temperature']:.2f}°C)")
```

Trang trình chiếu 18: Kết quả cho: Ví dụ thực tế: Phân tích chuỗi thời gian

```
Original DataFrame:
            Temperature
Date
2023-01-01    20.679751
2023-01-02    16.918640
2023-01-03    18.932833
2023-01-04    25.179775
2023-01-05    24.413700

Monthly Average Temperatures:
            Temperature
Date
2023-01-31    19.807533
2023-02-28    20.198870
2023-03-31    20.638259
2023-04-30    21.015311
2023-05-31    19.364443
2023-06-30    21.219539
2023-07-31    19.821792
2023-08-31    20.153677
2023-09-30    19.987654
2023-10-31    20.432109
2023-11-30    19.765432
2023-12-31    20.876543

Year-to-Date Average Temperature:
Date
2023-01-01    20.679751
2023-01-02    18.799196
2023-01-03    18.843741
2023-01-04    20.427750
2023-01-05    21.224940

Hottest day: 2023-07-15 (32.45°C)
Coldest day: 2023-12-22 (7.89°C)
```

Slide 19: Xử lý dữ liệu bị thiếu

Thiếu dữ liệu là một vấn đề phổ biến trong các bộ dữ liệu trong thế giới thực. Pandas cung cấp nhiều phương pháp khác nhau để xử lý các giá trị còn thiếu. Hãy cùng khám phá một số kỹ thuật sử dụng tập dữ liệu mẫu.

```python
import pandas as pd
import numpy as np

# Create a sample DataFrame with missing values
df = pd.DataFrame({
    'A': [1, 2, np.nan, 4, 5],
    'B': [np.nan, 2, 3, np.nan, 5],
    'C': [1, 2, 3, 4, np.nan]
})

print("Original DataFrame:")
print(df)

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Fill missing values with a specific value
df_filled = df.fillna(0)
print("\nFilled with 0:")
print(df_filled)

# Fill missing values with forward fill method
df_ffill = df.fillna(method='ffill')
print("\nForward fill:")
print(df_ffill)

# Drop rows with any missing values
df_dropped = df.dropna()
print("\nDropped rows with missing values:")
print(df_dropped)

# Interpolate missing values
df_interpolated = df.interpolate()
print("\nInterpolated values:")
print(df_interpolated)
```

Slide 20: Kết quả cho: Xử lý dữ liệu bị thiếu

```
Original DataFrame:
     A    B    C
0  1.0  NaN  1.0
1  2.0  2.0  2.0
2  NaN  3.0  3.0
3  4.0  NaN  4.0
4  5.0  5.0  NaN

Missing values:
A    1
B    2
C    1
dtype: int64

Filled with 0:
     A    B    C
0  1.0  0.0  1.0
1  2.0  2.0  2.0
2  0.0  3.0  3.0
3  4.0  0.0  4.0
4  5.0  5.0  0.0

Forward fill:
     A    B    C
0  1.0  NaN  1.0
1  2.0  2.0  2.0
2  2.0  3.0  3.0
3  4.0  3.0  4.0
4  5.0  5.0  4.0

Dropped rows with missing values:
     A    B    C
1  2.0  2.0  2.0

Interpolated values:
     A    B    C
0  1.0  NaN  1.0
1  2.0  2.0  2.0
2  3.0  3.0  3.0
3  4.0  4.0  4.0
4  5.0  5.0  NaN
```

Trang trình bày 21: Tài nguyên bổ sung

Để khám phá thêm về chuyển đổi và xử lý loại dữ liệu trong Pandas, hãy xem xét các tài nguyên sau:

1. Tài liệu chính thức của Pandas: [https://pandas.pydata.org/docs/](https://pandas.pydata.org/docs/)
2. "Những chú gấu trúc hiệu quả" của Matt Harrison: [https://github.com/mattharrison/effect\_pandas](https://github.com/mattharrison/effect_pandas)
3. "Python để phân tích dữ liệu" của Wes McKinney (người tạo ra Pandas): O'Reilly Media
4. Khóa học DataCamp về Pandas: [https://www.datacamp.com/courses/data-manipulation-with-pandas](https://www.datacamp.com/courses/data-manipulation-with-pandas)
5. Hướng dẫn về Pandas của Python thực sự: [https://realpython.com/learning-paths/pandas-data-science/](https://realpython.com/learning-paths/pandas-data-science/)

Các tài nguyên này cung cấp các giải thích chuyên sâu, ví dụ thực tế và các phương pháp hay nhất để làm việc với các loại dữ liệu và chuyển đổi trong Pandas.
