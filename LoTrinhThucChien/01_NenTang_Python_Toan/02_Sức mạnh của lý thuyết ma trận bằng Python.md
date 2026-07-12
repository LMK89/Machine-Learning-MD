## Sức mạnh của Lý thuyết Ma trận sử dụng Python
Slide 1: Giới thiệu về Lý thuyết Ma trận

Lý thuyết ma trận là một nhánh cơ bản của toán học với những ứng dụng rộng rãi trong nhiều lĩnh vực khác nhau. Nó cung cấp một khuôn khổ mạnh mẽ để giải quyết các vấn đề phức tạp trong đại số tuyến tính, đồ họa máy tính, cơ học lượng tử, v.v. Bài trình bày này sẽ khám phá các khái niệm chính và ứng dụng thực tế của lý thuyết ma trận bằng Python.

```python
import numpy as np

# Create a 2x2 matrix
A = np.array([[1, 2],
              [3, 4]])

print("Matrix A:")
print(A)

# Basic matrix operations
print("\nTranspose of A:")
print(A.T)

print("\nDeterminant of A:")
print(np.linalg.det(A))
```

Slide 2: Các phép toán ma trận: Phép cộng và phép trừ

Phép cộng và phép trừ ma trận là các phép toán cơ bản được thực hiện theo từng phần tử. Các phép toán này chỉ được xác định cho các ma trận có cùng kích thước. Hãy khám phá cách thực hiện các thao tác này bằng NumPy.

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("Matrix A:")
print(A)
print("\nMatrix B:")
print(B)

# Addition
print("\nA + B:")
print(A + B)

# Subtraction
print("\nA - B:")
print(A - B)
```

Slide 3: Phép nhân ma trận

Phép nhân ma trận là một phép toán quan trọng trong lý thuyết ma trận. Không giống như phép cộng và phép trừ, phép nhân không có tính giao hoán (A \* B ≠ B \* A). Số cột trong ma trận thứ nhất phải bằng số hàng trong ma trận thứ hai.

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("Matrix A:")
print(A)
print("\nMatrix B:")
print(B)

# Matrix multiplication
C = np.dot(A, B)
print("\nA * B:")
print(C)

# Note: A * B ≠ B * A
D = np.dot(B, A)
print("\nB * A:")
print(D)
```

Slide 4: Chuyển vị ma trận

Phép chuyển vị của ma trận có được bằng cách hoán đổi các hàng và cột của nó. Chuyển vị là một phép toán cơ bản trong lý thuyết ma trận và thường được sử dụng trong nhiều bài toán và tính toán khác nhau.

```python
import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6]])

print("Original matrix A:")
print(A)

# Transpose the matrix
A_transposed = A.T

print("\nTransposed matrix A:")
print(A_transposed)

# Verify properties
print("\nIs (A^T)^T = A?", np.array_equal(A_transposed.T, A))
```

Trang trình bày 5: Yếu tố quyết định

Định thức là một giá trị vô hướng có thể được tính từ ma trận vuông. Nó có nhiều ứng dụng quan trọng, bao gồm giải hệ phương trình tuyến tính và tìm ma trận nghịch đảo. Hãy tính các yếu tố quyết định bằng NumPy.

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print("Matrix A:")
print(A)
print("Determinant of A:", np.linalg.det(A))

print("\nMatrix B:")
print(B)
print("Determinant of B:", np.linalg.det(B))

# Create a singular matrix
C = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9]])
print("\nSingular Matrix C:")
print(C)
print("Determinant of C:", np.linalg.det(C))
```

Slide 6: Ma trận nghịch đảo

Nghịch đảo của ma trận vuông A, ký hiệu là A^(-1), là ma trận mà khi nhân với A sẽ thu được ma trận đẳng thức. Không phải tất cả các ma trận đều nghịch đảo; chỉ các ma trận không số ít (định thức ≠ 0) mới khả nghịch.

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])

print("Matrix A:")
print(A)

# Calculate the inverse
A_inv = np.linalg.inv(A)

print("\nInverse of A:")
print(A_inv)

# Verify: A * A^(-1) = I
I = np.dot(A, A_inv)
print("\nA * A^(-1):")
print(np.round(I, decimals=10))  # Round to avoid floating-point errors

# Try inverting a singular matrix
B = np.array([[1, 2], [2, 4]])
try:
    B_inv = np.linalg.inv(B)
except np.linalg.LinAlgError as e:
    print("\nError inverting singular matrix:", str(e))
```

Slide 7: Giá trị riêng và vectơ riêng

Giá trị riêng và vectơ riêng là những khái niệm cơ bản trong lý thuyết ma trận. Vector riêng của ma trận vuông A là vectơ v khác 0, khi nhân với A sẽ thu được bội số vô hướng của chính nó. Đại lượng vô hướng này được gọi là giá trị riêng.

```python
import numpy as np

A = np.array([[4, -2], [1, 1]])

print("Matrix A:")
print(A)

# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

print("\nEigenvalues:")
print(eigenvalues)

print("\nEigenvectors:")
print(eigenvectors)

# Verify Av = λv for the first eigenpair
lambda1 = eigenvalues[0]
v1 = eigenvectors[:, 0]

print("\nVerification:")
print("Av =", np.dot(A, v1))
print("λv =", lambda1 * v1)
```

Slide 8: Phân tích ma trận: Phân tích LU

Phân rã ma trận là một kỹ thuật mạnh mẽ trong lý thuyết ma trận. Phân rã LU phân tích ma trận thành tích của ma trận tam giác dưới (L) và ma trận tam giác trên (U). Sự phân rã này rất hữu ích cho việc giải các hệ thống tuyến tính và tính toán các định thức.

```python
import numpy as np
from scipy.linalg import lu

A = np.array([[2, 1, 1],
              [1, 3, 2],
              [1, 0, 0]])

print("Matrix A:")
print(A)

# Perform LU decomposition
P, L, U = lu(A)

print("\nLower triangular matrix L:")
print(L)

print("\nUpper triangular matrix U:")
print(U)

# Verify A = P * L * U
print("\nVerification:")
print("A =\n", A)
print("P * L * U =\n", np.dot(P, np.dot(L, U)))
```

Slide 9: Giải hệ phương trình tuyến tính

Một trong những ứng dụng quan trọng nhất của lý thuyết ma trận là giải hệ phương trình tuyến tính. Chúng ta có thể sử dụng các phép toán ma trận để giải các hệ thống này một cách hiệu quả.

```python
import numpy as np

# System of equations:
# 2x + y = 8
# -3x + y = -11

A = np.array([[2, 1],
              [-3, 1]])
b = np.array([8, -11])

print("Matrix A:")
print(A)
print("\nVector b:")
print(b)

# Solve the system
x = np.linalg.solve(A, b)

print("\nSolution x:")
print(x)

# Verify the solution
print("\nVerification:")
print("Ax =", np.dot(A, x))
print("b  =", b)
```

Slide 10: Xếp hạng ma trận

Thứ hạng của ma trận là thứ nguyên của không gian vectơ được kéo dài bởi các cột (hoặc hàng) của nó. Đó là thước đo tính "không suy biến" của hệ phương trình tuyến tính được biểu thị bằng ma trận.

```python
import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

B = np.array([[1, 2, 3],
              [2, 4, 6],
              [3, 6, 9]])

print("Matrix A:")
print(A)
print("Rank of A:", np.linalg.matrix_rank(A))

print("\nMatrix B:")
print(B)
print("Rank of B:", np.linalg.matrix_rank(B))

# Create a visualization of the column space
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 5))

ax1 = fig.add_subplot(121, projection='3d')
ax1.quiver(0, 0, 0, *A.T, length=0.1, normalize=True)
ax1.set_title("Column Space of A")

ax2 = fig.add_subplot(122, projection='3d')
ax2.quiver(0, 0, 0, *B.T, length=0.1, normalize=True)
ax2.set_title("Column Space of B")

plt.tight_layout()
plt.show()
```

Slide 11: Định mức ma trận

Các định mức ma trận cung cấp một cách để đo lường “kích thước” của ma trận. Chúng rất hữu ích trong việc phân tích tính ổn định của các thuật toán số và phân tích lỗi. Hãy cùng khám phá một số chuẩn mực ma trận phổ biến.

```python
import numpy as np

A = np.array([[1, 2],
              [3, 4]])

print("Matrix A:")
print(A)

# Frobenius norm
frob_norm = np.linalg.norm(A, 'fro')
print("\nFrobenius norm:", frob_norm)

# Spectral norm (2-norm)
spectral_norm = np.linalg.norm(A, 2)
print("Spectral norm:", spectral_norm)

# 1-norm (maximum absolute column sum)
one_norm = np.linalg.norm(A, 1)
print("1-norm:", one_norm)

# Infinity norm (maximum absolute row sum)
inf_norm = np.linalg.norm(A, np.inf)
print("Infinity norm:", inf_norm)
```

Slide 12: Ví dụ thực tế: Nén ảnh

Lý thuyết ma trận đóng một vai trò quan trọng trong kỹ thuật nén ảnh. Một phương pháp như vậy là Phân tách giá trị số ít (SVD), có thể được sử dụng để ước chừng các hình ảnh có ít điểm dữ liệu hơn.

```python
import numpy as np
import matplotlib.pyplot as plt

# Create a simple 100x100 grayscale image
image = np.zeros((100, 100))
image[25:75, 25:75] = 1  # White square in the middle

# Perform SVD
U, s, Vt = np.linalg.svd(image)

# Reconstruct the image using different numbers of singular values
def reconstruct(U, s, Vt, k):
    return np.matrix(U[:, :k]) * np.diag(s[:k]) * np.matrix(Vt[:k, :])

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs[0, 0].imshow(image, cmap='gray')
axs[0, 0].set_title('Original')
axs[0, 1].imshow(reconstruct(U, s, Vt, 5), cmap='gray')
axs[0, 1].set_title('k = 5')
axs[1, 0].imshow(reconstruct(U, s, Vt, 10), cmap='gray')
axs[1, 0].set_title('k = 10')
axs[1, 1].imshow(reconstruct(U, s, Vt, 20), cmap='gray')
axs[1, 1].set_title('k = 20')

plt.tight_layout()
plt.show()
```

Trang trình chiếu 13: Ví dụ thực tế: Xích Markov

Chuỗi Markov là hệ thống toán học chuyển từ trạng thái này sang trạng thái khác theo các quy tắc xác suất nhất định. Chúng được biểu diễn bằng ma trận ngẫu nhiên và có ứng dụng trong nhiều lĩnh vực khác nhau, bao gồm vật lý, sinh học và khoa học máy tính.

```python
import numpy as np
import matplotlib.pyplot as plt

# Define the transition matrix
P = np.array([[0.7, 0.2, 0.1],
              [0.3, 0.5, 0.2],
              [0.2, 0.3, 0.5]])

# Initial state
x0 = np.array([1, 0, 0])

# Simulate the Markov chain
steps = 20
X = np.zeros((steps, 3))
X[0] = x0

for i in range(1, steps):
    X[i] = np.dot(X[i-1], P)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(X[:, 0], label='State 0')
plt.plot(X[:, 1], label='State 1')
plt.plot(X[:, 2], label='State 2')
plt.xlabel('Steps')
plt.ylabel('Probability')
plt.title('Markov Chain State Probabilities')
plt.legend()
plt.grid(True)
plt.show()

# Calculate the stationary distribution
eigenvalues, eigenvectors = np.linalg.eig(P.T)
stationary = eigenvectors[:, np.isclose(eigenvalues, 1)].real
stationary /= stationary.sum()

print("Stationary distribution:", stationary.flatten())
```

Trang trình bày 14: Tài nguyên bổ sung

Đối với những người quan tâm đến việc tìm hiểu sâu hơn về lý thuyết ma trận và các ứng dụng của nó, đây là một số tài nguyên có giá trị:

1. "Phân tích ma trận" của Roger A. Horn và Charles R. Johnson ([https://arxiv.org/abs/1907.09263](https://arxiv.org/abs/1907.09263))
2. "Đại số tuyến tính số" của Lloyd N. Trefethen và David Bau III
3. "Giới thiệu về Đại số tuyến tính" của Gilbert Strang (MIT OpenCourseWare)
4. "Sách dạy nấu ăn ma trận" của Kaare Brandt Petersen và Michael Syskind Pedersen ([https://arxiv.org/abs/2111.11176](https://arxiv.org/abs/2111.11176))

Những tài nguyên này cung cấp những giải thích, bằng chứng chuyên sâu và những ứng dụng nâng cao của các khái niệm lý thuyết ma trận.
