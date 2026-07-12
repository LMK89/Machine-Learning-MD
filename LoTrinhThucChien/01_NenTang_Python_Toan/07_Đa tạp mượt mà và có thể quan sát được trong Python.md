## Làm mịn đa tạp và quan sát được trong Python
Slide 1: Giới thiệu về Đa tạp trơn

Đa tạp trơn là đối tượng cơ bản trong hình học vi phân, khái quát hóa khái niệm đường cong và bề mặt theo các chiều cao hơn. Chúng cung cấp một khuôn khổ để nghiên cứu các cấu trúc hình học giống với không gian Euclide một cách cục bộ.

```python
import numpy as np
import matplotlib.pyplot as plt

def sphere_coordinates(u, v):
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    return x, y, z

u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
u, v = np.meshgrid(u, v)

x, y, z = sphere_coordinates(u, v)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, cmap='viridis')
ax.set_title('Sphere: Example of a 2D Smooth Manifold')
plt.show()
```

Trang trình bày 2: Tọa độ và biểu đồ địa phương

Một đa tạp trơn được trang bị hệ tọa độ cục bộ gọi là biểu đồ. Những biểu đồ này cho phép chúng ta mô tả đa tạp cục bộ bằng cách sử dụng tọa độ Euclide quen thuộc.

```python
import numpy as np
import matplotlib.pyplot as plt

def stereographic_projection(x, y, z):
    u = x / (1 - z)
    v = y / (1 - z)
    return u, v

theta = np.linspace(0, 2 * np.pi, 100)
phi = np.linspace(0, np.pi, 50)
theta, phi = np.meshgrid(theta, phi)

x = np.sin(phi) * np.cos(theta)
y = np.sin(phi) * np.sin(theta)
z = np.cos(phi)

u, v = stereographic_projection(x, y, z)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot_surface(x, y, z, cmap='viridis')
ax1.set_title('Sphere')
ax2.plot(u, v, 'b.', alpha=0.1)
ax2.set_title('Stereographic Projection (Chart)')
plt.show()
```

Trang trình bày 3: Không gian tiếp tuyến và vectơ

Không gian tiếp tuyến rất quan trọng trong việc tìm hiểu cấu trúc cục bộ của đa tạp trơn. Chúng đại diện cho không gian của tất cả các hướng có thể có trong đó người ta có thể di chuyển trên đa tạp tại một điểm nhất định.

```python
import numpy as np
import matplotlib.pyplot as plt

def sphere_point(theta, phi):
    return np.array([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi)
    ])

def tangent_vectors(theta, phi):
    p = sphere_point(theta, phi)
    v1 = np.array([
        -np.sin(theta),
        np.cos(theta),
        0
    ])
    v2 = np.array([
        np.cos(theta) * np.cos(phi),
        np.sin(theta) * np.cos(phi),
        -np.sin(phi)
    ])
    return p, v1, v2

theta, phi = np.pi/4, np.pi/3
p, v1, v2 = tangent_vectors(theta, phi)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))

ax.plot_surface(x, y, z, color='b', alpha=0.2)
ax.quiver(*p, *v1, color='r', length=0.2)
ax.quiver(*p, *v2, color='g', length=0.2)
ax.set_title('Tangent Vectors on a Sphere')
plt.show()
```

Slide 4: Chức năng mượt mà trên Manifold

Các hàm trơn trên đa tạp là cần thiết để xác định các khái niệm hình học khác nhau. Các hàm này phải khác biệt khi được tạo bằng bản đồ biểu đồ.

```python
import numpy as np
import matplotlib.pyplot as plt

def sphere_to_plane(theta, phi):
    return theta, phi

def height_function(theta, phi):
    return np.cos(phi)

theta = np.linspace(0, 2*np.pi, 100)
phi = np.linspace(0, np.pi, 50)
theta, phi = np.meshgrid(theta, phi)

x = np.sin(phi) * np.cos(theta)
y = np.sin(phi) * np.sin(theta)
z = np.cos(phi)

h = height_function(theta, phi)

fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(x, y, z, facecolors=plt.cm.viridis(h), alpha=0.7)
ax1.set_title('Height Function on Sphere')

ax2 = fig.add_subplot(122)
im = ax2.imshow(h, extent=[0, 2*np.pi, 0, np.pi], origin='lower', aspect='auto', cmap='viridis')
ax2.set_title('Height Function in Chart Coordinates')
ax2.set_xlabel('θ')
ax2.set_ylabel('φ')
plt.colorbar(im)

plt.show()
```

Trang trình bày 5: Bản đồ có thể phân biệt và các bước đẩy

Các bản đồ khả vi giữa các đa tạp tạo ra các bản đồ tuyến tính giữa các không gian tiếp tuyến của chúng, được gọi là các phần đẩy về phía trước. Đây là những điều quan trọng để hiểu cách các cấu trúc hình học biến đổi dưới bản đồ.

```python
import numpy as np
import matplotlib.pyplot as plt

def sphere_to_cylinder(theta, phi):
    return theta, np.cos(phi)

def pushforward(theta, phi):
    return np.array([[1, 0], [0, -np.sin(phi)]])

theta = np.linspace(0, 2*np.pi, 20)
phi = np.linspace(0, np.pi, 10)
theta, phi = np.meshgrid(theta, phi)

x = np.sin(phi) * np.cos(theta)
y = np.sin(phi) * np.sin(theta)
z = np.cos(phi)

u, v = sphere_to_cylinder(theta, phi)

fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(x, y, z, alpha=0.7)
ax1.set_title('Sphere')

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(u, v, np.zeros_like(u), alpha=0.7)
ax2.set_title('Cylinder')

for i in range(5):
    for j in range(5):
        t, p = theta[i,j], phi[i,j]
        pf = pushforward(t, p)
        v1, v2 = pf @ np.eye(2)
        ax1.quiver(x[i,j], y[i,j], z[i,j], *v1, 0, color='r', length=0.1)
        ax1.quiver(x[i,j], y[i,j], z[i,j], *v2, 0, color='g', length=0.1)
        ax2.quiver(u[i,j], v[i,j], 0, *v1, 0, color='r', length=0.1)
        ax2.quiver(u[i,j], v[i,j], 0, *v2, 0, color='g', length=0.1)

plt.show()
```

Slide 6: Các dạng vi phân

Dạng vi phân là các bản đồ phản đối xứng, đa tuyến trên các không gian tiếp tuyến. Chúng cung cấp một cách độc lập với tọa độ để tích hợp trên các đa tạp.

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def sphere_coordinates(u, v):
    return np.array([np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)])

def area_form(u, v):
    return np.sin(v)

u = np.linspace(0, 2*np.pi, 30)
v = np.linspace(0, np.pi, 20)
u, v = np.meshgrid(u, v)

x, y, z = sphere_coordinates(u, v)
omega = area_form(u, v)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(x, y, z, facecolors=plt.cm.viridis(omega), alpha=0.7)
ax.set_title('Area Form on Sphere')

plt.colorbar(surf, ax=ax, label='Magnitude of Area Form')
plt.show()

# Compute the total area of the sphere
total_area = np.sum(omega) * (2*np.pi/30) * (np.pi/20)
print(f"Computed area of the sphere: {total_area:.4f}")
print(f"Actual area of unit sphere: {4*np.pi:.4f}")
```

Trang trình bày 7: Trường và dòng vectơ

Trường vectơ gán một vectơ tiếp tuyến cho mỗi điểm trên đa tạp. Chúng tạo ra các luồng, là các họ một tham số của các dạng khác nhau.

```python
import numpy as np
import matplotlib.pyplot as plt

def vector_field(x, y):
    return -y, x

def flow(x0, y0, t):
    return x0 * np.cos(t) - y0 * np.sin(t), x0 * np.sin(t) + y0 * np.cos(t)

x = np.linspace(-2, 2, 20)
y = np.linspace(-2, 2, 20)
X, Y = np.meshgrid(x, y)

U, V = vector_field(X, Y)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.quiver(X, Y, U, V)
ax1.set_title('Vector Field')

for x0, y0 in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
    t = np.linspace(0, 2*np.pi, 100)
    x, y = flow(x0, y0, t)
    ax2.plot(x, y)

ax2.set_title('Flow of the Vector Field')
plt.show()
```

Slide 8: Nhóm Lie và Đại số Lie

Nhóm Lie là các đa tạp trơn có cấu trúc nhóm tương thích. Không gian tiếp tuyến của chúng tại đẳng thức tạo thành đại số Lie, đại số này thể hiện cấu trúc cục bộ của nhóm.

```python
import numpy as np
import matplotlib.pyplot as plt

def rotation_matrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])

def exp_map(A):
    return np.linalg.matrix_power(np.eye(2) + A/1000, 1000)

theta = np.linspace(0, 2*np.pi, 100)
circle_x = np.cos(theta)
circle_y = np.sin(theta)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(circle_x, circle_y)
ax1.set_title('SO(2) Lie Group')
ax1.set_aspect('equal')

# Lie algebra elements
X = np.array([[0, -1], [1, 0]])
Y = np.array([[0, -2], [2, 0]])

t = np.linspace(0, 1, 100)
exp_tX = np.array([exp_map(t_i * X) for t_i in t])
exp_tY = np.array([exp_map(t_i * Y) for t_i in t])

ax2.plot(exp_tX[:, 0, 0], exp_tX[:, 1, 0], label='exp(tX)')
ax2.plot(exp_tY[:, 0, 0], exp_tY[:, 1, 0], label='exp(tY)')
ax2.set_title('Exponential Map from Lie Algebra to Lie Group')
ax2.legend()
ax2.set_aspect('equal')

plt.show()
```

Trang trình bày 9: Số liệu Riemannian

Số liệu Riemannian xác định khái niệm về khoảng cách và góc trên đa tạp, cho phép chúng ta đo chiều dài, diện tích và thể tích.

```python
import numpy as np
import matplotlib.pyplot as plt

def metric_tensor(u, v):
    return np.array([[1, 0], [0, np.sin(u)**2]])

def geodesic(u0, v0, du, dv, t):
    u = u0 + du * t
    v = v0 + dv * t / np.sin(u0)
    return u, v

u = np.linspace(0, np.pi, 100)
v = np.linspace(0, 2*np.pi, 100)
U, V = np.meshgrid(u, v)

X = np.sin(U) * np.cos(V)
Y = np.sin(U) * np.sin(V)
Z = np.cos(U)

fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, Z, alpha=0.7)
ax1.set_title('Sphere with Geodesics')

# Plot some geodesics
for u0, v0, du, dv in [(np.pi/4, 0, 0, 1), (np.pi/2, 0, 1, 1), (np.pi/4, np.pi/2, 1, 0)]:
    t = np.linspace(0, 2*np.pi, 100)
    u, v = geodesic(u0, v0, du, dv, t)
    x = np.sin(u) * np.cos(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(u)
    ax1.plot(x, y, z, color='r')

ax2 = fig.add_subplot(122)
im = ax2.imshow(metric_tensor(U, V)[1,1], extent=[0, 2*np.pi, 0, np.pi],
                origin='lower', aspect='auto', cmap='viridis')
ax2.set_title('Metric Tensor Component g_φφ')
ax2.set_xlabel('φ')
ax2.set_ylabel('θ')
plt.colorbar(im)

plt.show()
```

Slide 10: Kết nối và truyền tải song song

Các kết nối cung cấp một cách để so sánh các vectơ tiếp tuyến tại các điểm khác nhau trên một đa tạp, cho phép khái niệm vận chuyển song song.

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def sphere_point(theta, phi):
    return np.array([np.sin(phi) * np.cos(theta),
                     np.sin(phi) * np.sin(theta),
                     np.cos(phi)])

def parallel_transport(theta, phi, v, t):
    # Simplified parallel transport along a great circle
    rotation = np.array([[np.cos(t), -np.sin(t)],
                         [np.sin(t), np.cos(t)]])
    return rotation @ v

# Generate points on a great circle
theta = np.linspace(0, 2*np.pi, 100)
phi = np.pi/2  # Equator
points = np.array([sphere_point(t, phi) for t in theta])

# Initial vector to transport
v0 = np.array([0, 1])

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the sphere
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(x, y, z, color='b', alpha=0.1)

# Plot the great circle and transported vectors
ax.plot(points[:, 0], points[:, 1], points[:, 2], color='r')
for i in range(0, len(theta), 10):
    p = points[i]
    v = parallel_transport(theta[i], phi, v0, theta[i])
    ax.quiver(p[0], p[1], p[2], v[0], v[1], 0, color='g', length=0.1)

ax.set_title('Parallel Transport on a Sphere')
plt.show()
```

Slide 11: Độ cong

Độ cong đo lường độ lệch của một đa tạp khỏi trạng thái phẳng. Nó có thể được biểu thị thông qua tensor độ cong Riemann, định lượng sự thất bại của vận chuyển song song không phụ thuộc vào đường đi.

```python
import numpy as np
import matplotlib.pyplot as plt

def gaussian_curvature(u, v):
    # Gaussian curvature of a sphere is constant
    return np.ones_like(u)

def sectional_curvature(u, v):
    # Sectional curvature of a sphere is also constant
    return np.ones_like(u)

u = np.linspace(0, np.pi, 100)
v = np.linspace(0, 2*np.pi, 100)
U, V = np.meshgrid(u, v)

X = np.sin(U) * np.cos(V)
Y = np.sin(U) * np.sin(V)
Z = np.cos(U)

K = gaussian_curvature(U, V)

fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, Z, facecolors=plt.cm.viridis(K), alpha=0.7)
ax1.set_title('Sphere colored by Gaussian Curvature')

ax2 = fig.add_subplot(122)
im = ax2.imshow(K, extent=[0, 2*np.pi, 0, np.pi], origin='lower', aspect='auto', cmap='viridis')
ax2.set_title('Gaussian Curvature Map')
ax2.set_xlabel('φ')
ax2.set_ylabel('θ')
plt.colorbar(im)

plt.show()
```

Slide 12: Trắc địa

Đường trắc địa là những đường cong giảm thiểu cục bộ khoảng cách giữa các điểm trên một đa tạp. Họ khái quát hóa khái niệm đường thẳng thành không gian cong.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def geodesic_equation(y, t, a, b):
    theta, phi, dtheta, dphi = y
    d2theta = 2 * np.tan(phi) * dtheta * dphi
    d2phi = -np.sin(phi) * np.cos(phi) * dtheta**2
    return [dtheta, dphi, d2theta, d2phi]

def solve_geodesic(theta0, phi0, dtheta0, dphi0, t):
    y0 = [theta0, phi0, dtheta0, dphi0]
    solution = odeint(geodesic_equation, y0, t, args=(0, 0))
    return solution[:, 0], solution[:, 1]

t = np.linspace(0, 10, 1000)
theta, phi = solve_geodesic(0, np.pi/2, 1, 0, t)

X = np.sin(phi) * np.cos(theta)
Y = np.sin(phi) * np.sin(theta)
Z = np.cos(phi)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the sphere
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(x, y, z, color='b', alpha=0.1)

# Plot the geodesic
ax.plot(X, Y, Z, color='r', linewidth=2)
ax.set_title('Geodesic on a Sphere')

plt.show()
```

Slide 13: Ví dụ thực tế: Bề mặt Trái đất

Bề mặt Trái đất có thể được coi gần đúng như một đa tạp trơn. Hiểu hình học của nó là rất quan trọng cho việc điều hướng, bản đồ và địa vật lý.

```python
import numpy as np
import matplotlib.pyplot as plt

def haversine_distance(lat1, lon1, lat2, lon2, R=6371):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

# Example: Distance between New York and Tokyo
ny_lat, ny_lon = 40.7128, -74.0060
tokyo_lat, tokyo_lon = 35.6762, 139.6503

distance = haversine_distance(ny_lat, ny_lon, tokyo_lat, tokyo_lon)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the Earth
phi = np.linspace(0, np.pi, 100)
theta = np.linspace(0, 2*np.pi, 100)
x = np.outer(np.sin(phi), np.cos(theta))
y = np.outer(np.sin(phi), np.sin(theta))
z = np.outer(np.cos(phi), np.ones_like(theta))

ax.plot_surface(x, y, z, color='b', alpha=0.3)

# Plot New York and Tokyo
ny = np.array([np.cos(np.radians(ny_lat)) * np.cos(np.radians(ny_lon)),
               np.cos(np.radians(ny_lat)) * np.sin(np.radians(ny_lon)),
               np.sin(np.radians(ny_lat))])
tokyo = np.array([np.cos(np.radians(tokyo_lat)) * np.cos(np.radians(tokyo_lon)),
                  np.cos(np.radians(tokyo_lat)) * np.sin(np.radians(tokyo_lon)),
                  np.sin(np.radians(tokyo_lat))])

ax.scatter(*ny, color='r', s=50, label='New York')
ax.scatter(*tokyo, color='g', s=50, label='Tokyo')

# Plot the geodesic
t = np.linspace(0, 1, 100)
path = np.outer(1-t, ny) + np.outer(t, tokyo)
path /= np.linalg.norm(path, axis=1)[:, np.newaxis]
ax.plot(*path.T, color='r', linewidth=2)

ax.set_title(f'Geodesic on Earth: NY to Tokyo (Distance: {distance:.2f} km)')
ax.legend()

plt.show()
```

Slide 14: Ví dụ thực tế: Thuyết tương đối rộng

Thuyết tương đối rộng của Einstein mô tả lực hấp dẫn là độ cong của đa tạp không thời gian 4 chiều. Ví dụ này minh họa một mô hình đơn giản về độ cong của không thời gian.

```python
import numpy as np
import matplotlib.pyplot as plt

def schwarzschild_metric(r, M=1):
    c = 1  # Speed of light
    Rs = 2 * M * c**2  # Schwarzschild radius
    g00 = -(1 - Rs/r)
    g11 = 1 / (1 - Rs/r)
    g22 = r**2
    g33 = r**2 * np.sin(np.pi/4)**2  # Fixed θ = π/4 for simplicity
    return np.diag([g00, g11, g22, g33])

r = np.linspace(2.1, 10, 100)  # Start slightly outside event horizon
metric_components = np.array([schwarzschild_metric(ri) for ri in r])

fig, ax = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Schwarzschild Metric Components")

for i in range(4):
    row = i // 2
    col = i % 2
    ax[row, col].plot(r, metric_components[:, i, i])
    ax[row, col].set_xlabel('r')
    ax[row, col].set_ylabel(f'g{i}{i}')
    ax[row, col].set_title(f'Metric Component g{i}{i}')

plt.tight_layout()
plt.show()
```

Trang trình bày 15: Tài nguyên bổ sung

Đối với những người quan tâm đến việc tìm hiểu sâu hơn về đa tạp trơn và các chủ đề liên quan, đây là một số tài nguyên có giá trị:

1. "Giới thiệu về đa tạp trơn tru" của John M. Lee ArXiv: [https://arxiv.org/abs/math/9940009](https://arxiv.org/abs/math/9940009)
2. "Hình học vi phân của đường cong và bề mặt" của Manfredo P. do Carmo (Không có trên ArXiv, nhưng sách giáo khoa được sử dụng rộng rãi)
3. "Ghi chú về Hình học vi phân và Nhóm Lie" của Jean Gallier ArXiv: [https://arxiv.org/abs/0805.0287](https://arxiv.org/abs/0805.0287)
4. "Hình học Riemannian: Giới thiệu hiện đại" của Isaac Chavel ArXiv: [https://arxiv.org/abs/math/0306138](https://arxiv.org/abs/math/0306138)

Những tài nguyên này cung cấp sự khám phá sâu hơn về các khái niệm được đề cập trong bài trình bày này, đưa ra các phương pháp xử lý toán học nghiêm ngặt và các chủ đề nâng cao trong hình học vi phân.
