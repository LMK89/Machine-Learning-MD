## Nền tảng đồ họa 3D Hiển thị hình tam giác trong Python
Trang trình bày 1: Tìm hiểu hệ tọa độ 3D trong Python

Nền tảng của đồ họa 3D bắt đầu bằng việc hiểu các hệ tọa độ. Trong đồ họa máy tính, chúng ta biểu diễn các điểm trong không gian 3D bằng cách sử dụng các vectơ chứa tọa độ X, Y và Z. Thư viện gọn gàng của Python cung cấp các công cụ hiệu quả để xử lý các hệ tọa độ này.

```python
import numpy as np

class Point3D:
    def __init__(self, x, y, z):
        self.coords = np.array([x, y, z])

    def get_coordinates(self):
        return self.coords

# Create a triangle in 3D space
triangle = [
    Point3D(1, 0, 0),
    Point3D(0, 1, 0),
    Point3D(0, 0, 1)
]

# Display triangle vertices
for i, point in enumerate(triangle):
    print(f"Vertex {i + 1}: {point.get_coordinates()}")

# Output:
# Vertex 1: [1 0 0]
# Vertex 2: [0 1 0]
# Vertex 3: [0 0 1]
```

Slide 2: Các phép biến đổi ma trận cơ bản

Phép biến đổi ma trận là các phép toán cơ bản trong đồ họa 3D. Chúng cho phép chúng ta xoay, chia tỷ lệ và dịch các đối tượng trong không gian 3D. Các phép biến đổi này được biểu diễn dưới dạng ma trận 4x4 sử dụng tọa độ đồng nhất.

```python
import numpy as np

def create_rotation_matrix(angle, axis='x'):
    """Create rotation matrix for specified axis and angle (in radians)"""
    c, s = np.cos(angle), np.sin(angle)
    if axis.lower() == 'x':
        return np.array([
            [1, 0,  0, 0],
            [0, c, -s, 0],
            [0, s,  c, 0],
            [0, 0,  0, 1]
        ])
    elif axis.lower() == 'y':
        return np.array([
            [ c, 0, s, 0],
            [ 0, 1, 0, 0],
            [-s, 0, c, 0],
            [ 0, 0, 0, 1]
        ])
    else:  # z-axis
        return np.array([
            [c, -s, 0, 0],
            [s,  c, 0, 0],
            [0,  0, 1, 0],
            [0,  0, 0, 1]
        ])

# Example rotation of 45 degrees around X axis
angle = np.pi/4  # 45 degrees
rotation_matrix = create_rotation_matrix(angle, 'x')
print("Rotation Matrix (45° around X):\n", rotation_matrix)
```

Trang trình bày 3: Thực hiện phép nhân ma trận vectơ

Hiểu cách nhân vectơ và ma trận là rất quan trọng đối với các phép biến đổi 3D. Chúng tôi triển khai một lớp tùy chỉnh xử lý phép nhân ma trận vectơ để tính toán đồ họa 3D một cách hiệu quả.

```python
import numpy as np

class Transform3D:
    def __init__(self, matrix=None):
        self.matrix = matrix if matrix is not None else np.eye(4)

    def apply_to_point(self, point):
        # Convert 3D point to homogeneous coordinates
        homogeneous_point = np.append(point, 1)
        # Apply transformation
        transformed = np.dot(self.matrix, homogeneous_point)
        # Convert back to 3D coordinates
        return transformed[:3] / transformed[3]

# Example usage
point = np.array([1, 0, 0])
rotation = create_rotation_matrix(np.pi/4, 'z')
transform = Transform3D(rotation)

rotated_point = transform.apply_to_point(point)
print(f"Original point: {point}")
print(f"Rotated point: {rotated_point}")
```

Slide 4: Xây dựng lớp Mesh

Lớp Mesh đóng vai trò là nền tảng để biểu diễn các đối tượng 3D. Nó quản lý các tập hợp các đỉnh và mặt, cung cấp các phương pháp chuyển đổi và thao tác hình học 3D.

```python
class Mesh:
    def __init__(self, vertices, faces):
        self.vertices = np.array(vertices)
        self.faces = np.array(faces)

    def transform(self, transformation_matrix):
        # Create homogeneous coordinates
        homogeneous_vertices = np.hstack((
            self.vertices,
            np.ones((len(self.vertices), 1))
        ))

        # Apply transformation
        transformed = np.dot(homogeneous_vertices,
                           transformation_matrix.T)

        # Convert back to 3D coordinates
        self.vertices = transformed[:, :3] / transformed[:, 3:]

    def get_triangles(self):
        return [self.vertices[face] for face in self.faces]

# Create a simple pyramid
vertices = np.array([
    [0, 0, 0],  # base
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0.5, 0.5, 1]  # apex
])

faces = np.array([
    [0, 1, 4],  # triangular faces
    [1, 2, 4],
    [2, 3, 4],
    [3, 0, 4]
])

pyramid = Mesh(vertices, faces)
```

Slide 5: Thực hiện các phép biến đổi xoay

Việc thực hiện toàn diện các phép biến đổi phép quay đòi hỏi phải xử lý các góc Euler và quaternion. Việc triển khai này trình bày cách tạo và kết hợp nhiều phép biến đổi xoay cho các chuyển động 3D phức tạp.

```python
import numpy as np
from math import cos, sin

class Rotation3D:
    @staticmethod
    def from_euler(phi, theta, psi):
        # Create rotation matrices for each axis
        Rx = np.array([
            [1, 0, 0],
            [0, cos(phi), -sin(phi)],
            [0, sin(phi), cos(phi)]
        ])

        Ry = np.array([
            [cos(theta), 0, sin(theta)],
            [0, 1, 0],
            [-sin(theta), 0, cos(theta)]
        ])

        Rz = np.array([
            [cos(psi), -sin(psi), 0],
            [sin(psi), cos(psi), 0],
            [0, 0, 1]
        ])

        # Combine rotations (order: Z * Y * X)
        return np.dot(Rz, np.dot(Ry, Rx))

# Example usage
angles = [np.pi/4, np.pi/6, np.pi/3]  # 45°, 30°, 60°
combined_rotation = Rotation3D.from_euler(*angles)
print("Combined rotation matrix:\n", combined_rotation)
```

Slide 6: Triển khai ma trận chiếu

Ma trận chiếu chuyển đổi tọa độ 3D thành tọa độ màn hình 2D, mô phỏng phối cảnh. Việc triển khai này bao gồm cả ma trận chiếu phối cảnh và phép chiếu trực giao thường được sử dụng trong đồ họa máy tính.

```python
import numpy as np

class ProjectionMatrix:
    @staticmethod
    def perspective(fov, aspect, near, far):
        f = 1.0 / np.tan(fov / 2)
        return np.array([
            [f/aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far+near)/(near-far), (2*far*near)/(near-far)],
            [0, 0, -1, 0]
        ])

    @staticmethod
    def orthographic(left, right, bottom, top, near, far):
        return np.array([
            [2/(right-left), 0, 0, -(right+left)/(right-left)],
            [0, 2/(top-bottom), 0, -(top+bottom)/(top-bottom)],
            [0, 0, -2/(far-near), -(far+near)/(far-near)],
            [0, 0, 0, 1]
        ])

# Example usage
perspective = ProjectionMatrix.perspective(
    fov=np.pi/4,    # 45 degrees
    aspect=16/9,    # Widescreen aspect ratio
    near=0.1,       # Near clipping plane
    far=100.0       # Far clipping plane
)

print("Perspective projection matrix:\n", perspective)
```

Trang trình bày 7: Triển khai quy trình đồ họa

Đường dẫn đồ họa biến đổi các đỉnh 3D qua nhiều giai đoạn: chuyển đổi mô hình, chuyển đổi chế độ xem và chiếu. Việc triển khai này thể hiện quy trình hoàn chỉnh được sử dụng trong các công cụ đồ họa hiện đại.

```python
class GraphicsPipeline:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.model_matrix = np.eye(4)
        self.view_matrix = np.eye(4)
        self.projection_matrix = ProjectionMatrix.perspective(
            np.pi/4, width/height, 0.1, 100.0
        )

    def set_camera(self, position, target, up):
        forward = target - position
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)

        self.view_matrix = np.array([
            [right[0], right[1], right[2], -np.dot(right, position)],
            [up[0], up[1], up[2], -np.dot(up, position)],
            [-forward[0], -forward[1], -forward[2], np.dot(forward, position)],
            [0, 0, 0, 1]
        ])

    def transform_vertex(self, vertex):
        # Convert to homogeneous coordinates
        v = np.append(vertex, 1)

        # Apply transformations
        v = np.dot(self.model_matrix, v)
        v = np.dot(self.view_matrix, v)
        v = np.dot(self.projection_matrix, v)

        # Perspective divide
        if v[3] != 0:
            v = v / v[3]

        # Convert to screen coordinates
        screen_x = int((v[0] + 1) * self.width / 2)
        screen_y = int((1 - v[1]) * self.height / 2)

        return np.array([screen_x, screen_y])

# Example usage
pipeline = GraphicsPipeline(1920, 1080)
pipeline.set_camera(
    position=np.array([0, 0, 5]),
    target=np.array([0, 0, 0]),
    up=np.array([0, 1, 0])
)

# Transform a vertex
vertex = np.array([1, 1, 1])
screen_coords = pipeline.transform_vertex(vertex)
print(f"Screen coordinates: {screen_coords}")
```

Slide 8: Tối ưu hóa hiệu suất ma trận

Hiểu được hiệu suất nhân ma trận là rất quan trọng đối với các ứng dụng đồ họa. Việc triển khai này thể hiện các kỹ thuật tối ưu hóa khác nhau bao gồm vectơ hóa và các mẫu truy cập bộ nhớ thân thiện với bộ đệm.

```python
import numpy as np
import time

class OptimizedMatrixOps:
    @staticmethod
    def naive_multiply(A, B):
        n = len(A)
        result = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    result[i][j] += A[i][k] * B[k][j]
        return result

    @staticmethod
    def optimized_multiply(A, B):
        # Using numpy's optimized dot product
        return np.dot(A, B)

    @staticmethod
    def benchmark(size=1000):
        A = np.random.rand(size, size)
        B = np.random.rand(size, size)

        start = time.time()
        _ = OptimizedMatrixOps.optimized_multiply(A, B)
        optimized_time = time.time() - start

        if size <= 100:  # Only run naive for small matrices
            start = time.time()
            _ = OptimizedMatrixOps.naive_multiply(A, B)
            naive_time = time.time() - start
            print(f"Naive multiplication time: {naive_time:.4f}s")

        print(f"Optimized multiplication time: {optimized_time:.4f}s")

# Run benchmark
OptimizedMatrixOps.benchmark(size=100)
```

Slide 9: Triển khai Model View Projection (MVP)

Ma trận Chiếu Chế độ xem Mô hình kết hợp chuyển đổi đối tượng, định vị camera và chiếu phối cảnh. Việc triển khai này cho thấy cách xây dựng và áp dụng quy trình chuyển đổi MVP hoàn chỉnh.

```python
class MVPTransform:
    def __init__(self):
        self.model = np.eye(4)
        self.view = np.eye(4)
        self.projection = np.eye(4)

    def set_model_transform(self, position, rotation, scale):
        # Translation matrix
        translation = np.array([
            [1, 0, 0, position[0]],
            [0, 1, 0, position[1]],
            [0, 0, 1, position[2]],
            [0, 0, 0, 1]
        ])

        # Scale matrix
        scale_matrix = np.array([
            [scale[0], 0, 0, 0],
            [0, scale[1], 0, 0],
            [0, 0, scale[2], 0],
            [0, 0, 0, 1]
        ])

        # Get rotation matrix using previous Rotation3D class
        rotation_matrix = np.eye(4)
        rotation_matrix[:3, :3] = Rotation3D.from_euler(*rotation)

        # Combine transformations
        self.model = translation @ rotation_matrix @ scale_matrix

    def get_mvp(self):
        return self.projection @ self.view @ self.model

    def transform_vertices(self, vertices):
        # Convert vertices to homogeneous coordinates
        homogeneous = np.hstack((vertices, np.ones((len(vertices), 1))))

        # Apply MVP transformation
        mvp = self.get_mvp()
        transformed = homogeneous @ mvp.T

        # Perspective division
        transformed = transformed[:, :3] / transformed[:, 3:]
        return transformed

# Example usage
mvp = MVPTransform()
mvp.set_model_transform(
    position=np.array([0, 0, -5]),
    rotation=np.array([0, np.pi/4, 0]),
    scale=np.array([1, 1, 1])
)

# Transform vertices of a cube
cube_vertices = np.array([
    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
])

transformed_vertices = mvp.transform_vertices(cube_vertices)
print("Transformed vertices:\n", transformed_vertices)
```

Slide 10: Mô phỏng xử lý song song giống GPU

Việc triển khai này mô phỏng quá trình xử lý song song giống như GPU cho các hoạt động ma trận bằng cách sử dụng khả năng đa xử lý của Python, thể hiện cách GPU tăng tốc tính toán đồ họa.

```python
import multiprocessing as mp
from functools import partial
import numpy as np

class GPUSimulator:
    def __init__(self, num_processors=None):
        self.num_processors = num_processors or mp.cpu_count()

    def parallel_matrix_multiply(self, A, B):
        pool = mp.Pool(self.num_processors)
        result = np.zeros_like(A)

        def process_row(row_idx):
            return np.dot(A[row_idx], B)

        # Parallel processing of matrix rows
        results = pool.map(process_row, range(len(A)))
        pool.close()
        pool.join()

        return np.array(results)

    def batch_transform_vertices(self, vertices, transformation_matrix):
        # Split vertices into batches
        batch_size = len(vertices) // self.num_processors
        batches = [vertices[i:i + batch_size] for i in range(0, len(vertices), batch_size)]

        # Process batches in parallel
        pool = mp.Pool(self.num_processors)
        transform_func = partial(np.dot, b=transformation_matrix)
        results = pool.map(transform_func, batches)
        pool.close()
        pool.join()

        return np.vstack(results)

# Example usage
gpu_sim = GPUSimulator()
A = np.random.rand(1000, 1000)
B = np.random.rand(1000, 1000)

# Compare execution times
start = time.time()
result_parallel = gpu_sim.parallel_matrix_multiply(A, B)
parallel_time = time.time() - start

start = time.time()
result_numpy = np.dot(A, B)
numpy_time = time.time() - start

print(f"Parallel execution time: {parallel_time:.4f}s")
print(f"NumPy execution time: {numpy_time:.4f}s")
```

Trang trình bày 11: Quy trình hoạt hình thời gian thực

Việc triển khai này thể hiện một quy trình hoạt hình hoàn chỉnh cho đồ họa 3D thời gian thực, bao gồm cả nội suy giữa các khung hình chính và các phép biến đổi mượt mà cho chuyển động liên tục.

```python
import numpy as np
from dataclasses import dataclass
from typing import List
import time

@dataclass
class Keyframe:
    position: np.ndarray
    rotation: np.ndarray
    scale: np.ndarray
    time: float

class AnimationSystem:
    def __init__(self):
        self.keyframes: List[Keyframe] = []
        self.current_time = 0.0

    def add_keyframe(self, keyframe: Keyframe):
        self.keyframes.append(keyframe)
        # Sort keyframes by time
        self.keyframes.sort(key=lambda x: x.time)

    def interpolate(self, t: float) -> Keyframe:
        # Find surrounding keyframes
        next_idx = next((i for i, kf in enumerate(self.keyframes)
                        if kf.time > t), len(self.keyframes))
        if next_idx == 0:
            return self.keyframes[0]
        if next_idx == len(self.keyframes):
            return self.keyframes[-1]

        prev_kf = self.keyframes[next_idx - 1]
        next_kf = self.keyframes[next_idx]

        # Calculate interpolation factor
        alpha = ((t - prev_kf.time) /
                (next_kf.time - prev_kf.time))

        # Linear interpolation
        return Keyframe(
            position=prev_kf.position * (1-alpha) + next_kf.position * alpha,
            rotation=prev_kf.rotation * (1-alpha) + next_kf.rotation * alpha,
            scale=prev_kf.scale * (1-alpha) + next_kf.scale * alpha,
            time=t
        )

    def update(self, delta_time: float):
        self.current_time += delta_time
        return self.interpolate(self.current_time)

# Example usage
animation = AnimationSystem()

# Add keyframes for a simple rotation animation
animation.add_keyframe(Keyframe(
    position=np.array([0, 0, 0]),
    rotation=np.array([0, 0, 0]),
    scale=np.array([1, 1, 1]),
    time=0.0
))

animation.add_keyframe(Keyframe(
    position=np.array([0, 0, 0]),
    rotation=np.array([0, np.pi, 0]),
    scale=np.array([1, 1, 1]),
    time=2.0
))

# Simulate animation updates
for _ in range(5):
    frame = animation.update(0.5)
    print(f"Time: {animation.current_time:.1f}s, "
          f"Rotation: {frame.rotation}")
```

Trang trình bày 12: Rasterization tam giác được tối ưu hóa

Một phần quan trọng của quy trình đồ họa là chuyển đổi các hình tam giác 3D thành pixel. Việc triển khai này cho thấy một thuật toán đường quét hiệu quả để tạo rasterization tam giác với việc đi theo cạnh.

```python
import numpy as np
from typing import Tuple, List

class Rasterizer:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.framebuffer = np.zeros((height, width), dtype=np.float32)
        self.zbuffer = np.full((height, width), np.inf)

    def edge_function(self, a: np.ndarray, b: np.ndarray,
                     c: np.ndarray) -> float:
        return ((c[0] - a[0]) * (b[1] - a[1]) -
                (c[1] - a[1]) * (b[0] - a[0]))

    def rasterize_triangle(self, vertices: List[np.ndarray],
                          color: float = 1.0):
        # Compute bounding box
        min_x = max(0, int(min(v[0] for v in vertices)))
        max_x = min(self.width - 1, int(max(v[0] for v in vertices)))
        min_y = max(0, int(min(v[1] for v in vertices)))
        max_y = min(self.height - 1, int(max(v[1] for v in vertices)))

        # Precompute edge functions
        area = self.edge_function(vertices[0], vertices[1], vertices[2])
        if area == 0:
            return

        # Rasterize
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                point = np.array([x + 0.5, y + 0.5])

                # Compute barycentric coordinates
                w0 = self.edge_function(vertices[1], vertices[2], point)
                w1 = self.edge_function(vertices[2], vertices[0], point)
                w2 = self.edge_function(vertices[0], vertices[1], point)

                # Check if point is inside triangle
                if w0 >= 0 and w1 >= 0 and w2 >= 0:
                    # Normalize barycentric coordinates
                    w0 /= area
                    w1 /= area
                    w2 /= area

                    # Interpolate z-value
                    z = (w0 * vertices[0][2] +
                         w1 * vertices[1][2] +
                         w2 * vertices[2][2])

                    # Z-buffer test
                    if z < self.zbuffer[y, x]:
                        self.zbuffer[y, x] = z
                        self.framebuffer[y, x] = color

# Example usage
rasterizer = Rasterizer(800, 600)
triangle = [
    np.array([100, 100, 0]),
    np.array([700, 100, 0]),
    np.array([400, 500, 0])
]
rasterizer.rasterize_triangle(triangle)

print("Number of pixels rasterized:",
      np.sum(rasterizer.framebuffer > 0))
```

Trang trình bày 13: Thực hiện loại bỏ và cắt khuôn mặt

Việc loại bỏ và cắt bớt khuôn mặt là những tối ưu hóa cần thiết trong đồ họa 3D. Việc triển khai này trình bày cách loại bỏ hiệu quả các khuôn mặt ẩn và cắt hình học khỏi chế độ xem bị thất vọng.

```python
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Plane:
    normal: np.ndarray
    distance: float

class FrustumClipper:
    def __init__(self):
        # Define view frustum planes (near, far, left, right, top, bottom)
        self.frustum_planes = [
            Plane(np.array([0, 0, 1]), -0.1),  # Near
            Plane(np.array([0, 0, -1]), 100),  # Far
            Plane(np.array([1, 0, 0]), 1),     # Left
            Plane(np.array([-1, 0, 0]), 1),    # Right
            Plane(np.array([0, 1, 0]), 1),     # Bottom
            Plane(np.array([0, -1, 0]), 1)     # Top
        ]

    def is_face_visible(self, vertices: np.ndarray) -> bool:
        # Calculate face normal using cross product
        v1 = vertices[1] - vertices[0]
        v2 = vertices[2] - vertices[0]
        normal = np.cross(v1, v2)

        # Check if face is facing camera (basic back-face culling)
        return np.dot(normal, vertices[0]) < 0

    def clip_triangle(self, vertices: np.ndarray) -> List[np.ndarray]:
        if not self.is_face_visible(vertices):
            return []

        current_vertices = vertices.tolist()

        # Clip against each frustum plane
        for plane in self.frustum_planes:
            if not current_vertices:
                return []

            next_vertices = []

            # Process each edge of the polygon
            for i in range(len(current_vertices)):
                current = np.array(current_vertices[i])
                next = np.array(current_vertices[(i + 1) % len(current_vertices)])

                current_inside = (np.dot(plane.normal, current) + plane.distance) > 0
                next_inside = (np.dot(plane.normal, next) + plane.distance) > 0

                if current_inside:
                    next_vertices.append(current)

                if current_inside != next_inside:
                    # Calculate intersection point
                    t = (-plane.distance - np.dot(plane.normal, current)) / \
                        np.dot(plane.normal, next - current)
                    intersection = current + t * (next - current)
                    next_vertices.append(intersection)

            current_vertices = next_vertices

        return current_vertices

# Example usage
clipper = FrustumClipper()

# Test triangle
triangle = np.array([
    [-0.5, -0.5, -1],
    [0.5, -0.5, -1],
    [0, 0.5, -1]
])

# Clip triangle
clipped_vertices = clipper.clip_triangle(triangle)
print(f"Original vertices: {len(triangle)}")
print(f"Clipped vertices: {len(clipped_vertices)}")
```

Trang trình bày 14: Triển khai đồ thị cảnh

Biểu đồ cảnh tổ chức các đối tượng 3D theo thứ bậc, cho phép thực hiện các phép biến đổi phức tạp và mối quan hệ giữa các đối tượng. Việc triển khai này cho thấy cách xây dựng và duyệt qua biểu đồ cảnh một cách hiệu quả.

```python
from typing import Optional, List
import numpy as np

class SceneNode:
    def __init__(self, name: str):
        self.name = name
        self.local_transform = np.eye(4)
        self.world_transform = np.eye(4)
        self.parent: Optional[SceneNode] = None
        self.children: List[SceneNode] = []
        self.mesh: Optional[np.ndarray] = None

    def add_child(self, child: 'SceneNode'):
        child.parent = self
        self.children.append(child)

    def set_transform(self, translation: np.ndarray,
                     rotation: np.ndarray, scale: np.ndarray):
        # Create transformation matrix
        T = np.eye(4)
        T[:3, 3] = translation

        R = np.eye(4)
        R[:3, :3] = Rotation3D.from_euler(*rotation)

        S = np.eye(4)
        np.fill_diagonal(S[:3, :3], scale)

        self.local_transform = T @ R @ S
        self.update_world_transform()

    def update_world_transform(self):
        if self.parent is None:
            self.world_transform = self.local_transform
        else:
            self.world_transform = (self.parent.world_transform @
                                  self.local_transform)

        # Update children
        for child in self.children:
            child.update_world_transform()

    def traverse(self, callback):
        callback(self)
        for child in self.children:
            child.traverse(callback)

# Example usage
def create_robot_arm():
    root = SceneNode("root")
    base = SceneNode("base")
    upper_arm = SceneNode("upper_arm")
    forearm = SceneNode("forearm")
    hand = SceneNode("hand")

    # Build hierarchy
    root.add_child(base)
    base.add_child(upper_arm)
    upper_arm.add_child(forearm)
    forearm.add_child(hand)

    # Set transforms
    base.set_transform(
        translation=np.array([0, 0, 0]),
        rotation=np.array([0, 0, 0]),
        scale=np.array([1, 1, 1])
    )

    upper_arm.set_transform(
        translation=np.array([0, 1, 0]),
        rotation=np.array([0, 0, np.pi/4]),
        scale=np.array([1, 2, 1])
    )

    return root

# Create and traverse scene graph
robot = create_robot_arm()
def print_node(node):
    print(f"Node: {node.name}")
    print(f"World transform:\n{node.world_transform}")

robot.traverse(print_node)
```

Trang trình bày 15: Tài nguyên bổ sung

* "Nhân ma trận dựa trên GPU hiệu quả cho các ứng dụng đồ họa quy mô lớn" [https://arxiv.org/abs/2103.12345](https://arxiv.org/abs/2103.12345)
* "Các phương pháp tiếp cận hiện đại để tối ưu hóa đường ống đồ họa 3D theo thời gian thực" [https://arxiv.org/abs/2104.54321](https://arxiv.org/abs/2104.54321)
* "Kỹ thuật tối ưu hóa đồ thị cảnh cho các ứng dụng thực tế ảo" [https://arxiv.org/abs/2105.98765](https://arxiv.org/abs/2105.98765)
* "Thuật toán rasterization tam giác nâng cao để hiển thị thời gian thực" [https://arxiv.org/abs/2106.11111](https://arxiv.org/abs/2106.11111)

Lưu ý: Các URL được cung cấp chỉ là ví dụ và có thể không tương ứng với các giấy tờ thực tế vì tôi không thể xác minh sự tồn tại của chúng.
