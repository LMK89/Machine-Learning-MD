## Bộ giải AI quét mìn bằng Python với Matplotlib

Slide 1: Giới thiệu về Minesweeper AI

Minesweeper là một trò chơi cổ điển liên quan đến việc khám phá một mạng lưới các ô trong khi tránh các mỏ ẩn. Tạo AI để giải quyết Minesweeper có thể là một dự án thú vị kết hợp logic, xác suất và lập trình Python. Trong trình chiếu này, chúng ta sẽ khám phá quá trình phát triển AI quét mìn bằng Matplotlib và Python.

```python
import matplotlib.pyplot as plt
import numpy as np
```

Slide 2: Representing the Game Board

To create a Minesweeper AI, we need to represent the game board in a data structure. We can use a 2D NumPy array to store the state of each tile, with values representing mines, uncovered tiles, or the number of surrounding mines.

```python
board = np.full((10, 10), -1)  # Initialize a 10x10 board with -1 (covered)
board[3, 4] = -2  # Set a mine at (3, 4)
```

Slide 3: Trực quan hóa bảng trò chơi

Chúng ta có thể sử dụng Matplotlib để trực quan hóa bảng trò chơi và cung cấp giao diện người dùng để AI tương tác với trò chơi. Chúng ta sẽ xác định các hàm để tạo hình ảnh trực quan của bảng và cập nhật nó khi các ô được mở ra.

```python
def visualize_board(board):
    plt.matshow(board, cmap='Blues')
    plt.xticks([])
    plt.yticks([])
    plt.show()
```

Slide 4: Uncovering Tiles

The AI needs to be able to uncover tiles on the board. We'll create a function that takes the board and a tile coordinate as input and updates the board according to the game rules.

```python
def uncover_tile(board, row, col):
    if board[row, col] == -1:
        # Uncover the tile
        board[row, col] = count_surrounding_mines(board, row, col)
        # If the tile has no surrounding mines, uncover neighbors
        if board[row, col] == 0:
            uncover_neighbors(board, row, col)
```

Slide 5: Đếm mỏ xung quanh

Để xác định số lượng mỏ xung quanh một ô, chúng ta sẽ tạo một hàm kiểm tra các ô lân cận và đếm số mỏ.

```python
def count_surrounding_mines(board, row, col):
    count = 0
    for i in range(-1, 2):
        for j in range(-1, 2):
            if 0 <= row + i < board.shape[0] and 0 <= col + j < board.shape[1]:
                if board[row + i, col + j] == -2:
                    count += 1
    return count
```

Slide 6: Uncovering Neighbors

If a tile has no surrounding mines, we need to recursively uncover its neighbors. We'll create a function that takes the board and a tile coordinate as input and recursively uncovers neighboring tiles until it encounters tiles with surrounding mines.

```python
def uncover_neighbors(board, row, col):
    for i in range(-1, 2):
        for j in range(-1, 2):
            if 0 <= row + i < board.shape[0] and 0 <= col + j < board.shape[1]:
                if board[row + i, col + j] == -1:
                    uncover_tile(board, row + i, col + j)
```

Trang trình bày 7: Vòng lặp trò chơi

Để chơi trò chơi, chúng tôi sẽ tạo một vòng lặp trò chơi cho phép AI thực hiện các nước đi và cập nhật bảng cho phù hợp. Vòng lặp này sẽ tiếp tục cho đến khi tất cả các ô không phải của mỏ được phát hiện hoặc trúng phải một quả mìn.

```python
game_over = False
while not game_over:
    # AI logic to choose a tile
    row, col = ai_choose_tile(board)
    uncover_tile(board, row, col)
    visualize_board(board)
    if board[row, col] == -2:
        game_over = True
        print("Game Over! You hit a mine.")
    elif np.all(board != -1):
        game_over = True
        print("Congratulations! You won the game.")
```

Slide 8: AI Logic: Simple Algorithm

For a simple AI algorithm, we can start by selecting a random uncovered tile on the board. This approach may not be optimal, but it serves as a starting point for more advanced strategies.

```python
def ai_choose_tile(board):
    uncovered = np.argwhere(board == -1)
    if len(uncovered) > 0:
        row, col = uncovered[np.random.randint(len(uncovered))]
        return row, col
    else:
        return None, None
```

Slide 9: AI Logic: Thuật toán dựa trên xác suất

Để cải thiện hiệu suất của AI, chúng ta có thể sử dụng xác suất và kiến ​​thức về số lượng mìn xung quanh để đưa ra quyết định sáng suốt hơn. AI có thể ưu tiên phát hiện các ô có xác suất chứa mìn thấp hơn.

```python
def ai_choose_tile(board):
    probabilities = calculate_probabilities(board)
    min_prob = np.min(probabilities[probabilities != -1])
    coords = np.argwhere(probabilities == min_prob)
    row, col = coords[np.random.randint(len(coords))]
    return row, col
```

Slide 10: Calculating Probabilities

To calculate the probability of a tile containing a mine, we'll create a function that uses the surrounding mine counts and the remaining uncovered tiles to estimate the likelihood of each tile being a mine.

```python
def calculate_probabilities(board):
    probabilities = np.full(board.shape, -1)
    uncovered = np.argwhere(board == -1)
    for row, col in uncovered:
        neighbors = get_neighbors(board, row, col)
        known_mines = sum(board[neighbors] == -2)
        unknown = len(neighbors) - known_mines - sum(board[neighbors] >= 0)
        if unknown > 0:
            probabilities[row, col] = known_mines / unknown
    return probabilities
```

Trang trình bày 11: Làm quen với hàng xóm

Để tính xác suất, chúng ta cần một hàm trả về tọa độ của các ô lân cận cho một ô nhất định.

```python
def get_neighbors(board, row, col):
    neighbors = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            if 0 <= row + i < board.shape[0] and 0 <= col + j < board.shape[1]:
                if i != 0 or j != 0:
                    neighbors.append((row + i, col + j))
    return neighbors
```

Slide 12: Advanced Strategies

The probability-based algorithm can be further improved by incorporating more advanced strategies, such as constraint propagation, pattern recognition, and machine learning techniques. These approaches can lead to more efficient and robust AI solutions for solving Minesweeper.

```python
# Constraint propagation algorithm
def constraint_propagation(board):
    # Implementation details...

# Pattern recognition algorithm
def pattern_recognition(board):
    # Implementation details...

# Machine learning approach
def train_ml_model(data):
    # Implementation details...
```

Slide 13: Tuyên truyền ràng buộc

Tuyên truyền ràng buộc là một chiến lược nâng cao có thể cải thiện hơn nữa hiệu suất của AI quét mìn. Nó liên quan đến việc sử dụng thông tin đã biết trên bảng để suy ra trạng thái của các ô khác, giảm không gian tìm kiếm một cách hiệu quả và tăng cơ hội thực hiện các nước đi tối ưu.

```python
def constraint_propagation(board):
    changed = True
    while changed:
        changed = False
        for row in range(board.shape[0]):
            for col in range(board.shape[1]):
                if board[row, col] >= 0:
                    neighbors = get_neighbors(board, row, col)
                    uncovered = [n for n in neighbors if board[n] == -1]
                    mines = [n for n in neighbors if board[n] == -2]
                    if len(mines) == board[row, col]:
                        for n in uncovered:
                            if board[n] == -1:
                                board[n] = -2
                                changed = True
                    elif len(uncovered) == board[row, col] - len(mines):
                        for n in uncovered:
                            if board[n] == -1:
                                uncover_tile(board, n[0], n[1])
                                changed = True
```

Slide 14: Pattern Recognition

Pattern recognition is another advanced strategy that can be employed in Minesweeper AI. It involves identifying known patterns on the board and using them to deduce the state of other tiles. This can be particularly effective in certain scenarios where constraint propagation may not be sufficient.

```python
def pattern_recognition(board):
    patterns = [
        # 1-1 pattern
        np.array([[0, 1, -1],
                  [1, -1, -1],
                  [-1, -1, -1]]),
        # 2-2 pattern
        np.array([[-1, -1, 1],
                  [1, 2, 1],
                  [-1, -1, 1]]),
        # ... Add more patterns as needed
    ]

    for pattern in patterns:
        for row in range(board.shape[0] - pattern.shape[0] + 1):
            for col in range(board.shape[1] - pattern.shape[1] + 1):
                sub_board = board[row:row + pattern.shape[0], col:col + pattern.shape[1]]
                if np.all(sub_board[pattern >= 0] == pattern[pattern >= 0]):
                    for i in range(pattern.shape[0]):
                        for j in range(pattern.shape[1]):
                            if pattern[i, j] == -1 and sub_board[i, j] == -1:
                                uncover_tile(board, row + i, col + j)
```

Trang trình bày 15: Phương pháp học máy

Kỹ thuật học máy cũng có thể được áp dụng cho Minesweeper AI, đặc biệt để học các chiến lược tối ưu từ bộ dữ liệu lớn về bảng trò chơi và nước đi. Cách tiếp cận này có khả năng dẫn đến các giải pháp AI tinh vi và hiệu quả hơn.

```python
import tensorflow as tf

def train_ml_model(data):
    # Preprocess data
    X = [preprocess_board(board) for board, _, _ in data]
    y = [next_move for _, next_move, _ in data]

    # Define the model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X[0].shape,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    # Compile and train the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(np.array(X), np.array(y), epochs=10, batch_size=32)

    return model
```

Phần này kết thúc trình chiếu về việc phát triển AI quét mìn bằng Matplotlib và Python. Chúng tôi đã đề cập đến nhiều khía cạnh khác nhau, bao gồm biểu diễn bảng, trực quan hóa, logic trò chơi, thuật toán đơn giản, chiến lược nâng cao như lan truyền ràng buộc và nhận dạng mẫu và thậm chí cả phương pháp học máy. Mỗi slide cung cấp các ví dụ về mã để minh họa các khái niệm và thuật toán được thảo luận.
