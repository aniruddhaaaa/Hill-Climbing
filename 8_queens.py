def is_safe(board, row, col, n):
    # Check if there is a queen in the same column
    for i in range(row):
        if board[i][col] == 1:
            return False

    # Check upper-left diagonal
    for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
        if board[i][j] == 1:
            return False

    # Check upper-right diagonal
    for i, j in zip(range(row, -1, -1), range(col, n)):
        if board[i][j] == 1:
            return False

    return True

def solve_n_queens_util(board, row, n, solutions):
    if row == n:
        solutions.append(["".join(["Q" if x == 1 else "." for x in row]) for row in board])
        return

    for col in range(n):
        if is_safe(board, row, col, n):
            board[row][col] = 1
            solve_n_queens_util(board, row + 1, n, solutions)
            board[row][col] = 0

def solve_n_queens(n):
    board = [[0 for _ in range(n)] for _ in range(n)]
    solutions = []
    solve_n_queens_util(board, 0, n, solutions)
    return solutions

if __name__ == "__main__":
    n = 9
    solutions = solve_n_queens(n)
    print(f"Total solutions for {n}-Queens problem: {len(solutions)}\n")
    for i, solution in enumerate(solutions, 1):
        print(f"Solution {i}:")
        for row in solution:
            print(row)
        print("\n")
