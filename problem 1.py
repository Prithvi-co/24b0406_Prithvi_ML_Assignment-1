import numpy as np

def main():
    # Generate a 2D NumPy array of shape (5, 4) with random integers from 1 to 50
    np.random.seed(0)  # for reproducibility
    array = np.random.randint(1, 51, size=(5, 4))
    print("Original Array:\n", array)

    # Extract and print anti-diagonal elements (top-right to bottom-left)
    anti_diagonal = [array[i, -i-1] for i in range(min(array.shape))]
    print("Anti-Diagonal Elements:", anti_diagonal)

    # Compute and print the maximum value in each row
    row_max = np.max(array, axis=1)
    print("Maximum value in each row:", row_max)

    # Compute mean and create array with elements <= mean
    mean_val = np.mean(array)
    filtered_array = array[array <= mean_val]
    print("Elements less than or equal to mean (mean = {:.2f}):".format(mean_val), filtered_array)


def numpy_boundary_traversal(matrix):
    """
    Returns the boundary elements of a 2D NumPy array in clockwise order,
    starting from the top-left corner.

    Parameters:
        matrix (np.ndarray): 2D NumPy array.

    Returns:
        list: Boundary elements in clockwise order.
    """
    if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
        raise ValueError("Input must be a 2D NumPy array")

    rows, cols = matrix.shape
    result = []

    # Top row
    for col in range(cols):
        result.append(matrix[0, col])

    # Right column
    for row in range(1, rows):
        result.append(matrix[row, cols - 1])

    # Bottom row
    if rows > 1:
        for col in range(cols - 2, -1, -1):
            result.append(matrix[rows - 1, col])

    # Left column
    if cols > 1:
        for row in range(rows - 2, 0, -1):
            result.append(matrix[row, 0])

    return result


if __name__ == "__main__":
    main()
    print("\nBoundary Traversal of the Array:")
    test_matrix = np.random.randint(1, 51, size=(5, 4))  # Example for testing
    print(test_matrix)
    boundary_elements = numpy_boundary_traversal(test_matrix)
    print("Boundary Elements:", boundary_elements)

