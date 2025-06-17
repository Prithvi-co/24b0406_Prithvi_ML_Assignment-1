 import numpy as np

def numpy_alternate_sort(array):
    """
    Sorts a 1D NumPy array in alternating pattern:
    smallest, largest, second smallest, second largest, etc.

    Parameters:
        array (np.ndarray): 1D NumPy array of numbers

    Returns:
        np.ndarray: Alternating sorted array
    """
    if not isinstance(array, np.ndarray) or array.ndim != 1:
        raise ValueError("Input must be a 1D NumPy array")

    sorted_arr = np.sort(array)
    result = []

    i, j = 0, len(sorted_arr) - 1
    while i <= j:
        result.append(sorted_arr[i])
        if i != j:
            result.append(sorted_arr[j])
        i += 1
        j -= 1

    return np.array(result)


def main():
    # 1. Create 1D NumPy array of 20 random floats between 0 and 10
    np.random.seed(42)  # For reproducibility
    arr = np.random.uniform(0, 10, 20)

    # 2. Round all elements to 2 decimal places and print
    arr_rounded = np.round(arr, 2)
    print("Original Array (rounded):\n", arr_rounded)

    # 3. Calculate and print min, max, median
    print("Minimum:", np.min(arr_rounded))
    print("Maximum:", np.max(arr_rounded))
    print("Median:", np.median(arr_rounded))

    # 4. Replace all elements < 5 with their squares
    arr_modified = np.where(arr_rounded < 5, np.round(arr_rounded**2, 2), arr_rounded)
    print("Modified Array (squared elements < 5):\n", arr_modified)

    # 5. Apply numpy_alternate_sort and print result
    alt_sorted = numpy_alternate_sort(arr_rounded)
    print("Alternating Sorted Array:\n", alt_sorted)


if __name__ == "__main__":
    main()
