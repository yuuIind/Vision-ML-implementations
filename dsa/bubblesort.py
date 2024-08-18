""" Bubble Sort

A Python implementation of Bubble Sort

"""
from typing import List

def bubble_sort(arr: List[int], inplace: bool=False) -> List[int]:
    """
    Sorts an array using the Bubble Sort algorithm.

    Parameters:
    arr (List[int]): The list of integers to sort.
    inplace (bool): If True, sorts the array in place. If False, returns a new sorted array.

    Returns:
    List[int]: The sorted list of integers.
    """
    if not inplace:
        arr = arr.copy() # Do not modify the original array if inplace=False
    
    arr_len = len(arr)
    for i in range(arr_len-1):
        for j in range(arr_len-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j] # Swap the elements
    return arr


def run_tests():
    """
    Runs a series of test cases to validate the bubble_sort function.
    """
    test_cases = [
        ([0, 4, 2, 8, 9, 6, 5], False),
        ([1, 2, 3, 4, 5], True),
        ([5, 4, 3, 2, 1], False),
        ([10, -1, 2, 8, 3], False),
        ([], False),
        ([7, 3], True)
    ]

    for i, (arr, inplace) in enumerate(test_cases, 1):
        result = bubble_sort(arr, inplace=inplace)
        print(f"Test case {i}: {'In-place' if inplace else 'New list'}")
        print(f"Original list: {arr}")
        print(f"Sorted list:   {result}")
        print("-" * 40)


if __name__ == "__main__":
    print("Running Bubble Sort Tests...\n")
    run_tests()