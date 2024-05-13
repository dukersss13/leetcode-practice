from solutions import *

def test_fourSum():
    # Four Sum Problem
    nums = [-3, -1, 0, 2, 4, 5]
    target = 2
    results = fourSum(nums, target)

    assert results == [[-1, 4, 0, 3], [0, 5, 0, 2], [2, 4, 0, 1]]


def test_reorganize_string():
    # Reorganize String
    string = "aab"
    results = reorganizeString(string)

    assert results == "aba"


def test_merge_intervals():
    # Merge intervals
    intervals = [[1, 3], [2, 6], [8, 10], [15, 18]]
    results = merge_intervals(intervals)

    assert results == [[1, 6], [8, 10], [15, 18]]


def test_meeting_rooms():
    # Meeting Rooms
    intervals = [[8, 11], [9, 10], [12, 14]]
    results = meeting_rooms(intervals)

    assert results == 2


def test_matrix_mul():
    # Matrix Multiplication
    mat1 = [[1, 0, 0], [-1, 0, 3]]
    mat2 = [[7, 0, 0], [0, 0, 0], [0, 0, 1]]
    results = matrix_multiplication(mat1, mat2)

    assert results == [[7, 0, 0], [-7, 0, 3]]


def test_is_substring():
    s = "abe"
    t = "abcde"
    results = is_substring(s, t)

    assert results


def test_unique_paths():
    obstacle_grid = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    results = unique_paths(obstacle_grid)

    assert results == 2