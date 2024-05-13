import numpy as np

def fourSum(nums: list[int], target: int) -> list[list[int]]:
    """
    Finds all unique quadruplets that sum to the target in an array.

    Args:
        nums: A list of integers.
        target: The target sum.

    Returns:
        A list of all unique quadruplets that sum to the target.
    """
    results = []
    seen = {}  # Hash map to store seen pairs and their sum

    # Iterate through the array
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            current_sum = nums[i] + nums[j]
            complement = target - current_sum
            
            # Check if the complement (remaining sum) exists in the hash map
            if complement in seen:
                # If complement exists, iterate through its corresponding pairs
                for pair in seen[complement]:
                    # Check if the current pair and the found pair don't share indexes
                    if i > pair[0] and j > pair[1]:
                        results.append([nums[i], nums[j], pair[0], pair[1]])

            # Add the current pair (nums[i], nums[j]) and its sum to the hash map
            seen.setdefault(current_sum, []).append([i, j])

    return results



from collections import Counter
import heapq

def reorganizeString(s: str) -> str:
    count = Counter(s)

    maxHeap = [[-cnt, char]for char, cnt in count.items()]
    heapq.heapify(maxHeap)

    prev = None
    res = ""
    while maxHeap or prev:
        if prev and not maxHeap:
            return ""
        cnt, char = heapq.heappop(maxHeap)
        res += char
        cnt += 1
        if prev:
            heapq.heappush(maxHeap, prev)
            prev = None

        if cnt != 0:
            prev = [cnt, char]
    
    return res


def merge_intervals(intervals: list[list[int]]) -> list[list[int]]:
    """
    Given an array of intervals where intervals[i] = [starti, endi], merge all
    overlapping intervals, and return an array of the non-overlapping intervals
    that cover all the intervals in the input.
    """
    merged_intervals = []

    for interval in intervals:
        if not merged_intervals or interval[0] > merged_intervals[-1][1]:
            merged_intervals.append(interval)
        else:
            merged_intervals[-1][1] = max(merged_intervals[-1][1], interval[1])
    
    return merged_intervals


def meeting_rooms(intervals: list[list[int]]) -> int:
    """
    Given an array of meeting time intervals intervals where intervals[i] =
    [starti, endi], return the minimum number of conference rooms required.
    """
    if not intervals:
        return 0
    
    # Separate start and end times and sort them
    start_times = sorted([interval[0] for interval in intervals])
    end_times = sorted([interval[1] for interval in intervals])
    
    rooms_needed = 0
    end_ptr = 0
    
    # Iterate over start times
    for start_time in start_times:
        if start_time < end_times[end_ptr]:
            rooms_needed += 1
        else:
            end_ptr += 1
    
    return rooms_needed


def matrix_multiplication(mat1: list[list[int]], mat2: list[list[int]]) -> list[list[int]]:
    """
    Given two sparse matrices mat1 of size m x k and mat2 of size k x n,
    return the result of mat1 x mat2. You may assume that multiplication is
    always possible.
    """
    m, k = len(mat1), len(mat1[0])
    n = len(mat2[0])

    # Initialize result matrix with zeros
    result = [[0] * n for _ in range(m)]

    # Perform matrix multiplication
    for i in range(m):
        for j in range(n):
            for p in range(k):
                result[i][j] += mat1[i][p] * mat2[p][j]

    return result


def is_substring(s: str, t: str) -> bool:
    """
    Given two strings s and t, return true if s is a subsequence of t, or false otherwise.

    A subsequence of a string is a new string that is formed from the original
    string by deleting some (can be none) of the characters without disturbing the
    relative positions of the remaining characters. (i.e., "ace" is a subsequence of
    "abcde" while "aec" is not).
    """
    if len(s) > len(t):
        return False

    s_pointer = 0
    t_pointer = 0

    while s_pointer < len(s) or t_pointer < len(t):
        if s[s_pointer] == t[t_pointer]:
            s_pointer += 1
        t_pointer += 1
    
    return s_pointer == len(s)


def unique_paths(obstacle_grid: list[list[int]]) -> int:
    """
    You are given an m x n integer array grid. There is a robot initially
    located at the top-left corner (i.e., grid[0][0]). The robot tries to move
    to the bottom-right corner (i.e., grid[m - 1][n - 1]). The robot can only
    move either down or right at any point in time.

    An obstacle and space are marked as 1 or 0 respectively in grid. A path that the
    robot takes cannot include any square that is an obstacle.

    Return the number of possible unique paths that the robot can take to reach the
    bottom-right corner.
    """
    rows, cols = len(obstacle_grid), len(obstacle_grid[0])
    dp = [0] * cols
    dp[-1] = 1

    # O(m*n)
    for m in reversed(range(rows)):
        for n in reversed(range(cols)):
            if obstacle_grid[m][n]:
                dp[n] = 0
            elif n + 1 < cols:
                dp[n] = dp[n] + dp[n + 1]
    
    return dp[0]
