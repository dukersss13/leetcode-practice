from typing import List
import numpy as np


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


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


from collections import Counter, defaultdict, deque
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


def string_compression(chars: list[str]) -> int:
    """
    Given an array of characters chars, compress it using the following
    algorithm:

    Begin with an empty string s. For each group of consecutive repeating characters
    in chars:

    If the group's length is 1, append the character to s. Otherwise, append the
    character followed by the group's length. The compressed string s should not be
    returned separately, but instead, be stored in the input character array chars.
    Note that group lengths that are 10 or longer will be split into multiple
    characters in chars.

    After you are done modifying the input array, return the new length of the
    array.

    You must write an algorithm that uses only constant extra space.

    Example: 
    Input: chars = ["a","a","b","b","c","c","c"]
    Output: Return: ["a","2","b","2","c","3"] so length of this is 6
    Explanation: The groups are "aa", "bb", and "ccc". This compresses to "a2b2c3".
    """
    read_index, counter = 0, 0

    while read_index < len(chars):
        start_of_sequence = read_index
        # Have this read < len(chars) again in case chars contains all of the same string
        # i.e. contains all a's
        while read_index < len(chars) and chars[read_index] == chars[start_of_sequence]:
            read_index += 1

        chars[counter] = chars[start_of_sequence]
        counter += 1

        if read_index - start_of_sequence > 1:
            for digit in str(read_index - start_of_sequence):
                chars[counter] = digit
                counter += 1
    
    return counter


def reconstruct_itinerary(tickets: list[list[str]]) -> list[str]:
    """
    You are given a list of airline tickets where tickets[i] = [fromi, toi]
    represent the departure and the arrival airports of one flight. Reconstruct
    the itinerary in order and return it.

    All of the tickets belong to a man who departs from "JFK", thus, the itinerary
    must begin with "JFK". If there are multiple valid itineraries, you should
    return the itinerary that has the smallest lexical order when read as a single
    string.

    For example, the itinerary ["JFK", "LGA"] has a smaller lexical order than
    ["JFK", "LGB"]. You may assume all tickets form at least one valid itinerary.
    You must use all the tickets once and only once.

    Time Complexity: O(N logN)
    Space Complexity: O(N)
    """
    graph = defaultdict(list)
    for src, dest in sorted(tickets, reverse=True):
        graph[src].append(dest)
    
    itinerary = []

    def dfs(src: str):
        while graph[src]:
            next_destination = graph[src].pop()
            dfs(next_destination)

        itinerary.append(src)
    
    dfs("JFK")

    return itinerary[::-1]


def searchInsert(nums: list[int], target: int) -> list:
    """
    Given a sorted array of distinct integers and a target value, return the
    index if the target is found. If not, return the index where it would be if
    it were inserted in order.

    You must write an algorithm with O(log n) runtime complexity.
    """
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    # If the target is not found, left will be the position where the target should be inserted
    return left


def minTransfers(transactions: list[list[int]]) -> int:
    """
    You are given an array of transactions transactions where transactions[i] =
    [from_i, to_i, amount_i] indicates that the person with ID = from_i gave amount_i
    $ to the person with ID = to_i.

    Return the minimum number of transactions required to settle the debt.
    """
    graph = defaultdict(int)

    for f, t, a in transactions:
        graph[f] -= a
        graph[t] += a

    positives = [val for val in graph.values() if val > 0]
    negatives = [val for val in graph.values() if val < 0]

    def recurse(positives: list[int], negatives: list[int]):
        if len(positives) + len(negatives) == 0:
            return 0

        count = 0
        negative = negatives[0]
        for positive in positives:
            new_positives = positives.copy()
            new_negatives = negatives.copy()

            positives.remove(positive)
            negatives.remove(negative)

            balance = positive + negative

            if balance == 0:
                continue
            elif balance > 0:
                new_positives.append(balance)
            else:
                new_negatives.append(balance)
        
            count = min(count, recurse(new_positives, new_negatives))

        return count + 1
    
    return recurse(positives, negatives)


def moveZeros(nums: list[int]) -> list[int]:
    """
    Given an integer array nums, move all 0's to the end of it while maintaining
    the relative order of the non-zero elements.

    Note that you must do this in-place without making a copy of the array.
    """
    slow_ptr = 0

    for fast_ptr in range(len(nums)):
        if nums[fast_ptr] != 0:
            nums[fast_ptr], nums[slow_ptr] = nums[slow_ptr], nums[fast_ptr]
            slow_ptr += 1

    return nums 


def least_intervals(tasks: list[str], n: int) -> int:
    """
    You are given an array of CPU tasks, each represented by letters A to Z, and a
    cooling time, n. Each cycle or interval allows the completion of one task. Tasks
    can be completed in any order, but there's a constraint: identical tasks must be
    separated by at least n intervals due to cooling time.

    ​Return the minimum number of intervals required to complete all tasks.
    """
    # Count the frequency of each task
    task_counts = Counter(tasks)

    # Get the maximum frequency
    max_freq = max(task_counts.values())

    # Count the number of tasks with maximum frequency
    max_freq_tasks_count = sum(1 for count in task_counts.values() if count == max_freq)

    # Calculate the minimum intervals required based on the tasks with maximum frequency
    min_intervals = (max_freq - 1) * (n + 1) + max_freq_tasks_count

    # Return the maximum of minimum intervals and the length of tasks list
    return max(min_intervals, len(tasks))


def longest_word(words: list[str]) -> str:
    """_summary_

    :param words: _description_
    :return: _description_
    """
   # Sort words to ensure lexicographical order
    words.sort()
    
    # Create a set to store valid words
    valid_words = set()
    
    # Variable to keep track of the longest valid word
    longest = ""
    
    # Add an empty string to the set to start the building process
    valid_words.add("")
    
    # Iterate through each word in the sorted list
    for word in words:
        # Check if the word can be formed by adding one character to a valid word
        if word[:-1] in valid_words:
            # Add the word to the set of valid words
            valid_words.add(word)
            # Update the longest word if the current word is longer or lexicographically smaller
            if len(word) > len(longest):
                longest = word
    
    return longest


def shortestWay(source: str, target: str) -> int:
    if set(source) & set(target) != set(target):
        return -1

    source_len = len(source)
    target_len = len(target)
    
    # Pointers for source and target strings
    target_pointer = 0
    count = 0
    
    while target_pointer < target_len:
        source_pointer = 0
        # Scan through the source string
        while source_pointer < source_len and target_pointer < target_len:
            if source[source_pointer] == target[target_pointer]:
                target_pointer += 1
            source_pointer += 1
        
        
        # Increment the count of subsequences used
        count += 1
    
    return count

def countSubarrays(nums: list[int], k: int) -> int:
    """
    The score of an array is defined as the product of its sum and its length.

    For example, the score of [1, 2, 3, 4, 5] is (1 + 2 + 3 + 4 + 5) * 5 = 75. Given
    a positive integer array nums and an integer k, return the number of non-empty
    subarrays of nums whose score is strictly less than k.

    A subarray is a contiguous sequence of elements within an array.

    Example 1:

    Input: nums = [2,1,4,3,5], k = 10 Output: 6 Explanation: The 6 subarrays having
    scores less than 10 are: 
    - [2] with score 2 * 1 = 2
    - [1] with score 1 * 1 = 1.
    - [3] with score 3 * 1 = 3.
    - [4] with score 4 * 1 = 4.
    - [5] with score 5 * 1 = 5.
    - [2,1] with score (2 + 1) * 2 = 6. Note that subarrays such as [1,4] and
    [4,3,5] are not considered because their scores are 10 and 36 respectively,
    while we need scores strictly less than 10.
    """
    n = len(nums)
    start = 0
    current_sum = 0
    count = 0

    for end in range(n):
        current_sum += nums[end]

        # Shrink the window from the start if the score is not less than k
        while start <= end and current_sum * (end - start + 1) >= k:
            current_sum -= nums[start]
            start += 1

        # All subarrays ending at `end` and starting from `start` to `end` are valid
        count += (end - start + 1)

    return count



def jobScheduling(startTime: List[int], endTime: List[int], profit: List[int]) -> int:
    """
    We have n jobs, where every job is scheduled to be done from startTime[i] to
    endTime[i], obtaining a profit of profit[i].

    You're given the startTime, endTime and profit arrays, return the maximum profit
    you can take such that there are no two jobs in the subset with overlapping time
    range.

    If you choose a job that ends at time X you will be able to start another job
    that starts at time X.
    """
    schedule = sorted(zip(startTime, endTime, profit), key=lambda x: x[1])

    # max_profit = 0
    # current_profit = 0
    # last_job = []

    # for start, end, profit in schedule:
    #     eligible_start_idx = bisect_right
    #         current_profit += profit
    #         max_profit = max(max_profit, current_profit)
    
    # return max_profit


def canReach(arr: list[int], start: int) -> bool:
    """
    Use queue and track visited nodes
    """
    queue = [start]
    visited = set()

    while queue:
        i = queue.pop()
        jump = arr[i]

        if jump == 0:
            return True
        
        if i not in visited:
            visited.add(i)
            if i + jump < len(arr):
                queue.append(i + jump)
            if i - jump >= 0:
                queue.append(i - jump)
    
    return False


def countAndSay(n: int) -> str:
    """
    Fuck this problem
    """
    if n == 1:
        return "1"
    
    prev = countAndSay(n-1)
    result = ""
    count = 1

    for i in range(len(prev)):
        if i + 1 < len(prev) and prev[i] == prev[i+1]:
            count += 1
        else:
            result += str(count) + prev[i]
            count = 1
    
    return result


def rightSideView(root: TreeNode) -> List[int]:
    """
    Apply BFS to traverse the tree and append only the
    right most nodes
    """
    if not root:
        return []

    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        for i in range(level_size):
            node = queue.popleft()
            # If it's the last node in the current level, add it to the result
            if i == level_size - 1:
                result.append(node.val)
            # Add left and right children to the queue
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
    
    return result


def isBipartite(graph: List[List[int]]) -> bool:
    """
    Determine given a graph if it is bipartite

    Solve this with coloring technique using BFS
    """
    n = len(graph)
    color = [-1] * n  # -1 indicates that the node has not been colored yet

    for start in range(n):
        if color[start] == -1:  # Not colored yet
            queue = deque([start])
            color[start] = 0  # Color the start node with 0

            while queue:
                node = queue.popleft()
                for neighbor in graph[node]:
                    if color[neighbor] == -1:  # If the neighbor hasn't been colored yet
                        # Color it with the opposite color
                        color[neighbor] = 1 - color[node]
                        queue.append(neighbor)
                    elif color[neighbor] == color[node]:  # If the neighbor has the same color
                        return False
    return True


def count_pairs(nums, mid):
    count = 0
    j = 0
    for i in range(len(nums)):
        while j < len(nums) and nums[j] - nums[i] <= mid:
            j += 1
        count += j - i - 1
    return count

def smallest_distance_pair(nums, k):
    nums.sort()
    low, high = 0, nums[-1] - nums[0]
    
    while low < high:
        mid = (low + high) // 2
        if count_pairs(nums, mid) >= k:
            high = mid
        else:
            low = mid + 1
    
    return low


def compress(chars: List[str]) -> int:
    read_index, counter = 0, 0

    while read_index < len(chars):
        start_of_sequence = read_index
        # Have this read < len(chars) again in case chars contains all of the same string
        # i.e. contains all a's
        while read_index < len(chars) and chars[read_index] == chars[start_of_sequence]:
            read_index += 1

        chars[counter] = chars[start_of_sequence]
        counter += 1

        if read_index - start_of_sequence > 1:
            for digit in str(read_index - start_of_sequence):
                chars[counter] = digit
                counter += 1
    
    return counter


def letterCombinations(digits: str) -> List[str]:
    output = []

    if digits in ["", "1"]:
        return output

    d = {"2": ["a", "b", "c"],
            "3": ["d", "e", "f"],
            "4": ["g", "h", "i"],
            "5": ["j", "k", "l"],
            "6": ["m", "n", "o"],
            "7": ["p", "q", "r", "s"],
            "8": ["t", "u", "v"], "9": ["w", "x", "y", "z"]}

    from itertools import product

    def cartesian_product(lists):
        result = list(product(*lists))
        concatenated_result = ["".join(combination) for combination in result]
        return concatenated_result

    for n in digits:
        output.append(d[n])

    output = cartesian_product(output)

    return output

def return_bigrams(string: str) -> list[str]:
    """
    Return a list of bigrams given an input string
    """
    splits = string.split(" ")
    bigrams = [(splits[i], splits[i+1]) for i in range(len(splits)-1)]

    return bigrams
