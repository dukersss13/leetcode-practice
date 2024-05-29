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


def test_string_compression():
    chars = ["a", "b", "b", "b", "b", "b", "b", "b", "b", "b", "b", "b", "b"]
    results = string_compression(chars)

    # Because results should be "["a", "b", "1", "2"]"
    assert results == 4


def test_reconstruct_itinerary():
    tickets = [["JFK", "SFO"], ["JFK", "ATL"], ["SFO", "ATL"], ["ATL", "JFK"], ["ATL", "SFO"]]
    results = reconstruct_itinerary(tickets)


def test_search_insert():
    nums = [1, 3, 5, 6]
    target = 2
    results = searchInsert(nums, target)

    assert results == 1


def test_min_transfers():
    transactions = [[0, 1, 10], [1, 0, 1], [1, 2, 5], [2, 0, 5]]
    results = minTransfers(transactions)
    
    assert results == 1


def test_moveZeros():
    nums = [0, 1, 0, 3, 12]
    results = moveZeros(nums)

    assert results == [1, 3, 12, 0, 0]


def test_least_intervals():
     tasks = ["A","A","A","B","B","B"]
     n = 2
     results = least_intervals(tasks, n)


def test_longest_word():
    words = ["b","br","bre","brea","break","breakf","breakfa","breakfas",
             "breakfast","l","lu","lun","lunc","lunch","d","di","din","dinn","dinne","dinner"]
    results = longest_word(words)


def test_shortest_way():
    source = "xyz"
    target = "xzyxz"
    results = shortestWay(source, target)

    assert results == 3


def test_count_subarrays():
    nums = [2, 1, 4, 3, 5]
    k = 10
    results = countSubarrays(nums, k)

    assert results == 6


def test_jobScheduling():
    startTime = [1, 2, 3, 3]
    endTime = [3, 4, 5, 6]
    profit = [50, 10, 40, 70]

    results = jobScheduling(startTime, endTime, profit)
    assert results == 120


def test_canReach():
    arr = [4,2,3,0,3,1,2]
    start = 5

    results = canReach(arr, start)

    assert results

def test_countAndSay():
    n = 4
    result = countAndSay(n)

    assert result == "1211"


def test_right_side_view():
# Constructing the binary tree
#        1
#       / \
#      2   3
#       \   \
#        5   4
    root = TreeNode(1)                          
    root.left = TreeNode(2)
    root.right = TreeNode(3)
    root.left.right = TreeNode(5)
    root.right.right = TreeNode(4)

    results = rightSideView(root)

    assert results == [1, 3, 4]


def test_isBipartite():
    graph = [[1,3], [0,2], [1,3], [0,2]]
    results = isBipartite(graph)
    assert results


def test_smallest_distance_pair():
    # Example usage
    nums = [9, 10, 7, 10, 6, 1, 5, 4, 9, 8]
    k = 18
    assert smallest_distance_pair(nums, k) == 2


def test_letterCombo():
    digits = "23"

    results = letterCombinations(digits)
    assert results == ["ad","ae","af","bd","be","bf","cd","ce","cf"]


def test_bigrams():
    string = "Hello World my name is Duc"

    results = return_bigrams(string)

    assert results == [('Hello', 'World'), ('World', 'my'), ('my', 'name'), ('name', 'is'), ('is', 'Duc')]
