import pytest
from solve_sudoku import *

puzzle_1 = [
    [0, 1, 0, 2, 0, 0, 0, 0, 0],
    [0, 0, 5, 0, 0, 0, 2, 9, 0],
    [9, 2, 0, 4, 5, 0, 1, 0, 0],
    [0, 0, 1, 0, 6, 2, 8, 0, 0],
    [0, 8, 2, 0, 0, 0, 5, 3, 0],
    [0, 0, 6, 7, 8, 0, 9, 0, 0],
    [0, 0, 9, 0, 1, 4, 0, 5, 2],
    [0, 6, 4, 0, 0, 0, 7, 0, 0],
    [0, 0, 0, 0, 0, 6, 0, 1, 0]
    ]

solution_1 = [
    [6, 1, 7, 2, 9, 8, 3, 4, 5],
    [8, 4, 5, 6, 3, 1, 2, 9, 7],
    [9, 2, 3, 4, 5, 7, 1, 6, 8],
    [5, 9, 1, 3, 6, 2, 8, 7, 4],
    [7, 8, 2, 1, 4, 9, 5, 3, 6],
    [4, 3, 6, 7, 8, 5, 9, 2, 1],
    [3, 7, 9, 8, 1, 4, 6, 5, 2],
    [1, 6, 4, 5, 2, 3, 7, 8, 9],
    [2, 5, 8, 9, 7, 6, 4, 1, 3]
    ]

puzzle_2 = [
    [0, 0, 0, 0, 0, 9, 0, 0, 0],
    [0, 0, 0, 1, 3, 0, 0, 5, 0],
    [0, 0, 0, 5, 8, 0, 'w', 0, 9],
    [2, 0, 8, 0, 0, 0, 0, 9, 4],
    [0, 4, 9, 0, 0, 0, 1, 8, 0],
    [3, 1, 0, 0, 0, 0, 5, 0, 2],
    [1, 0, 4, 0, 6, 8, 0, 0, 0],
    [0, 9, 0, 0, 1, 7, 0, 0, 0],
    [0, 0, 0, 2, 0, 0, 0, 0, 0]
    ]

puzzle_3 = [
    [0, 5, 5, 7, 0, 0, 0, 0, 0],
    [6, 0, 6, 0, 6, 0, 0, 0, 0],
    [2, 0, 0, 5, 6, 0, 7, 0, 0],
    [6, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 2, 0, 0, 0, 5, 0, 0, 0],
    [6, 0, 6, 0, 0, 0, 0, 6, 0],
    [0, 0, 0, 0, 9, 0, 0, 3, 0],
    [0, 4, 0, 3, 0, 0, 0, 0, 8],
    [7, 0, 0, 0, 0, 0, 4, 0, 0]
    ]

puzzle_4 = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 2, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 6, 0, 0, 9, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 7, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 4, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 3, 0, 0, 0, 0, 0, 0]
    ]

# Tests

# Test to confirm that the puzzle and solution are not the same before invoking the solve_puzzle() method
def test_puzzle_before_solver(): 
    assert puzzle_1 != solution_1

# Test to confirm that the puzzle has an element that is not of type int
def test_puzzle_has_a_wrong_element_type(): 
    with pytest.raises(TypeError):
        solve_puzzle(puzzle_2, debug=True)

# Test to confirm that the solved puzzle and the provide solution are the same
def test_solver():
    puzzle_1_solved = solve_puzzle(puzzle_1)
    assert puzzle_1_solved == solution_1

# Test to confirm that there is a duplicate in a puzzle row
def test_duplicate_row():
    with pytest.raises(ValueError):
        solve_puzzle(puzzle_3, debug=True)

# Test to confirm that there is a duplicate in a puzzle block
def test_duplicate_block():
    with pytest.raises(ValueError):
        solve_puzzle(puzzle_4, debug=True)