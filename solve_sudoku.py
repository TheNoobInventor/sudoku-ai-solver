#!/usr/bin/python3

# This script contains function used to solve a sudoku puzzle.

def solve_puzzle(puzzle, debug=False):
    """
    This function attempts to solve the passed in sudoku puzzle using backtracking. It returns the solved puzzle
    if it is solved and False if there is no solution.
    """

    # If debug is set to True, check if puzzle is valid
    if debug:
        if not is_puzzle_valid(puzzle):
            return False

    # Find the next empty space of the puzzle to guess in
    row, col = find_next_empty(puzzle)

    # If find_next_empty() returns None, None that means there is no free space/element left in the
    # puzzle. Since only valid guesses are filled in (using is_guess_valid() function), the puzzle has been solved.
    if row is None:
        return puzzle

    # If there is a space in the puzzle, pick a number from 1 - 9
    for guess in range(1, 10): 

        # Check if guess is valid
        if is_guess_valid(puzzle, guess, row, col):

            # Place valid guess in the current empty space of the puzzle
            puzzle[row][col] = guess

            # Recursively call the solve_puzzle() function. Returning True means that the puzzle is solved, 
            # from row/column having a value of None.
            if solve_puzzle(puzzle):
                return puzzle

        # If the current guess is not valid, then backtrack and try a different number
        puzzle[row][col] = 0

    # If none of the guesses work, this current iteration of the recursive function is closed and 
    # operation is moved up the previous level. If at the end, none of the numbers tried works, the puzzle
    # is declared unsolvable.
    return False

def is_puzzle_valid(puzzle):
    """
    This function confirms that the puzzle passed in is valid: the puzzle is a list of list, each row is a list, all elements in the 
    puzzle are integers, and there are no duplicate numbers in a row, column or a 3x3 block before an attempt is made to solve it.
    
    Return True if all the checks passed.
    """

    # Puzzle list check
    if not isinstance(puzzle, list):
        raise TypeError('Puzzle should be a list of lists.')
    
    # Row list check
    for i in range(9):
        if not isinstance(puzzle[i], list):
            raise TypeError('Each row should be a list. There should be 9 rows.')

    # Element integer check
    for i in range(9):
        for j in range(9):
            if not isinstance(puzzle[i][j], int):
                raise TypeError('Each element of the puzzle should be an integer.')

    # Duplicate checks
    for i in range(9):
        # Store row and column values
        row_vals = puzzle[i] 
        col_vals = [puzzle[row][i] for row in range(9)] 

        duplicate_check(row_vals) # Row duplicate check
        duplicate_check(col_vals)  # Column duplicate check

        block_vals = [] # Store block values
        
        for r in range(9):
            row_start = (r // 3) * 3 # Get starting row index for the 3x3 block to search in
            
            for c in range(9):
                col_start = (c // 3) * 3 # Get starting column index for the 3x3 block to search in

                for i in range(row_start, row_start+3):
                    for j in range(col_start, col_start+3):
                        block_vals.append(puzzle[i][j])

                duplicate_check(block_vals) # Block duplicate check
                block_vals.clear()

    # All checks passed if this level is reached, therefore the guess is valid and True is returned. 
    return True   

def duplicate_check(list_of_values):
    """
    This function checks for any duplicate numbers in a list.
    """

    for i in range(9):
        # Not concerned with zero values
        if list_of_values[i] == 0:
            continue

        # Count frequency of a number in the list
        list_duplicate = list_of_values.count(list_of_values[i])

        # If a number occurs in the list more than once
        if list_duplicate > 1:
            raise ValueError('The starting puzzle has a duplicate number in a row, column or a 3x3 block.')

def find_next_empty(puzzle):
    """
    This function returns a tuple of the location of the next empty space/cell in the puzzle. If there are no
    more empty spaces, it returns a tuple of (None, None).
    """
    
    # For a 9x9 board, the list indices are from 0 to 8 for both rows and columns.
    for r in range(9):
        for c in range(9):
            if puzzle[r][c] == 0:
                return r, c 
    return None, None 

def is_guess_valid(puzzle, guess, row, col):
    """
    This function checks if a particular guess is valid by checking if the guessed number already appears
    somewhere in the same row, column or 3x3 block associated with it. 
    
    Returns True if the guess is valid and False if it is not.
    """
    
    # Check if the guessed number is already in the associated row. Returns false if already present.
    row_vals = puzzle[row]
    if guess in row_vals:
        return False

    # Check if the guessed number is already in the associated column. Returns false if already present.
    col_vals = [puzzle[i][col] for i in range(9)] 
    if guess in col_vals:
        return False

    row_start = (row // 3) * 3 # Get starting row index for the 3x3 block to search in
    col_start = (col // 3) * 3 # Get starting column index for the 3x3 block to search in
    
    # Check if the guessed number is already in the 3x3 block it is associated in. Returns false if already present.
    for i in range(row_start, row_start+3):
        for j in range(col_start, col_start+3):
            if guess == puzzle[i][j]:
                return False

    # All checks passed if this level is reached, therefore the guess is valid and True is returned. 
    return True   