#!/usr/bin/python3

# This program creates a sudoku class which is used to solve a puzzle or declare it unsolvable if there is 
# no solution.

import time
from pprint import pprint

class Sudoku:
    # Instantiate a sudoku object with a puzzle to be solved and a file to write into. 
    def __init__(self, puzzle, file):
        self.puzzle = puzzle 
        self.file = file
        self.file = open(self.file, 'w')

    def cleanse_puzzle(self): # come up with a better name
        """
        This method confirms that the puzzle is of type list, the rows of the puzzle
        are lists and that the element of the puzzles are integers. Any other
        option raises a TypeError.

        """
        # Puzzle check
        if not isinstance(self.puzzle, list):
            raise TypeError('Puzzle should be a list of lists.')
        
        # Row check
        for i in range(9):
            if not isinstance(self.puzzle[i], list):
                raise TypeError('Each row should be a list. There should be 9 rows.')

        # Element check
        for i in range(9):
            for j in range(9):
                if not isinstance(self.puzzle[i][j], int):
                    raise TypeError('Each element of the puzzle should be an integer.')

        # All checks passed if this level is reached, therefore the contents of the puzzle are valid.
        return True

    def solve_puzzle(self):
        """
        This method attempts to solve the passed in sudoku puzzle using backtracking. The puzzle is a 
        list of lists where each inner list is a row in the puzzle.
    
        Returns
        -------
        bool: True
                If the puzzle can be solved.  
              False
                If there is no solution to the puzzle. 
    
        """
        
        # Find the next empty space of the puzzle to guess in
        row, col = self.find_next_empty()
        
        next_empty_string = ''.join(str(self.find_next_empty()))
        self.file.write('Next empty space in the puzzle: ')
        self.file.write(next_empty_string)
        self.file.write('\n')
    
        # If find_next_empty() returns None, None that means there is no free space/element left in the
        # puzzle. Since only valid guesses are filled in (using is_guess_valid() method), the puzzle has been solved.
        if row is None:
            return True
         
        # If there is a space in the puzzle, pick a number from 1 - 9
        for guess in range(1, 10): 

            # Check if guess is valid
            if self.is_guess_valid(guess, row, col):
                
                self.file.write('\n')
                self.file.write('--------------\n')
                self.file.write('Valid guess: ')
                self.file.write(str(guess))
                self.file.write('\n')
                
                # Place valid guess in the current empty space of the puzzle
                self.puzzle[row][col] = guess
                
                self.file.write('--------------\n')
                valid_row_as_string = ''.join(str(self.puzzle[row]))
                self.file.write(valid_row_as_string)
                self.file.write('\n')
                self.file.write('\n')
                 
                # Recursively call the solve_puzzle() method. Returning True means that the puzzle is solved, 
                # from row/column having a value of None. The file writing the solver outputs is also closed.
                if self.solve_puzzle():
                    self.file.close()
                    return True
             
            # If the current guess is not valid, then backtrack and try a different number
            self.puzzle[row][col] = 0
            
            self.file.write("\nTrial guess: {}".format(guess))
            self.file.write("\n")
            trial_row_as_string = ''.join(str(self.puzzle[row]))
            self.file.write(trial_row_as_string)
            self.file.write('\n')
         
        self.file.write('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        self.file.write('\nGuesses exhausted, backtracking to a previous level of recursion. \n')
        self.file.write('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n')
        self.file.write('Next empty space in the puzzle: ')
        self.file.write(next_empty_string)
        self.file.write('\n')
        
        # If none of the guesses work, this current iteration of the recursive function is closed and 
        # operation is moved up the previous level. If at the end, none of the numbers tried works, the puzzle
        # is declared unsolvable.
        return False

    def find_next_empty(self):
        """
        This methods returns a tuple of the location of the next empty space/cell in the puzzle. If there are no
        more empty spaces, it returns a tuple of (None, None).

        Returns
        -------
        Tuple
            (row_location, column, location). If an empty space is present. 
        Tuple
            (None, None). If there are no empty spaces left. 

        """
        
        # For a 9x9 board, the list indices are from 0 to 8 for both rows and columns.
        for r in range(9):
            for c in range(9):
                if self.puzzle[r][c] == 0:
                    return r, c 
        return None, None 

    def is_guess_valid(self, guess, row, col):
        """
        This method checks if a particular guess is valid by checking if the guessed number already appears
        somewhere in the same row, column or 3x3 block associated with it.
        
        Parameters
        ----------
        guess : int
            Guessed number to validate.
        row : int
            Index of the row to check if the guessed number is already present.
        col : int
            Index of the column to check if the guessed number is already present..

        Returns
        -------
        bool: True
                If the guess is valid.
              False
                If the guess is not valid. 

        """
        # Check if the guessed number is already in the associated row. Returns false if already present.
        row_vals = self.puzzle[row]
        if guess in row_vals:
            return False
        
        # Check if the guessed number is already in the associated column. Returns false if already present.
        col_vals = [self.puzzle[i][col] for i in range(9)] 
        if guess in col_vals:
            return False
        
        # Check if the guessed number is already in the 3x3 block it is associated in. Returns false if already present.
        row_start = (row // 3) * 3 # To get the starting row index for the 3x3 block to search in
        col_start = (col // 3) * 3 # To get the starting column index for the 3x3 block to search in
        
        for i in range(row_start, row_start+3):
            for j in range(col_start, col_start+3):
                if guess == self.puzzle[i][j]:
                    return False
                
        # All checks passed if this level is reached, therefore the guess is valid and True is returned. 
        return True   

if __name__ == '__main__':
    try:
        # Sudoku puzzle to solve; where zeroes denote spaces to be filled out.
        puzzle_1 = [
            [0, 0, 2, 0, 6, 7, 9, 0, 8],
            [0, 4, 0, 0, 0, 0, 2, 0, 7],
            [7, 0, 0, 4, 9, 0, 0, 0, 0],
            [0, 9, 3, 0, 0, 1, 0, 0, 0],
            [2, 0, 0, 8, 0, 3, 0, 0, 9],
            [0, 0, 0, 9, 0, 0, 3, 2, 0],
            [0, 0, 0, 0, 4, 5, 0, 0, 6],
            [9, 0, 4, 0, 0, 0, 0, 3, 0],
            [5, 0, 6, 2, 3, 0, 7, 0, 0]
            ]
        
        print("\nSudoku puzzle to solve: ")
        pprint(puzzle_1)
        
        # Solution to the above puzzle
        soln_1 = [
            [1, 3, 2, 5, 6, 7, 9, 4, 8],
            [6, 4, 9, 3, 1, 8, 2, 5, 7],
            [7, 8, 5, 4, 9, 2, 6, 1, 3],
            [4, 9, 3, 6, 2, 1, 8, 7, 5],
            [2, 5, 1, 8, 7, 3, 4, 6, 9],
            [8, 6, 7, 9, 5, 4, 3, 2, 1],
            [3, 2, 8, 7, 4, 5, 1, 9, 6],
            [9, 7, 4, 1, 8, 6, 5, 3, 2],
            [5, 1, 6, 2, 3, 9, 7, 8, 4]
            ]

        filename = 'sudoku_printout.txt'
        sudoku = Sudoku(puzzle_1, filename)
        if sudoku.cleanse_puzzle():
            start = time.time()
            
            sudoku.solve_puzzle()
            end = time.time()
            
            if soln_1 == sudoku.puzzle:
                print('\nPuzzle as solved below: ')
                pprint(soln_1)
                print("\nPuzzle was solved in {:.3} seconds.".format(end - start))
            else:
                print("\nThe puzzle has no solution!")

    except FileNotFoundError:
        print('\nFile not found, create a file with the name specified')

