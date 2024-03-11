from sudokusolver import *
import time

size = 1000

puzzles = [line.strip() for line in open("example.txt", 'r').readlines()]

dSize = 0
solved = [0] * math.ceil(len(puzzles)/size)
times = [None] * math.ceil(len(puzzles)/size)
start = time.time()
for e, clueString in enumerate(puzzles):
    # Import the Puzzle
    puzzle = puzzle_from_string(clueString)

    # Set up NeighborInfo if the puzzle shape has changed
    if dSize != digit_size(puzzle):
        dSize = digit_size(puzzle)
        nInfo = NeighborInfo(dSize)

    bboard = iter_solve(puzzle, dSize, nInfo)
    if valid_solution(clueString, puzzle):
        solved[e//size] += 1
    if e%size == size-1:
        times[e//size] = time.time() - start
        start = time.time()
        print(e+1)
    
print("run times:", times)
print("solved: %s"%solved)
input()
