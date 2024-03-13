from sudokusolver import *
import time

s = 2

testSets = [('smallTest.txt', 20),
            ('midTest.txt', 100),
            ('example.txt', 1000)]

filename, size = testSets[s]

puzzles = [line.strip() for line in open(filename, 'r').readlines()][:4000]

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
    else:
        print(e)
        print(puzzle_from_string(clueString).reshape((dSize, dSize)))
    if e%size == size-1:
        times[e//size] = time.time() - start
        start = time.time()
        print(e+1)
    
print("run times:", times)
print("solved: %s"%solved)
print("total runtime:", sum(times))
input()
