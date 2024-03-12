from sudokusolver import *
import time

s = 1

testSets = [('smallTest.txt', 20),
            ('midTest.txt', 100),
            ('example.txt', 1000)]

filename, size = testSets[s]

puzzles = [line.strip() for line in open(filename, 'r').readlines()]

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
print("total runtime:", sum(times))
input()

# TESTS (mid)
################################################################################
# Tuples (2, n)
#     [100, 100, 100, 100, 97, 97, 100, 94, 84, 100, 99, 82, 92, 53, 41, 100, 100, 37, 100, 68, 14]
#     46.9102578163147
# Tuples (3, n)
#     [100, 100, 100, 100, 97, 97, 100, 94, 84, 100, 99, 82, 92, 54, 44, 100, 100, 38, 100, 70, 18]
#     59.921911001205444
# Tuples (4, n)
#     [100, 100, 100, 100, 97, 97, 100, 94, 84, 100, 99, 82, 92, 54, 44, 100, 100, 38, 100, 70, 18]
#     84.0196623802185
# Tuples (4, n) + Pointing
#     [100, 100, 100, 100, 98, 97, 100, 94, 87, 100, 99, 83, 92, 58, 49, 100, 100, 43, 100, 73, 25]
#     113.53811240196228
# Tuples (4, n) + Pointing + BLR
#     []
#     xxx.x

