import numpy as np
import itertools
import math

# COMMON DATA DEFINITIONS:
# A dSize is an integer representing the maximum value of a digit within the Sudoku grid
# A Puzzle is a numpy matrix representing the current solve progress of a Sudoku grid


# SIZE OPERATIONS
################################################################################
# Find the number of digits in a puzzle
def digit_size(puzzle):
    return math.isqrt(len(puzzle))

# Find the shape of a standard (non-puzzle) block of a given digit size
def block_shape(dSize):
    h = math.isqrt(dSize)
    while dSize%h != 0:
        h -= 1
    return (dSize//h, h)


# NEIGHBOR DETECTION
################################################################################
# Create arrays for row membership
def row_ref(dSize):
    rowNums = np.array([i for i in range(dSize) for j in range(dSize)])
    rowNeigbors = np.array([[y * dSize + x for x in range(dSize)]
                            for y in range(dSize)])
    return rowNums, rowNeigbors

# Create arrays for column membership
def column_ref(dSize):
    columnNums = np.array([i for j in range(dSize) for i in range(dSize)])
    columnNeigbors = np.array([[y * dSize + x for y in range(dSize)]
                               for x in range(dSize)])
    return columnNums, columnNeigbors

# Create arrays for block membership
def block_ref(dSize):
    cw, ch = block_shape(dSize)
    blockNums = np.array([j//ch * ch + i//cw for j in range(dSize)
                          for i in range(dSize)])
    blockNeigbors = np.array([[c%cw+(c//cw)*dSize + (b//ch)*dSize*ch + b%ch*cw
                               for c in range(dSize)] for b in range(dSize)])
    return blockNums, blockNeigbors

# A NeighborInfo, or nInfo, is an consolidation of all neighborhoods
class NeighborInfo:
    def __init__(self, dSize):
        self._size = dSize
        self._rows, self._rowNeighbors = row_ref(dSize)
        self._columns, self._columnNeighbors = column_ref(dSize)
        self._blocks, self._blockNeighbors = block_ref(dSize)

    def Neighbors(self, index):
        if index < 0 or index >= self._size**2:
            return None
        return np.unique(
            np.concatenate((self._rowNeighbors[self._rows[index]],
                            self._columnNeighbors[self._columns[index]],
                            self._blockNeighbors[self._blocks[index]]),
                           axis = 0),
            axis = 0)


# BITBOARDS
################################################################################
# Create a per-cell helper bitboard
def basic_bitboard(dSize):
    return np.array([[True for i in range(dSize)]]*(dSize**2))

#y, x, val
def to_rowboard(dSize, bitboard):
    return np.array([bitboard[i*dSize:(i+1)*dSize,:] for i in range(dSize)])

#x, y, val
def to_columnboard(dSize, bitboard):
    return np.array([bitboard[i::dSize,:] for i in range(dSize)])

#block, inblock, val
def to_blockboard(dSize, bitboard, blockmap):
    return np.array([np.concatenate(tuple([bitboard[i:i+1,:]for i in block]))
                     for block in blockmap])

# PUZZLE IMPORT
################################################################################
# Import a puzzle from a simple clue string
def puzzle_from_string(cluestring):
    return np.array([int(s) if s<='9' else
                     int.from_bytes(s.encode())-87 for s in cluestring])
# TODO: Handling of other methods of import
# TODO: Non-Regular puzzle shapes

# PUZZLE SOLVE
################################################################################
# Update the bitboard focusing on a specific square
def update_square(index, puzzle, dSize, nInfo, bitboard):
    mask = np.array([puzzle[index] != j+1 for j in range(dSize)])
    bitboard[index] = np.array([False] * dSize)
    for n in nInfo.Neighbors(index):
        bitboard[n] = np.all((bitboard[n], mask), axis = 0)

# Initialize the bitboard
def init_bitboard(puzzle, dSize, nInfo, bitboard):
    for i in range(len(puzzle)):
        if puzzle[i]>0:
            update_square(i, puzzle, dSize, nInfo, bitboard)
    return dSize**3 - np.sum(bitboard)

# Find naked singles
def naked_singles(puzzle, dSize, nInfo, bitboard):
    count = np.sum(bitboard)
    for i in range(len(puzzle)):
        if np.sum(bitboard[i]) == 1:
            puzzle[i] = np.where(bitboard[i] == True)[0][0]+1
            update_square(i, puzzle, dSize, nInfo, bitboard)
    return count - np.sum(bitboard)

# Find hidden singles
def hidden_singles(puzzle, dSize, nInfo, bitboard):
    count = np.sum(bitboard)
    
    rowboard = to_rowboard(dSize, bitboard)
    for y, val in itertools.product(range(dSize), repeat=2):
        if np.sum(rowboard[y, :, val]) == 1:
            x = np.where(rowboard[y, :, val] == True)[0][0]
            index = y*dSize+x
            puzzle[index] = val+1
            update_square(index, puzzle, dSize, nInfo, bitboard)
            rowboard = to_rowboard(dSize, bitboard)
    
    columnboard = to_columnboard(dSize, bitboard)
    for x, val in itertools.product(range(dSize), repeat=2):
        if np.sum(columnboard[x, :, val]) == 1:
            y = np.where(columnboard[x, :, val] == True)[0][0]
            index = y*dSize+x
            puzzle[index] = val+1
            update_square(index, puzzle, dSize, nInfo, bitboard)
            columnboard = to_columnboard(dSize, bitboard)

    blockboard = to_blockboard(dSize, bitboard, nInfo._blockNeighbors)
    for block, val in itertools.product(range(dSize), repeat=2):
        if np.sum(blockboard[block, :, val]) == 1:
            inblock = np.where(blockboard[block, :, val] == True)[0][0]
            index = nInfo._blockNeighbors[block, inblock]
            puzzle[index] = val+1
            update_square(index, puzzle, dSize, nInfo, bitboard)
            blockboard = to_blockboard(dSize, bitboard, nInfo._blockNeighbors)
            
    return count - np.sum(bitboard)

# Find naked pairs
# TODO: Adapt this to idenitfy naked tuples of many sizes
def naked_pairs(puzzle, dSize, nInfo, bitboard):
    count = np.sum(bitboard)
    
    rowboard = to_rowboard(dSize, bitboard)
    for y in range(dSize):
        for i, j in itertools.combinations(range(dSize), 2):
            if np.array_equal(rowboard[y,i], rowboard[y,j]) and np.sum(rowboard[y,i]) == 2:
                mask = np.logical_not(rowboard[y,i])
                for x in range(dSize):
                    if x!=i and x!=j:
                        bitboard[y*dSize+x] = np.logical_and(mask,bitboard[y*dSize+x])
            rowboard = to_rowboard(dSize, bitboard)

    columnboard = to_columnboard(dSize, bitboard)
    for x in range(dSize):
        for i, j in itertools.combinations(range(dSize), 2):
            if np.array_equal(columnboard[x,i], columnboard[x,j]) and np.sum(columnboard[x,i]) == 2:
                mask = np.logical_not(columnboard[x,i])
                for y in range(dSize):
                    if y!=i and y!=j:
                        bitboard[y*dSize+x] = np.logical_and(mask,bitboard[y*dSize+x])
                columnboard = to_columnboard(dSize, bitboard)

    blockboard = to_blockboard(dSize, bitboard, nInfo._blockNeighbors)
    for b in range(dSize):
        for i, j in itertools.combinations(range(dSize), 2):
            if np.array_equal(blockboard[b,i], blockboard[b,j]) and np.sum(blockboard[b,i]) == 2:
                mask = np.logical_not(blockboard[b,i])
                for n in range(dSize):
                    if n!=i and n!=j:
                        index = nInfo._blockNeighbors[b, n]
                        bitboard[index] = np.logical_and(mask, bitboard[index])
                blockboard = to_blockboard(dSize, bitboard, nInfo._blockNeighbors) 
            
    return count - np.sum(bitboard)


# Find naked pairs
# TODO: Adapt this to idenitfy naked tuples of many sizes
def naked_tuples(puzzle, dSize, nInfo, bitboard):
    count = np.sum(bitboard)

    maxSize = dSize//2 + 1

    for tupleSize in range(2, 3):#maxSize
        rowboard = to_rowboard(dSize, bitboard)
        for y in range(dSize):
            for t in itertools.combinations(range(dSize), tupleSize):
                intersection = np.any(tuple(rowboard[y, x] for x in t), axis = 0)
                if False not in [np.any((rowboard[y,x])) for x in t] and np.sum(intersection) > 0 and np.sum(intersection) <= tupleSize:
                    mask = np.logical_not(intersection)
                    for x in range(dSize):
                        if x not in t:
                            bitboard[y*dSize+x] = np.logical_and(mask,bitboard[y*dSize+x])
                    print()
                rowboard = to_rowboard(dSize, bitboard)

        columnboard = to_columnboard(dSize, bitboard)
        for x in range(dSize):
            for t in itertools.combinations(range(dSize), tupleSize):
                intersection = np.any(tuple(rowboard[x, y] for y in t), axis = 0)
                print(x,y)
                if np.sum(intersection) > 0 and np.sum(intersection) <= tupleSize:
                    print("Columns", x, t)
                    mask = np.logical_not(intersection)
                    print(mask)
                    for y in range(dSize):
                        if y not in t:
                            print(y)
                            bitboard[y*dSize+x] = np.logical_and(mask,bitboard[y*dSize+x])
                            print(bitboard[y*dSize+x])
                columnboard = to_columnboard(dSize, bitboard)

        blockboard = to_blockboard(dSize, bitboard, nInfo._blockNeighbors)
        for b in range(dSize):
            for t in itertools.combinations(range(dSize), tupleSize):
                intersection = np.any(tuple(rowboard[b, n] for n in t), axis = 0)
                if np.sum(intersection) > 0 and  np.sum(intersection) <= tupleSize:
                    mask = np.logical_not(intersection)
                    for n in range(dSize):
                        if n not in t:
                            index = nInfo._blockNeighbors[b, n]
                            bitboard[index] = np.logical_and(mask, bitboard[index])
                blockboard = to_blockboard(dSize, bitboard, nInfo._blockNeighbors)

        if np.sum(bitboard) < count:
            return count - np.sum(bitboard)
            
    return count - np.sum(bitboard)



# Find hidden pairs
# TODO: Adapt this to idenitfy hidden tuples of many sizes
##def hidden_pairs(puzzle, dSize, nInfo, bitboard):
##    count = np.sum(bitboard)
##    
##    rowboard = to_rowboard(dSize, bitboard)
##    for y in range(dSize):
##        for u, v in itertools.combinations(range(dSize), 2):
##            if np.array_equal(rowboard[y,:,u], rowboard[y,:,v]) and np.sum(rowboard[y,:,u]) == 2:
##                mask = np.array([i+1 not in (u,v) for i in range(dSize)])
##                for x in range(dSize):
##                    if x!=i and x!=j:
##                        bitboard[y*dSize+x] = np.logical_and(mask,bitboard[y*dSize+x])
##            rowboard = to_rowboard(dSize, bitboard)
##
##    columnboard = to_columnboard(dSize, bitboard)
##    for x in range(dSize):
##        for i, j in itertools.combinations(range(dSize), 2):
##            if np.array_equal(columnboard[x,i], columnboard[x,j]) and np.sum(columnboard[x,i]) == 2:
##                mask = np.logical_not(columnboard[x,i])
##                for y in range(dSize):
##                    if y!=i and y!=j:
##                        bitboard[y*dSize+x] = np.logical_and(mask,bitboard[y*dSize+x])
##                columnboard = to_columnboard(dSize, bitboard)
##
##    blockboard = to_blockboard(dSize, bitboard, nInfo._blockNeighbors)
##    for b in range(dSize):
##        for i, j in itertools.combinations(range(dSize), 2):
##            if np.array_equal(blockboard[b,i], blockboard[b,j]) and np.sum(blockboard[b,i]) == 2:
##                mask = np.logical_not(blockboard[b,i])
##                for n in range(dSize):
##                    if n!=i and n!=j:
##                        index = nInfo._blockNeighbors[b, n]
##                        bitboard[index] = np.logical_and(mask, bitboard[index])
##                blockboard = to_blockboard(dSize, bitboard, nInfo._blockNeighbors) 
##            
##    return count - np.sum(bitboard)


# Determine if the puzzle has been solved
def is_solved(puzzle):
    return 0 not in puzzle

#TODO: Verify Solution as Valid
def valid_solution(clueString, puzzle):
    refPuzzle = puzzle_from_string(clueString)
    dSize = digit_size(refPuzzle)
    if 0 in puzzle:
        return False
    for i in range(dSize):
        if (len(np.unique(puzzle[i*dSize:(i+1)*dSize])) < dSize or
            len(np.unique(puzzle[i::dSize])) < dSize or
            (refPuzzle[i] != 0 and refPuzzle[i] != puzzle[i])):
            return False
    return True

SOLVE_METHODS = [naked_singles, hidden_singles, naked_tuples] # naked_pairs

# Iteratively solve the puzzlet
def iter_solve(puzzle, dSize, nInfo):
    bitboard = basic_bitboard(dSize)
    rSolved = init_bitboard(puzzle, dSize, nInfo, bitboard)
    while not is_solved(puzzle) and rSolved > 0:
        rSolved = 0
        for method in SOLVE_METHODS:
            rSolved += method(puzzle, dSize, nInfo, bitboard)
            if rSolved > 0:
                break
    return bitboard

if __name__ == "__main__":
    import time
    puzzles = [line.strip() for line in open("example.txt", 'r').readlines()]

# Used for some testing purposes
##    bb = np.array([["%s %d"%(p, i) for i in range(4)]
##                   for p in itertools.product(range(4), repeat=2)])

    dSize = 0
    solved = [0] * (1+len(puzzles)//1000)

    puzzle = puzzle_from_string(puzzles[6000])
    dSize = digit_size(puzzle)
    print(np.reshape(puzzle, (dSize, dSize)))
    nInfo = NeighborInfo(dSize)
    print(iter_solve(puzzle, dSize, nInfo))
    print(np.reshape(puzzle, (dSize, dSize)))
    
##    start = time.time()
##    for e, clueString in enumerate(puzzles):
##        # Import the Puzzle
##        puzzle = puzzle_from_string(clueString)
##
##        # Set up NeighborInfo if the puzzle shape has changed
##        if dSize != digit_size(puzzle):
##            dSize = digit_size(puzzle)
##            nInfo = NeighborInfo(dSize)
##
##        bboard = iter_solve(puzzle, dSize, nInfo)
##        if valid_solution(clueString, puzzle):
##            solved[e//1000] += 1
##        else:
##            print(e)
##        
##    print("run time:", time.time() - start)
##    print("solved: %s"%solved)
##    input()
