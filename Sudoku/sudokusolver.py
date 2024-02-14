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
        
def to_rowboard(dSize, bitboard):
    return np.array([bitboard[i*dSize:(i+1)*dSize,:] for i in range(dSize)])

def to_columnboard(dSize, bitboard):
    return np.array([bitboard[i::dSize,:] for i in range(dSize)])

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
    print("Naked Singles")
    count = np.sum(bitboard)
    for i in range(len(puzzle)):
        if np.sum(bitboard[i]) == 1:
            puzzle[i] = np.where(bitboard[i] == True)[0][0]+1
            update_square(i, puzzle, dSize, nInfo, bitboard)
    return count - np.sum(bitboard)

# Find hidden singles
def hidden_singles(puzzle, dSize, nInfo, bitboard):
    print("Hidden Singles")
    count = np.sum(bitboard)
    
    rowboard = to_rowboard(dSize, bitboard)
    for y, x in itertools.product(range(dSize), repeat=2):
        if np.sum(rowboard[y, x]) == 1:
            val = np.where(rowboard[y, x] == True)[0][0]+1
            index = y*dSize+x
            puzzle[index] = val
            update_square(index, puzzle, dSize, nInfo, bitboard)
            rowboard = to_rowboard(dSize, bitboard)
    
    columnboard = to_columnboard(dSize, bitboard)
    for x, y in itertools.product(range(dSize), repeat=2):
        if np.sum(columnboard[x, y]) == 1:
            val = np.where(columnboard[x, y] == True)[0][0]+1
            index = y*dSize+x
            puzzle[index] = val
            update_square(index, puzzle, dSize, nInfo, bitboard)
            columnboard = to_columnboard(dSize, bitboard)

    blockboard = to_blockboard(dSize, bitboard, nInfo._blockNeighbors)
    for block, inblock in itertools.product(range(dSize), repeat=2):
        if np.sum(blockboard[block, inblock]) == 1:
            val = np.where(blockboard[block,inblock] == True)[0][0] +1
            index = nInfo._blockNeighbors[block, inblock]
            puzzle[index] = val
            update_square(index, puzzle, dSize, nInfo, bitboard)
            blockboard = to_blockboard(dSize, bitboard, nInfo._blockNeighbors)
            
    return count - np.sum(bitboard)

# Find naked pairs
def naked_pairs(puzzle, dSize, nInfo, bitboard):
    count = np.sum(bitboard)
    
    rowboard = to_rowboard(dSize, bitboard)
    for y in range(dSize):
        for i, j in itertools.combinations(range(dSize), 2):
            if np.array_equal(rowboard[y,i], rowboard[y,j]) and np.sum(rowboard[y,i]) == 2:
                mask = np.logical_not(rowboard[y,i])
                print(mask)
                for x in range(dSize):
                    if x!=i and x!=j:
                        bitboard[y*dSize+x] = np.logical_and(mask,bitboard[y*dSize+x])
            rowboard = to_rowboard(dSize, bitboard)

    columnboard = to_columnboard(dSize, bitboard)
    for x in range(dSize):
        for i, j in itertools.combinations(range(dSize), 2):
            if np.array_equal(columnboard[x,i], columnboard[x,j]) and np.sum(columnboard[x,i]) == 2:
                mask = np.logical_not(columnboard[x,i])
                print(mask)
                for y in range(dSize):
                    if y!=i and y!=j:
                        bitboard[y*dSize+x] = np.logical_and(mask,bitboard[y*dSize+x])
                columnboard = to_columnboard(dSize, bitboard)

    blockboard = to_blockboard(dSize, bitboard, nInfo._blockNeighbors)
    for b in range(dSize):
        for i, j in itertools.combinations(range(dSize), 2):
            if np.array_equal(blockboard[b,i], blockboard[b,j]) and np.sum(blockboard[b,i]) == 2:
                mask = np.logical_not(blockboard[b,i])
                print(mask)
                for n in range(dSize):
                    if n!=i and n!=j:
                        index = nInfo._blockNeighbors[b, n]
                        bitboard[index] = np.logical_and(mask, bitboard[index])
                blockboard = to_blockboard(dSize, bitboard, nInfo._blockNeighbors) 
            
    return count - np.sum(bitboard)




# Determine if the puzzle has been solved
def is_solved(puzzle):
    return 0 not in puzzle

SOLVE_METHODS = [naked_singles, hidden_singles] #, naked_pairs

# Iteratively solve the puzzlet
def iter_solve(puzzle, dSize, nInfo):
    bitboard = basic_bitboard(dSize)
    rSolved = init_bitboard(puzzle, dSize, nInfo, bitboard)
    iterCount = 0
    while not is_solved(puzzle) and rSolved > 0:
        iterCount += 1
        rSolved = 0
        for method in SOLVE_METHODS:
            rSolved += method(puzzle, dSize, nInfo, bitboard)
            if rSolved > 0:
                break
        print("Itercount:", iterCount)
        print(np.reshape(puzzle, (dSize, dSize)))
    print(bitboard)
    print(to_rowboard(dSize, bitboard))
    print('\n')
    print(to_columnboard(dSize, bitboard))
    print('\n')
    print(to_blockboard(dSize, bitboard,nInfo._blockNeighbors)) 
    

if __name__ == "__main__":
    import time
    puzzles = [line.strip() for line in open("example.txt", 'r').readlines()]

# Used for some testing purposes
##    bb = np.array([["%s %d"%(p, i) for i in range(4)]
##                   for p in itertools.product(range(4), repeat=2)])
    
    dSize = 0
    solved = 0
    printed = False
    puzzle = puzzle_from_string(puzzles[3000])
    dSize = digit_size(puzzle)
    nInfo = NeighborInfo(dSize)
    bboard = basic_bitboard(dSize)
    init_bitboard(puzzle, dSize, nInfo, bboard)
    print(np.reshape(puzzle, (dSize, dSize)))
##    print(bboard)
##    print()
    iter_solve(puzzle, dSize, nInfo)
    print(np.reshape(puzzle, (dSize, dSize)))
    
##    start = time.time()
##    
##    for e, puzzleString in enumerate(puzzles):
##        # Import the Puzzle
##        puzzle = puzzle_from_string(puzzleString)
##
##        # Set up NeighborInfo if the puzzle shape has changed
##        if dSize != digit_size(puzzle):
##            dSize = digit_size(puzzle)
##            nInfo = NeighborInfo(dSize)
##
##        iter_solve(puzzle, dSize, nInfo)
##        if is_solved(puzzle):
##            solved += 1
##        elif not printed:
##            print(np.reshape(puzzle, (dSize, dSize)))
##            print(np.reshape(nInfo._blockNeighbors,(dSize, dSize)))
##            printed = True
##        
##    print("run time:", time.time() - start)
##    print("solved: %d/%d"%(solved, len(puzzles)))
##    input()
