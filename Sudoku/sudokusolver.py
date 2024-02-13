import numpy as np
import itertools
import math

# COMMON DATA DEFINITIONS:
# A dSize is an integer representing the maximum value of a digit within the Sudoku grid
# A Puzzle is a numpy matrix representing the current solve progress of a Sudoku grid


# SIZE OPERATIONS
############################################################################
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
############################################################################
# Create arrays for row membership
def row_ref(dSize):
    rowNums = np.array([i for i in range(dSize) for j in range(dSize)])
    rowNeigbors = np.array([[y * dSize + x for x in range(dSize)] for y in range(dSize)])
    return rowNums, rowNeigbors

# Create arrays for column membership
def column_ref(dSize):
    columnNums = np.array([i for j in range(dSize) for i in range(dSize)])
    columnNeigbors = np.array([[y * dSize + x for y in range(dSize)] for x in range(dSize)])
    return columnNums, columnNeigbors

# Create arrays for block membership
def block_ref(dSize):
    cw, ch = block_shape(dSize)
    blockNums = np.array([j//ch * ch + i//cw for j in range(dSize) for i in range(dSize)])
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
############################################################################
# Create a per-cell helper bitboard
def basic_bitboard(dSize):
    return np.array([[True for i in range(dSize)]]*(dSize**2))
        
def to_rowboard(dSize, bitboard):
    return np.concatenate(tuple(np.swapaxes(bitboard[i*dSize:(i+1)*dSize,:], 0,1)
                                for i in range(dSize)))

def to_columnboard(dSize, bitboard):
    return np.concatenate(tuple(np.swapaxes(bitboard[i::dSize,:], 0,1)
                                for i in range(dSize)))

def to_blockboard(dSize, bitboard, blockmap):
    return np.concatenate(tuple(np.swapaxes(np.concatenate(
        tuple([bitboard[i:i+1,:] for i in block])), 0, 1) for block in blockmap))

# PUZZLE IMPORT
############################################################################
# Import a puzzle from a simple clue string
def puzzle_from_string(cluestring):
    return np.array([int(s) if s<='9' else int.from_bytes(s.encode())-87 for s in cluestring])
# TODO: Handling of other methods of import
# TODO: Non-Regular puzzle shapes


if __name__ == "__main__":
    import time
    puzzles = [line.strip() for line in open("example.txt", 'r').readlines()]

##    bb = np.array([["%s %d"%(p, i) for i in range(4)]
##                   for p in itertools.product(range(4), repeat=2)])
    
    dSize = 0
    solved = 0
    
    start = time.time()
    
    for puzzleString in puzzles:
        # Import the Puzzle
        puzzle = puzzle_from_string(puzzleString)

        # Set up NeighborInfo if the puzzle shape has changed
        if dSize != digit_size(puzzle):
            dSize = digit_size(puzzle)
            nInfo = NeighborInfo(dSize)

        #TODO: Solve
        
    print("run time:", time.time() - start)
    print("solved: %d/%d"%(solved, len(puzzles)))
    input()
