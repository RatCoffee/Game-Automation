import numpy as np
import itertools
import math
import re

# COMMON DATA DEFINITIONS:
# A dSize is an integer representing the maximum value of a digit within the Sudoku grid
# A Puzzle is a numpy matrix representing the current solve progress of a Sudoku grid


# PUZZLE IMPORT
################################################################################
# Import a puzzle from a simple clue string
def puzzle_from_string(cluestring):
    # Get special format
    f = re.match(r'\[.*\]', cluestring)
    if f:
        f = f.group()[1:-1]

    # TODO: Cagemap for Jigsaw Sudoku
    puzzle =  np.array([int(s) if s<='9' else
                        int.from_bytes(s.encode())-87 for s in cluestring])
    dSize = math.isqrt(len(puzzle))

    return puzzle, dSize
# TODO: Handling of other methods of import
# TODO: Non-Regular puzzle shapes
# FORMATS:
#     Butterfly Sudoku - Center, +1R, +1DR, +1D
#     Cross Sudoku - Center, +2U, +2R, +2D, +2L
#     Flower Sudoku - Center, +1U, +1R, +1D, +1L
#     Gattai-3 - Center, +1U, +1R, +1DL
#     Kazaguruma - Center, +1U2R, +1R2D, +1D2L, +1L2U
#     Samauri Sudoku - Center, +2UR, +2RD, +2DL, +2LU
#     Sohei Sudoku - +2U, +2R, +2D, +2L
#     Tripledoku - Center, +1RD, +1LU
#     Twodoku - +1RD, +1LU



# SIZE OPERATIONS
################################################################################
# Find the number of digits in a puzzle

# Find the shape of a standard (non-puzzle) block of a given digit size
def block_shape(dSize):
    h = math.isqrt(dSize)
    while dSize%h != 0:
        h -= 1
    return (dSize//h, h)

# TODO: Refactor so weird row configurations are possible

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

#block, inblock, val
def to_houseboard(dSize, bitboard, housemap):
    return np.array([np.concatenate(tuple([bitboard[i:i+1,:]for i in block]))
                     for block in housemap])


if __name__ == "__main__":
    print(puzzle_from_string('[bfly]1000020000300004'))
