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
    for i in np.where(np.sum(bitboard, axis = 1) == 1)[0]:
        puzzle[i] = np.where(bitboard[i] == True)[0][0]+1
        update_square(i, puzzle, dSize, nInfo, bitboard)
    return count - np.sum(bitboard)

# Find hidden singles
def hidden_singles(puzzle, dSize, nInfo, bitboard):
    count = np.sum(bitboard)
    
    rowboard = to_rowboard(dSize, bitboard)
    for y, val in zip(*np.where(np.sum(rowboard, axis = 1) == 1)):
        x = np.where(rowboard[y, :, val] == True)[0][0]
        index = y*dSize+x
        puzzle[index] = val+1
        update_square(index, puzzle, dSize, nInfo, bitboard)
        rowboard = to_rowboard(dSize, bitboard)
    
    columnboard = to_columnboard(dSize, bitboard)
    for x, val in zip(*np.where(np.sum(columnboard, axis = 1) == 1)):
        y = np.where(columnboard[x, :, val] == True)[0][0]
        index = y*dSize+x
        puzzle[index] = val+1
        update_square(index, puzzle, dSize, nInfo, bitboard)
        columnboard = to_columnboard(dSize, bitboard)

    blockboard = to_blockboard(dSize, bitboard, nInfo._blockNeighbors)
    for block, val in zip(*np.where(np.sum(blockboard, axis = 1) == 1)):
        inblock = np.where(blockboard[block, :, val] == True)[0][0]
        index = nInfo._blockNeighbors[block, inblock]
        puzzle[index] = val+1
        update_square(index, puzzle, dSize, nInfo, bitboard)
        blockboard = to_blockboard(dSize, bitboard, nInfo._blockNeighbors)
            
    return count - np.sum(bitboard)

### Find naked tuples
def naked_tuples(puzzle, dSize, nInfo, bitboard):
    count = np.sum(bitboard)
    maxSize = 2#min(dSize//2 + 1, 4)

    for tupleSize in range(2, 3):
        rowboard = to_rowboard(dSize, bitboard)
        for y in range(dSize):
            for t in itertools.combinations(range(dSize), tupleSize):
                union = np.any(tuple(rowboard[y, x] for x in t), axis = 0)
                if np.sum(union) == tupleSize and False not in [np.any(rowboard[y,x]) for x in t]:
                    mask = np.logical_not(union)
                    for x in range(dSize):
                        if x not in t:
                            bitboard[y*dSize+x] = np.logical_and(mask,bitboard[y*dSize+x])
                rowboard = to_rowboard(dSize, bitboard)

        columnboard = to_columnboard(dSize, bitboard)
        for x in range(dSize):
            for t in itertools.combinations(range(dSize), tupleSize):
                union = np.any(tuple(columnboard[x, y] for y in t), axis = 0)
                if np.sum(union) == tupleSize and False not in [np.any(columnboard[x,y]) for y in t]:
                    mask = np.logical_not(union)
                    for y in range(dSize):
                        if y not in t:
                            bitboard[y*dSize+x] = np.logical_and(mask,bitboard[y*dSize+x])
                columnboard = to_columnboard(dSize, bitboard)

        blockboard = to_blockboard(dSize, bitboard, nInfo._blockNeighbors)
        for b in range(dSize):
            for t in itertools.combinations(range(dSize), tupleSize):
                intersection = np.any(tuple(blockboard[b, n] for n in t), axis = 0)
                if np.sum(intersection) == tupleSize and False not in [np.any(blockboard[b,n]) for n in t]:
                    mask = np.logical_not(intersection)
                    for n in range(dSize):
                        if n not in t:
                            index = nInfo._blockNeighbors[b, n]
                            bitboard[index] = np.logical_and(mask, bitboard[index])
                blockboard = to_blockboard(dSize, bitboard, nInfo._blockNeighbors)
        if count > np.sum(bitboard):
            return count - np.sum(bitboard)

    return count - np.sum(bitboard)


def hidden_tuples(puzzle, dSize, nInfo, bitboard):
    count = np.sum(bitboard)
    maxSize = 2#min(dSize//2 + 1, 4)


    for tupleSize in range(2, maxSize):
        
        rowboard = to_rowboard(dSize, bitboard)
        for y in range(dSize):
            for t in itertools.combinations(range(dSize), tupleSize):
                union = np.any(tuple(rowboard[y,:,v] for v in t), axis = 0)
                if np.sum(union) == tupleSize and False not in [np.any(rowboard[y,:,v]) for v in t]:
                    mask = np.array([i in t for i in range(dSize)])
                    for x in range(dSize):
                        if union[x]:
                            bitboard[y*dSize+x] = np.logical_and(mask,bitboard[y*dSize+x])
                rowboard = to_rowboard(dSize, bitboard)

        columnboard = to_columnboard(dSize, bitboard)
        for x in range(dSize):
            for t in itertools.combinations(range(dSize), tupleSize):
                union = np.any(tuple(columnboard[x,:,v] for v in t), axis = 0)
                if np.sum(union) == tupleSize and False not in [np.any(columnboard[x,:,v]) for v in t]:
                    mask = np.array([i in t for i in range(dSize)])
                    for y in range(dSize):
                        if union[y]:
                            bitboard[y*dSize+x] = np.logical_and(mask,bitboard[y*dSize+x])
                    columnboard = to_columnboard(dSize, bitboard)

        blockboard = to_blockboard(dSize, bitboard, nInfo._blockNeighbors)
        for b in range(dSize):
            for t in itertools.combinations(range(dSize), tupleSize):
                union = np.any(tuple(blockboard[b,:,v] for v in t), axis = 0)
                if np.sum(union) == tupleSize and False not in [np.any(blockboard[b,:,v]) for v in t]:
                    mask = np.array([i in t for i in range(dSize)])
                    for n in range(dSize):
                        if union[n]:
                            index = nInfo._blockNeighbors[b, n]
                            bitboard[index] = np.logical_and(mask, bitboard[index])
                    blockboard = to_blockboard(dSize, bitboard, nInfo._blockNeighbors) 
            
    return count - np.sum(bitboard)


# Find pointing pairs
def pointing_digits(puzzle, dSize, nInfo, bitboard):
    count = np.sum(bitboard)
    
    blockboard = to_blockboard(dSize, bitboard, nInfo._blockNeighbors)
    for b in range(dSize):
        for v in range(dSize):
            ix = np.array([nInfo._blockNeighbors[b,n] for n in np.where(blockboard[b,:,v])[0]])
            if len(np.unique(ix//dSize)) == 1:
                mask = np.array([i != v for i in range(dSize)])
                y = (ix//dSize)[0]
                for x in range(dSize):
                    index = y*dSize + x
                    if index not in nInfo._blockNeighbors[b]:
                        bitboard[index] = np.logical_and(mask, bitboard[index])
                blockboard = to_blockboard(dSize, bitboard, nInfo._blockNeighbors)
            elif len(np.unique(ix%dSize)) == 1:
                mask = np.array([i != v for i in range(dSize)])
                x = (ix%dSize)[0]
                for y in range(dSize):
                    index = y*dSize + x
                    if index not in nInfo._blockNeighbors[b]:
                        bitboard[index] = np.logical_and(mask, bitboard[index])
                blockboard = to_blockboard(dSize, bitboard, nInfo._blockNeighbors)
    return count - np.sum(bitboard)

def box_line_redux(puzzle, dSize, nInfo, bitboard):
    count = np.sum(bitboard)
    bw, bh = block_shape(dSize)
    
    rowboard = to_rowboard(dSize, bitboard)
    for y in range(dSize):
        for v in range(dSize):
            ix = np.where(rowboard[y, :, v])[0]
            if len(np.unique(ix//bw)) == 1:
                mask = np.array([i != v for i in range(dSize)])
                b = y//bh * bh + ix[0]//bw
                for n in range(dSize):
                    index = nInfo._blockNeighbors[b, n]
                    if index not in [y*dSize + x for x in ix]:
                        bitboard[index] =  np.logical_and(mask, bitboard[index])

    columnboard = to_columnboard(dSize, bitboard)
    for x in range(dSize):
        for v in range(dSize):
            ix = np.where(rowboard[x, :, v])[0]
            if len(np.unique(ix//bh)) == 1:
                mask = np.array([i != v for i in range(dSize)])
                b = ix[0]//bh * bh + x//bw
                for n in range(dSize):
                    index = nInfo._blockNeighbors[b, n]
                    if index not in [y*dSize + x for y in ix]:
                        bitboard[index] =  np.logical_and(mask, bitboard[index])
    
    return count - np.sum(bitboard)

#Algorithms TODO:
#Box Line Reduction
    #If, within a line, a specific digit can only be in one block
    #that digit cannot be in that box for any other line
################################################################
#X-Wing
#Coloring
#Y-Wing
#Rectangle Elimination
#Swordfish
#XYZ-Wing
#BUG

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

SOLVE_METHODS = [naked_singles, hidden_singles, naked_tuples, hidden_tuples, pointing_digits]
#naked_singles, hidden_singles, naked_tuples, hidden_tuples, pointing_digits, box_line_redux

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
    print ("Sudoku Solver Demo")

    # Easy and Medium puzzles: courtesy of Sudoku Universe Game]
    # Difficult Named puzzles: courtesy of sudokuwiki.org

    puzzles = [
("Easy",
'000000000000003085001020000000507000004000100090000000500000073002010000000040009'
 ),
("Medium",
'100070009008096300050000020010000000940060072000000040030000080004720100200050003'
),
("Escargot",
"100007090030020008009600500005300900010080002600004000300000010041000007007000300"
),
("Steering Wheel",
"000102000060000070008000900400000003050007000200080001009000805070000060000304000"
),
("Arto Inkala",
"800000000003600000070090200050007000000045700000100030001000068008500010090000400"
)]
    dSize = 0
    for puzzleName, clueString in puzzles:
        start = time.time()
        print ("Puzzle", puzzleName)
        print (clueString)
        puzzle = puzzle_from_string(clueString)
        if dSize != digit_size(puzzle):
            dSize = digit_size(puzzle)
            nInfo = NeighborInfo(dSize)
        bboard = iter_solve(puzzle, dSize, nInfo)
        if not valid_solution(clueString, puzzle):
            print("Partial solution")
        print(puzzle)
        print("Run time:", time.time()-start)
        print ("="*80)
