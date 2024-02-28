import numpy as np
import itertools

def to_rowboard(dSize, bitboard):
    return np.array([bitboard[i*dSize:(i+1)*dSize,:] for i in range(dSize)])


bitboard = np.array([[True, True, False],
                     [True, True, False],
                     [True, True, True],
                     [True, False, True],
                     [True, True, True],
                     [True, False, True],
                     [False, False, False],
                     [True, True, True],
                     [False, True, True]])

dSize = 3
tupleSize = 2
rowboard = to_rowboard(dSize, bitboard)
for y in range(dSize):
    for t in itertools.combinations(range(dSize), tupleSize):
        union = np.any(tuple(rowboard[y,x] for x in t), axis = 0)
        #Make sure you're only unionizing living cells
        if np.sum(union) == tupleSize and 0 not in [np.sum(rowboard[y,x]) for x in t]:
            print(y, t)
            mask = np.logical_not(union)
            for x in range(dSize):
                if x not in t:
                    index = y*dSize + x
                    bitboard[index] = np.logical_and(mask, bitboard[index])
            rowboard = to_rowboard(dSize, bitboard)
print(rowboard)


### Find naked tuples
### TODO: Adapt this to idenitfy naked tuples of many sizes
def naked_tuples(puzzle, dSize, nInfo, bitboard):
    count = np.sum(bitboard)

    maxSize = dSize//2 + 1

    for tupleSize in range(2, maxSize):
        rowboard = to_rowboard(dSize, bitboard)
        for y in range(dSize):
            for t in itertools.combinations(range(dSize), tupleSize):
                intersection = np.any(tuple(rowboard[y, x] for x in t), axis = 0)
                if np.sum(intersection) == tupleSize:
                    mask = np.logical_not(intersection)
                    for x in range(dSize):
                        if x not in t:
                            bitboard[y*dSize+x] = np.logical_and(mask,bitboard[y*dSize+x])
                rowboard = to_rowboard(dSize, bitboard)

        columnboard = to_columnboard(dSize, bitboard)
        for x in range(dSize):
            for t in itertools.combinations(range(dSize), tupleSize):
                intersection = np.any(tuple(columnboard[x, y] for y in t), axis = 0)
                if np.sum(intersection) == tupleSize:
                    mask = np.logical_not(intersection)
                    for y in range(dSize):
                        if y not in t:
                            bitboard[y*dSize+x] = np.logical_and(mask,bitboard[y*dSize+x])
                columnboard = to_columnboard(dSize, bitboard)

        blockboard = to_blockboard(dSize, bitboard, nInfo._blockNeighbors)
        for b in range(dSize):
            for t in itertools.combinations(range(dSize), tupleSize):
                intersection = np.any(tuple(blockboard[b, n] for n in t), axis = 0)
                if np.sum(intersection) == tupleSize:
                    mask = np.logical_not(intersection)
                    for n in range(dSize):
                        if n not in t:
                            index = nInfo._blockNeighbors[b, n]
                            bitboard[index] = np.logical_and(mask, bitboard[index])
                blockboard = to_blockboard(dSize, bitboard, nInfo._blockNeighbors)

        if np.sum(bitboard) < count:
            return count - np.sum(bitboard)
            
    return count - np.sum(bitboard)


### Find hidden pairs
### TODO: Adapt this to idenitfy hidden tuples of many sizes
##def hidden_pairs(puzzle, dSize, nInfo, bitboard):
##    count = np.sum(bitboard)
##    
##    rowboard = to_rowboard(dSize, bitboard)
##    for y in range(dSize):
##        for u, v in itertools.combinations(range(dSize), 2):
##            if np.array_equal(rowboard[y,:,u], rowboard[y,:,v]) and np.sum(rowboard[y,:,u]) == 2:
##                mask = np.array([i in (u,v) for i in range(dSize)])
##                for x in range(dSize):
##                    if rowboard[y,x,u]:
##                        bitboard[y*dSize+x] = np.logical_and(mask,bitboard[y*dSize+x])
##            rowboard = to_rowboard(dSize, bitboard)
##
##    columnboard = to_columnboard(dSize, bitboard)
##    for x in range(dSize):
##        for u, v in itertools.combinations(range(dSize), 2):
##            if np.array_equal(columnboard[x,:,u], columnboard[x,:,v]) and np.sum(columnboard[x,:,u]) == 2:
##                mask = np.array([i in (u,v) for i in range(dSize)])
##                for y in range(dSize):
##                    if columnboard[x,y,u]:
##                        bitboard[y*dSize+x] = np.logical_and(mask,bitboard[y*dSize+x])
##                columnboard = to_columnboard(dSize, bitboard)
##
##    blockboard = to_blockboard(dSize, bitboard, nInfo._blockNeighbors)
##    for b in range(dSize):
##        for u, v in itertools.combinations(range(dSize), 2):
##            if np.array_equal(blockboard[b,:,u], blockboard[b,:,v]) and np.sum(blockboard[b,:,u]) == 2:
##                mask = np.array([i in (u,v) for i in range(dSize)])
##                for n in range(dSize):
##                    if blockboard[b,n,u]:
##                        index = nInfo._blockNeighbors[b, n]
##                        bitboard[index] = np.logical_and(mask, bitboard[index])
##                blockboard = to_blockboard(dSize, bitboard, nInfo._blockNeighbors) 
##            
##    return count - np.sum(bitboard)


