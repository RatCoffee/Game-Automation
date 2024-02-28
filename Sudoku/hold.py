import numpy as np
import itertools

### Find naked tuples
### TODO: Adapt this to idenitfy naked tuples of many sizes
def naked_tuples(puzzle, dSize, nInfo, bitboard):
    count = np.sum(bitboard)
    maxSize = min(dSize//2 + 1, 4)

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


# Find hidden pairs
# TODO: Adapt this to idenitfy hidden tuples of many sizes
def hidden_tuples(puzzle, dSize, nInfo, bitboard):
    count = np.sum(bitboard)
    maxSize = min(dSize//2 + 1, 4)


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


