# Find naked tuples
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
                rowboard = to_rowboard(dSize, bitboard)

        columnboard = to_columnboard(dSize, bitboard)
        for x in range(dSize):
            for t in itertools.combinations(range(dSize), tupleSize):
                intersection = np.any(tuple(columnboard[x, y] for y in t), axis = 0)
                if np.sum(intersection) > 0 and np.sum(intersection) <= tupleSize:
                    mask = np.logical_not(intersection)
                    for y in range(dSize):
                        if y not in t:
                            bitboard[y*dSize+x] = np.logical_and(mask,bitboard[y*dSize+x])
                columnboard = to_columnboard(dSize, bitboard)

        blockboard = to_blockboard(dSize, bitboard, nInfo._blockNeighbors)
        for b in range(dSize):
            for t in itertools.combinations(range(dSize), tupleSize):
                intersection = np.any(tuple(blockboard[b, n] for n in t), axis = 0)
                if np.sum(intersection) > 0 and  np.sum(intersection) <= tupleSize:
                    mask = np.logical_not(intersection)
                    for n in range(dSize):
                        if n not in t:
                            index = nInfo._blockNeighbors[b, n]
                            bitboard[index] = np.logical_and(mask, bitboard[index])
                #blockboard = to_blockboard(dSize, bitboard, nInfo._blockNeighbors)

        if np.sum(bitboard) < count:
            return count - np.sum(bitboard)
            
    return count - np.sum(bitboard)