

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

# Find hidden pairs
# TODO: Adapt this to idenitfy hidden tuples of many sizes
def hidden_pairs(puzzle, dSize, nInfo, bitboard):
    count = np.sum(bitboard)
    
    rowboard = to_rowboard(dSize, bitboard)
    for y in range(dSize):
        for u, v in itertools.combinations(range(dSize), 2):
            if np.array_equal(rowboard[y,:,u], rowboard[y,:,v]) and np.sum(rowboard[y,:,u]) == 2:
                mask = np.array([i in (u,v) for i in range(dSize)])
                for x in range(dSize):
                    if rowboard[y,x,u]:
                        bitboard[y*dSize+x] = np.logical_and(mask,bitboard[y*dSize+x])
            rowboard = to_rowboard(dSize, bitboard)

    columnboard = to_columnboard(dSize, bitboard)
    for x in range(dSize):
        for u, v in itertools.combinations(range(dSize), 2):
            if np.array_equal(columnboard[x,:,u], columnboard[x,:,v]) and np.sum(columnboard[x,:,u]) == 2:
                mask = np.array([i in (u,v) for i in range(dSize)])
                for y in range(dSize):
                    if columnboard[x,y,u]:
                        bitboard[y*dSize+x] = np.logical_and(mask,bitboard[y*dSize+x])
                columnboard = to_columnboard(dSize, bitboard)

    blockboard = to_blockboard(dSize, bitboard, nInfo._blockNeighbors)
    for b in range(dSize):
        for u, v in itertools.combinations(range(dSize), 2):
            if np.array_equal(blockboard[b,:,u], blockboard[b,:,v]) and np.sum(blockboard[b,:,u]) == 2:
                mask = np.array([i in (u,v) for i in range(dSize)])
                for n in range(dSize):
                    if blockboard[b,n,u]:
                        index = nInfo._blockNeighbors[b, n]
                        bitboard[index] = np.logical_and(mask, bitboard[index])
                blockboard = to_blockboard(dSize, bitboard, nInfo._blockNeighbors) 
            
    return count - np.sum(bitboard)
