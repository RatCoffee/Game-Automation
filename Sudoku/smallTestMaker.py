puzzles = [line.strip() for line in open("example.txt", 'r').readlines()]

outfile = open("smallTest.txt", 'w')
for i in range(0, len(puzzles), 50):
    outfile.write(puzzles[i] + '\n')
outfile.close()
