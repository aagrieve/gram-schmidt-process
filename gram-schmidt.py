import numpy as np

small_matrix = np.array(([[1, 2, 3],[4, 5, 6], [7, 8, 8]]))
large_matrix = np.random.rand(50, 50)
### change below to run
#input_matrix = small_matrix
input_matrix = large_matrix

# DEFINITIONS
###############################################################

def gramSchmidtClassic(A):
    row, col = A.shape
    q = np.zeros((row, col))
    r = np.zeros((row, col))
    for k in range(0, col):
        w = A[:, k]
        for j in range(0, k):
            r[j][k] = np.dot(np.transpose(q[:,j]), w)
        for j in range(0, k):
            w = w - r[j][k] * q[:,j]
        r[k][k] = np.linalg.norm(w)
        q[:, k] = w / r[k][k]

    return [q, r]

def gramSchmidtMod(A):
    row, col = A.shape
    q = np.zeros((row, col))
    r = np.zeros((row, col))
    for k in range(0, col):
        w = A[:, k]
        for j in range(0, k):
            r[j][k] = np.dot(np.transpose(q[:,j]), w)
            w = w - r[j][k] * q[:,j]
        r[k][k] = np.linalg.norm(w)
        q[:, k] = w / r[k][k]
    return [q, r]

def checkDecompDiff(A, q, r):
    C = np.matmul(q, r) - A
    max = np.absolute(C.max())
    min = np.absolute(C.min())
    return min if min > max else max

def checkOrthogDiff(q):
    print(q)
    row, col = q.shape
    Orth = np.matmul(np.transpose(q), q)
    Orth = np.subtract(Orth, np.identity(row))
    max = np.absolute(Orth.max())
    min = np.absolute(Orth.min())
    return min if min > max else max

def compareDecomp(c, m):
    if c < m:
        return "Classic"
    if c > m: 
        return "Modified"
    else:
        return "Same"

def compareOrtho(c, m):
    if c < m:
        return "Classic"
    if c > m: 
        return "Modified"
    else:
        return "Same"

###############################################################

# CLASSIC
###############################################################

qr = gramSchmidtClassic(input_matrix)
print("Classic:")
print("Q:")
print(qr[0])
print("R:")
print(qr[1])
print("Decomp Diff:")
c_DecompDiff = checkDecompDiff(input_matrix, qr[0], qr[1])
print(c_DecompDiff)
print("Ortho Diff:")
c_OrthoDiff = checkOrthogDiff(qr[0])
print(c_OrthoDiff)

###############################################################

# MODIFIED
###############################################################

qrMod = gramSchmidtMod(input_matrix)
print("\nModified")
print("Q:")
print(qrMod[0])
print("R:")
print(qrMod[1])
print("Decomp Diff:")
m_DecompDiff = checkDecompDiff(input_matrix, qrMod[0], qrMod[1])
print(m_DecompDiff)
print("Ortho Diff:")
m_OrthoDiff = checkOrthogDiff(qrMod[0])
print(m_OrthoDiff)

###############################################################

# COMPARE
###############################################################

print("\nBetter Decomp:")
print(compareDecomp(c_DecompDiff, m_DecompDiff))
print("More Orthogonal:")
print(compareOrtho(c_OrthoDiff, m_OrthoDiff))

###############################################################