import numpy as np

n = int(input("Enter number of patterns: "))

X = []
Y = []

print("Enter input values (x):")

for i in range(n):
    x = float(input(f"x{i+1}: "))
    X.append([1, x, x**2])

print("Enter target outputs (+1 or -1):")

for i in range(n):
    Y.append(int(input(f"Target {i+1}: ")))

X = np.array(X)
Y = np.array(Y)

# integer weights
w = np.zeros(3, dtype=int)

lr = 1
converged = False
epoch = 0

print("Initial weights:", w)

while not converged:
    converged = True
    epoch += 1
    print("\nEpoch", epoch)

    for i in range(n):
        x = X[i][1]   # actual input value

        net = np.dot(w, X[i])

        print(f"g({int(x)}) = {int(net)}")

        y_pred = 1 if net >= 0 else -1
   
        if y_pred != Y[i]:
            w = w + lr * Y[i] * X[i]
            converged = False
            print("Updated weights:", w.astype(int))

print("\nTraining Converged")
print("Final Weight Vector:", w.astype(int))
print("Decision function: g(x) = {} + {}x + {}x^2".format(int(w[0]), int(w[1]), int(w[2])))
