import numpy as np

# Step 1: Input number of patterns
n = int(input("Enter number of input patterns: "))

# Step 2: Input number of features
d = int(input("Enter number of features: "))

X = []
Y = []

print("\nEnter input patterns:")

# Step 3: Take input patterns
for i in range(n):
    pattern = list(map(float, input(f"Enter pattern {i+1}: ").split()))
    
    # Step 4: Augmentation (add bias term 1)
    pattern = [1] + pattern
    
    X.append(pattern)

# Step 5: Take target outputs
print("\nEnter target outputs (+1 or -1):")
for i in range(n):
    y = int(input(f"Target for pattern {i+1}: "))
    Y.append(y)

X = np.array(X)
Y = np.array(Y)

# Step 6: Initialize weight vector
w = np.zeros(d + 1, dtype=int)

# Step 7: Learning rate
lr = 1

print("\nInitial Weight Vector:", w)

# Step 8: Training until convergence
converged = False
epoch = 0

while not converged:
    converged = True
    epoch += 1
    print(f"\nEpoch {epoch}")
    
    for i in range(n):
        net = np.dot(w, X[i])
        
        if net >= 0:
            y_pred = 1
        else:
            y_pred = -1
        
        # Check misclassification
        if y_pred != Y[i]:
            w = w + lr * Y[i] * X[i]
            converged = False
            print(f"Updated weight: {w.astype(int)}")

# Step 9: Final result
print("\nTraining Converged!")
print("Final Weight Vector:", w.astype(int))
