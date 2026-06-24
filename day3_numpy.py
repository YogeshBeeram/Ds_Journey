import numpy as np

arr = np.array([10,20,30,40,50])
print("Array:",arr)
print("Shape:", arr.shape)

print(np.mean(arr))
print(np.max(arr))
print(np.min(arr))
print(np.sum(arr))

marks = np.array([
    [85, 90, 78],
    [92, 88, 95],
    [70, 75, 80]])

print(marks)
print("Average per student:", np.mean(marks, axis= 1))
print("Average per subject:", np.mean(marks, axis= 0))

python_list = list(range(1000000))
numpy_array = np.arange(1000000)


import time
start = time.time()
total = sum(python_list)
print(f"\n python list sum time: {time.time() - start:.4f} seconds")

start = time.time()
total = np.sum(numpy_array)
print(f"numpy array sum time: {time.time() - start:.4f} seconds")
print("Numpy is way faster ")