from tqdm import tqdm

n = 1

for i in tqdm(range(100000000)):
    n /= 2
print(n)
