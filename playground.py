import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from pymatreader import read_mat


def generate_batch(X, Y, batch_size):
    data = []
    for i in range(len(X)):
        data.append([X[i], Y[i]])
    mini_batch = []
    for i in range(batch_size):
        if i % 50 == 0:
            random.shuffle(data)
        mini_batch.append(data.pop(0))
    return np.array(mini_batch)


def main():
    data = read_mat('Data/SwissRollData.mat')
    # print(data)
    Yt_df = pd.DataFrame(data['Yt'])
    Yt_nparr = Yt_df.to_numpy()
    batch = generate_batch(Yt_nparr[0], Yt_nparr[1], 4000)
    x, y = batch[:, 0], batch[:, 1]

    plt.figure()
    plt.title('Training data')
    plt.scatter(x, y, label='data')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()