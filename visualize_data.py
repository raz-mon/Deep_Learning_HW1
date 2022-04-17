from pymatreader import read_mat
import pandas as pd
import matplotlib.pyplot as plt

def visualize_2D_data(path):
    data = read_mat(path)
    # print(data)

    Ct_df = pd.DataFrame(data['Ct'])
    Cv_df = pd.DataFrame(data['Cv'])
    Yt_df = pd.DataFrame(data['Yt'])
    Yv_df = pd.DataFrame(data['Yv'])

    Ct_nparr = Ct_df.to_numpy()
    Cv_nparr = Cv_df.to_numpy()
    Yt_nparr = Yt_df.to_numpy()
    Yv_nparr = Yv_df.to_numpy()

    plt.figure()
    plt.title('Training data')
    plt.scatter(Yt_nparr[0], Yt_nparr[1], label='data')
    plt.legend()


    plt.figure()
    plt.title('Validation data')
    plt.scatter(Yv_nparr[0], Yv_nparr[1], label='data')
    plt.legend()

    plt.show()


visualize_2D_data('Data/SwissRollData.mat')
visualize_2D_data('Data/PeaksData.mat')















