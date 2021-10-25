import pickle

with open('./mean_diff_positive' + '.pkl', 'rb') as f:
    read_dic = pickle.load(f)
    print(read_dic)

with open('./mean_bd_real' + '.pkl', 'rb') as f:
    read_dic = pickle.load(f)
    print(read_dic)


