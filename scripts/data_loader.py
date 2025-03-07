import pickle
import torch
import torch.utils.data as Data
import numpy as np


class StrToBytes:
    def __init__(self, fileobj):
        self.fileobj = fileobj

    def read(self, size):
        return self.fileobj.read(size).encode()

    def readline(self, size=-1):
        return self.fileobj.readline(size).encode()


def binding_vec_pos(bs_str, N):
    if bs_str == 'NoBinding':
        print('Error! This record is positive.')
        return None
    if bs_str == '-99999':
        bs_vec = np.zeros(N)
        bs_vec.fill(0)
        return bs_vec
    else:
        bs_list = [int(x) for x in bs_str.split(',')]
        bs_list = [x for x in bs_list if x < N]
        bs_vec = np.zeros(N)
        bs_vec[bs_list] = 1

        return bs_vec


def binding_vec_neg(bs_str, N):
    if bs_str != 'NoBinding':
        print('Error! This record is negative.')
        return None
    else:
        bs_vec = np.zeros(N)
        return bs_vec


# pad or cut all protein sequence mask to same length
def get_mask(protein_seq, pad_seq_len):
    if len(protein_seq) <= pad_seq_len:
        a = np.zeros(pad_seq_len)
        a[:len(protein_seq)] = 1
    else:
        cut_protein_seq = protein_seq[:pad_seq_len]
        a = np.zeros(pad_seq_len)
        a[:len(cut_protein_seq)] = 1
    return a


# pad or cut all sequences to same length
def get_same_len(protein_seq, pad_seq_len):
    if len(protein_seq) <= pad_seq_len:
        # a = protein_seq + (pad_seq_len - len(protein_seq)) * '0'
        a = protein_seq + (pad_seq_len - len(protein_seq)) * '<pad>'
    else:
        cut_protein_seq = protein_seq[:pad_seq_len]
        a = cut_protein_seq
    return a

def get_dataloader(dic_list, data_index_list, model_mode, shuf, config):
    pad_pep_len = config.pad_pep_len
    pad_seq_len = config.pad_prot_len
    model_mode = config.model_mode
    print('loading features:')
    protein_feature_dict = dic_list[0]
    peptide_feature_dict = dic_list[1]
    protein_ss_feature_dict = dic_list[2]
    peptide_ss_feature_dict = dic_list[3]
    protein_2_feature_dict = dic_list[4]
    peptide_2_feature_dict = dic_list[5]
    protein_dense_feature_dict = dic_list[6]
    peptide_dense_feature_dict = dic_list[7]

    datafile = "../data/Dataset_all_balanced_new_2"
    X_pep, X_p, X_SS_pep, X_SS_p, X_2_pep, X_2_p = [], [], [], [], [], []
    X_dense_pep, X_dense_p = [], []
    pep_sequence, prot_sequence, Y = [], [], []
    X_pep_mask, X_bs_flag = [], []
    X_prot_mask, X_prot_bs_flag = [], []
    labels, X_bs, X_prot_bs = [], [], []

    # count = 0
    with open(datafile) as f:
        lines = f.readlines()[1:]
        data_list = [lines[i] for i in data_index_list]

        for line in data_list:
            # print(line)
            # count += 1
            seq, peptide, peptide_ss, seq_ss, label, pep_bs, prot_bs = line.strip().split('\t')
            if int(label) == 1:
                pep_bs_vec = binding_vec_pos(pep_bs, pad_pep_len)
                prot_bs_vec = binding_vec_pos(prot_bs, pad_seq_len)
                if pep_bs == '-99999':
                    flag = 0.0
                else:
                    flag = 1.0
                if prot_bs == '-99999':
                    flag_prot = 0.0
                else:
                    flag_prot = 1.0
            if int(label) == 0:
                flag = 0.0
                flag_prot = 0.0
                pep_bs_vec = binding_vec_neg(pep_bs, pad_pep_len)
                prot_bs_vec = binding_vec_neg(prot_bs, pad_seq_len)
            # flag = 1.0  # For prediction, set flag==1 to generate binding sites
            X_pep_mask.append(get_mask(peptide, pad_pep_len))
            X_prot_mask.append(get_mask(seq, pad_seq_len))
            X_bs_flag.append(flag)
            X_prot_bs_flag.append(flag_prot)
            X_bs.append(pep_bs_vec)
            X_prot_bs.append(prot_bs_vec)
            labels.append(label)


            pep_sequence.append(get_same_len(peptide, pad_pep_len))
            prot_sequence.append(get_same_len(seq, pad_seq_len))
            X_pep.append(peptide_feature_dict[peptide])
            X_p.append(protein_feature_dict[seq])
            X_SS_pep.append(peptide_ss_feature_dict[peptide_ss])
            X_SS_p.append(protein_ss_feature_dict[seq_ss])
            X_2_pep.append(peptide_2_feature_dict[peptide])
            X_2_p.append(protein_2_feature_dict[seq])
            X_dense_pep.append(peptide_dense_feature_dict[peptide])
            X_dense_p.append(protein_dense_feature_dict[seq])


    X_pep = np.array(X_pep)
    X_p = np.array(X_p)
    X_SS_pep = np.array(X_SS_pep)
    X_SS_p = np.array(X_SS_p)
    X_2_pep = np.array(X_2_pep)
    X_2_p = np.array(X_2_p)
    X_dense_pep = np.array(X_dense_pep)
    X_dense_p = np.array(X_dense_p)


    X_pep_mask = np.array(X_pep_mask)
    X_bs_flag = np.array(X_bs_flag)
    X_bs = np.array(X_bs)
    X_prot_mask = np.array(X_prot_mask)
    X_prot_bs_flag = np.array(X_prot_bs_flag)
    X_prot_bs = np.array(X_prot_bs)
    labels = np.array(labels)

    pep_sequence = np.array(pep_sequence)
    prot_sequence = np.array(prot_sequence)
    train_idx = list(range(X_pep.shape[0]))
    np.random.shuffle(train_idx)
    X_pep = X_pep[train_idx]
    X_p = X_p[train_idx]
    X_SS_pep = X_SS_pep[train_idx]
    X_SS_p = X_SS_p[train_idx]
    X_2_pep = X_2_pep[train_idx]
    X_2_p = X_2_p[train_idx]
    X_dense_pep = X_dense_pep[train_idx]
    X_dense_p = X_dense_p[train_idx]


    X_pep_mask = X_pep_mask[train_idx]
    X_bs_flag = X_bs_flag[train_idx]
    X_bs = X_bs[train_idx]
    X_prot_mask = X_prot_mask[train_idx]
    X_prot_bs_flag = X_prot_bs_flag[train_idx]
    X_prot_bs = X_prot_bs[train_idx]
    labels = labels[train_idx]
    pep_sequence = pep_sequence[train_idx]
    prot_sequence = prot_sequence[train_idx]
    print('features loading over and start constructing dataloader')
    if model_mode == 1:
        data = [X_pep, X_p, X_SS_pep, X_SS_p, X_2_pep, X_2_p, X_dense_pep, X_dense_p, pep_sequence, prot_sequence,
                labels]
        return construct_dataset(data, config, shuf)
    else:
        data = [X_pep, X_p, X_SS_pep, X_SS_p, X_2_pep, X_2_p, X_dense_pep, X_dense_p, pep_sequence, prot_sequence,
                X_pep_mask, X_bs_flag, X_bs, labels, X_prot_mask, X_prot_bs_flag, X_prot_bs]
        return construct_dataset(data, config, shuf)



def construct_dataset(data, config, shuf):
    pad_pep_len = config.pad_pep_len
    pad_seq_len = config.pad_prot_len
    model_mode = config.model_mode
    cuda = config.cuda
    batch_size = config.batch_size

    if model_mode == 1:
        X_pep, X_p, X_SS_pep, X_SS_p, X_2_pep, X_2_p, X_dense_pep, X_dense_p, labels = torch.cuda.LongTensor(
            data[0].astype(float)), torch.cuda.LongTensor(data[1].astype(float)), torch.cuda.LongTensor(
            data[2].astype(float)), torch.cuda.LongTensor(data[3].astype(float)), torch.cuda.LongTensor(
            data[4].astype(float)), torch.cuda.LongTensor(data[5].astype(float)), torch.tensor(data[6].astype(float),
                                                                                               dtype=torch.float).cuda(), torch.tensor(
            data[7].astype(float), dtype=torch.float).cuda(), torch.cuda.LongTensor(data[10].astype(float))
    else:
        X_pep, X_p, X_SS_pep, X_SS_p, X_2_pep, X_2_p, X_dense_pep, X_dense_p, pep_sequence, prot_sequence, X_pep_mask, X_bs_flag, X_bs, labels, X_prot_mask, X_prot_bs_flag, X_prot_bs= torch.LongTensor(
            data[0].astype(float)), torch.LongTensor(data[1].astype(float)), torch.LongTensor(
            data[2].astype(float)), torch.LongTensor(data[3].astype(float)), torch.LongTensor(
            data[4].astype(float)), torch.LongTensor(data[5].astype(float)), torch.tensor(
            data[6].astype(float), dtype=torch.float), torch.tensor(
            data[7].astype(float), dtype=torch.float), data[8], data[9], torch.LongTensor(
            data[10].astype(float)), torch.LongTensor(data[11].astype(float)), torch.LongTensor(
            data[12].astype(float)), torch.LongTensor(data[13].astype(float)), torch.LongTensor(
            data[14].astype(float)), torch.LongTensor(data[15].astype(float)), torch.LongTensor(
            data[16].astype(float))

    print('-' * 20, '[construct_dataset]: check data shape', '-' * 20)
    print('labels:', labels.shape)  # [num_sequences, seq_len]

    if model_mode == 1:
        data_loader = Data.DataLoader(
            MyDataSet(X_pep, X_p, X_SS_pep, X_SS_p, X_2_pep, X_2_p, X_dense_pep, X_dense_p, labels),
            batch_size=batch_size,
            shuffle=shuf,
            drop_last=False)
    else:
        data_loader = Data.DataLoader(
            MyDataSet1(X_pep, X_p, X_SS_pep, X_SS_p, X_2_pep, X_2_p, X_dense_pep, X_dense_p, pep_sequence,
                       prot_sequence, X_pep_mask, X_bs_flag, X_bs, labels, X_prot_mask, X_prot_bs_flag, X_prot_bs),
            batch_size=batch_size,
            shuffle=shuf,
            drop_last=False)
    print('len(data_loader)', len(data_loader))
    return data_loader


class MyDataSet(Data.Dataset):
    def __init__(self, X_pep, X_p, X_SS_pep, X_SS_p, X_2_pep, X_2_p, X_dense_pep, X_dense_p, labels):
        self.X_pep = X_pep
        self.X_p = X_p
        self.X_SS_pep = X_SS_pep
        self.X_SS_p = X_SS_p
        self.X_2_pep = X_2_pep
        self.X_2_p = X_2_p
        self.X_dense_pep = X_dense_pep
        self.X_dense_p = X_dense_p
        self.labels = labels

    def __len__(self):
        return len(self.X_pep)

    def __getitem__(self, idx):
        return self.X_pep[idx], self.X_p[idx], self.X_SS_pep[idx], self.X_SS_p[idx], self.X_2_pep[idx], self.X_2_p[idx], \
            self.X_dense_pep[idx], self.X_dense_p[idx], self.labels[idx]


class MyDataSet1(Data.Dataset):
    def __init__(self, X_pep, X_p, X_SS_pep, X_SS_p, X_2_pep, X_2_p, X_dense_pep, X_dense_p, pep_sequence,
                 prot_sequence, X_pep_mask, X_bs_flag, X_bs, labels, X_prot_mask, X_prot_bs_flag, X_prot_bs):
        self.X_pep = X_pep
        self.X_p = X_p
        self.X_SS_pep = X_SS_pep
        self.X_SS_p = X_SS_p
        self.X_2_pep = X_2_pep
        self.X_2_p = X_2_p
        self.X_dense_pep = X_dense_pep
        self.X_dense_p = X_dense_p
        self.pep_sequence = pep_sequence
        self.prot_sequence = prot_sequence
        self.X_pep_mask = X_pep_mask
        self.X_bs_flag = X_bs_flag
        self.X_bs = X_bs
        self.labels = labels
        self.X_prot_mask = X_prot_mask
        self.X_prot_bs_flag = X_prot_bs_flag
        self.X_prot_bs = X_prot_bs

    def __len__(self):
        return len(self.X_pep)

    def __getitem__(self, idx):
        return self.X_pep[idx], self.X_p[idx], self.X_SS_pep[idx], self.X_SS_p[idx], self.X_2_pep[idx], self.X_2_p[idx], \
            self.X_dense_pep[idx], self.X_dense_p[idx], self.pep_sequence[idx], self.prot_sequence[idx], \
            self.X_pep_mask[
                idx], self.X_bs_flag[idx], self.X_bs[idx], self.labels[idx], self.X_prot_mask[idx], self.X_prot_bs_flag[
            idx], self.X_prot_bs[idx]


def load_data(config, k):
    setting = config.setting
    clu_thre = config.clu_thre
    model_mode = config.model_mode
    path_data_index_list = "../data/clu_thre_" + str(clu_thre) + "/" + str(setting) + "_balanced"
    list_all = torch.load(path_data_index_list, encoding='latin1')
    train_idx_list, valid_idx_list, test_idx_list = list_all[0], list_all[1], list_all[2]
    train_loader, valid_loader, test_loader = [], [], []

    with open('../data_feature_dict/protein_feature_dict', 'r', encoding='utf-8') as f:
        protein_feature_dict = pickle.load(StrToBytes(f), encoding='iso-8859-1')
    with open('../data_feature_dict/peptide_feature_dict', 'r', encoding='utf-8') as f:
        peptide_feature_dict = pickle.load(StrToBytes(f), encoding='iso-8859-1')
    with open('../data_feature_dict/protein_ss_feature_dict', 'r', encoding='utf-8') as f:
        protein_ss_feature_dict = pickle.load(StrToBytes(f), encoding='iso-8859-1')
    with open('../data_feature_dict/peptide_ss_feature_dict', 'r', encoding='utf-8') as f:
        peptide_ss_feature_dict = pickle.load(StrToBytes(f), encoding='iso-8859-1')
    with open('../data_feature_dict/protein_2_feature_dict', 'r', encoding='utf-8') as f:
        protein_2_feature_dict = pickle.load(StrToBytes(f), encoding='iso-8859-1')
    with open('../data_feature_dict/peptide_2_feature_dict', 'r', encoding='utf-8') as f:
        peptide_2_feature_dict = pickle.load(StrToBytes(f), encoding='iso-8859-1')
    with open('../data_feature_dict/protein_dense_feature_dict', 'r', encoding='utf-8') as f:
        protein_dense_feature_dict = pickle.load(StrToBytes(f), encoding='iso-8859-1')
    with open('../data_feature_dict/peptide_dense_feature_dict', 'r', encoding='utf-8') as f:
        peptide_dense_feature_dict = pickle.load(StrToBytes(f), encoding='iso-8859-1')

    dic_list = [protein_feature_dict, peptide_feature_dict, protein_ss_feature_dict, peptide_ss_feature_dict,
                protein_2_feature_dict, peptide_2_feature_dict, protein_dense_feature_dict, peptide_dense_feature_dict]
    for i, data_index_list in enumerate(train_idx_list):
        config.batch_size = 128
        if i == k:
            shuf = True
            data_loader = get_dataloader(dic_list, data_index_list, model_mode, shuf, config)
            train_loader.append(data_loader)
    for i, data_index_list in enumerate(valid_idx_list):
        if i == k:
            shuf = False
            data_loader = get_dataloader(dic_list, data_index_list, model_mode, shuf, config)
            valid_loader.append(data_loader)
    for i, data_index_list in enumerate(test_idx_list):
        config.batch_size = 1
        if i == k:
            shuf = False
            data_loader = get_dataloader(dic_list, data_index_list, model_mode, shuf, config)
            test_loader.append(data_loader)

    return train_loader, valid_loader, test_loader
