import torch.utils.data as data
import hickle as hkl
import numpy as np


class MapData(data.Dataset):
    def __init__(self, data_file, source_file, nt, nt_predict=None,
                 output_mode='error', sequence_start_mode='all'):
        self.X = hkl.load(data_file)  # X will be like (n_images, nb_cols, nb_rows, nb_channels)
        self.sources = hkl.load(source_file) # source for each image so when creating sequences can assure that consecutive frames are from same video
        self.nt = nt
        self.nt_predict = nt_predict
        self.sequence_start_mode = sequence_start_mode
        self.output_mode = output_mode

        self.im_shape = self.X[0].shape
        if self.nt_predict != None:
            if self.sequence_start_mode == 'all':  # allow for any possible sequence, starting from any frame
                self.possible_starts = np.array([i for i in range(self.X.shape[0] - self.nt - self.nt_predict) if self.sources[i] == self.sources[i + self.nt - 1 + self.nt_predict]])
            elif self.sequence_start_mode == 'unique':  #create sequences where each unique frame is in at most one sequence
                curr_location = 0
                possible_starts = []
                while curr_location < self.X.shape[0] - self.nt + 1:
                    if self.sources[curr_location] == self.sources[curr_location + self.nt + self.nt_predict - 1]:
                        possible_starts.append(curr_location)
                        curr_location += self.nt
                    else:
                        curr_location += 1
                self.possible_starts = possible_starts

        else:
            if self.sequence_start_mode == 'all':  # allow for any possible sequence, starting from any frame
                self.possible_starts = np.array([i for i in range(self.X.shape[0] - self.nt) if self.sources[i] == self.sources[i + self.nt - 1]])
            elif self.sequence_start_mode == 'unique':  #create sequences where each unique frame is in at most one sequence
                curr_location = 0
                possible_starts = []
                while curr_location < self.X.shape[0] - self.nt + 1:
                    if self.sources[curr_location] == self.sources[curr_location + self.nt - 1]:
                        possible_starts.append(curr_location)
                        curr_location += self.nt
                    else:
                        curr_location += 1
                self.possible_starts = possible_starts

    def __len__(self):
        return len(self.possible_starts)

    def __getitem__(self, index):
        start = self.possible_starts[index]
        if self.nt_predict != None:
            return self.X[start: start + self.nt], self.X[start + self.nt: start + self.nt+self.nt_predict]
        else:
            return self.X[start: start + self.nt]

    def next(self):
        with self.lock:
            current_index = (self.batch_index * self.batch_size) % self.n
            index_array, current_batch_size = next(self.index_generator), self.batch_size
        batch_x = np.zeros((current_batch_size, self.nt) + self.im_shape, np.float16)
        for i, idx in enumerate(index_array):
            idx = self.possible_starts[idx]
            batch_x[i] = self.preprocess(self.X[idx:idx+self.nt])
        if self.output_mode == 'error':  # model outputs errors, so y should be zeros
            batch_y = np.zeros(current_batch_size, np.float16)
        elif self.output_mode == 'prediction':  # output actual pixels
            batch_y = batch_x
        return batch_x, batch_y

    def preprocess(self, X):
        return X.astype(np.float16)

    def create_all(self):
        X_all = np.zeros((self.N_sequences, self.nt) + self.im_shape, np.float16)
        for i, idx in enumerate(self.possible_starts):
            X_all[i] = self.preprocess(self.X[idx:idx+self.nt])
        return X_all

if __name__=='__main__':
    from evaluation import evaluate
    import cv2
    a = '/home/ana/Study/Occupancy_flow/Short_PredNet/data_evidential_grid_splits/double_prong/X_train.hkl'
    b = '/home/ana/Study/Occupancy_flow/Short_PredNet/data_evidential_grid_splits/double_prong/sources_train.hkl'
    dataset = KITTI_Test(a, b, 5, 10)
    for i in range(10):
        input_, test = dataset.__getitem__(i)
        # input_ = np.expand_dims(input_, axis=0)

        img = (input_[0]*255).astype(np.uint8)
        img2 = (test*255).astype(np.uint8)
        print(img.shape)
        print(np.max(img))
        print(np.min(img))
        cv2.imshow('a', img)
        cv2.imshow('b', img2)
        cv2.waitKey(0)
