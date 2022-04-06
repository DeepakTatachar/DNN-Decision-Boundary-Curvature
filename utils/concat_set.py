import torch.utils.data as data

class ConcatSet(data.Dataset):
    def __init__(self,
                 datasets,
                 indices):
        super(ConcatSet, self).__init__()
        self.datasets = datasets
        self.split = [0]
        self.indices = indices
        self.length = 0

        for idx, _ in enumerate(datasets):
            self.length += len(indices[idx])
            self.split.append(self.length)


    def __len__(self):
        return self.length

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        set_index = 0
        while set_index < len(self.datasets):
            

            if self.split[set_index] <= index < self.split[set_index+1]:
                break

            set_index = set_index + 1
        
        index = index - self.split[set_index]
        index_in_dataset = self.indices[set_index][index]
        return self.datasets[set_index][index_in_dataset]