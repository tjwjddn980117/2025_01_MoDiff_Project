import torch
import torch.nn
import numpy as np
import os
import os.path
import nibabel
import torch as th
from skimage import io, transform
import random
from matplotlib import pyplot as plt

class MS_MRI(torch.utils.data.Dataset):
    def __init__(self, directory):
        '''
        directory is expected to contain some folder structure:
                  if some subfolder contains only files, all of these
                  files are assumed to have a name like
                  brats_train_001_XXX_123_w.nii.gz
                  where XXX is one of t1, t1ce, t2, flair, seg
                  we assume these five files belong to the same image
                  seg is supposed to contain the segmentation
        '''
        super().__init__()
        self.directory = os.path.expanduser(directory)
        
        self.seqtypes = ['image-flair', 'image-mprage', 'image-pd', 'image-t2', 'label1', 'label2']

        self.seqtypes_set = set(self.seqtypes)
        self.database = []
        for root, dirs, files in os.walk(self.directory):
            # if there are no subdirs, we have data
            if not dirs:
                files.sort()
                datapoint = dict()
                # extract all files as channels
                for f in files:
                    seqtype = f.split('_')[0]
                    datapoint[seqtype] = os.path.join(root, f)
                    #print(datapoint)
                assert set(datapoint.keys()) == self.seqtypes_set, \
                    f'datapoint is incomplete, keys are {datapoint.keys()}'
                self.database.append(datapoint)
                

    def __getitem__(self, x):
        out = []
        filedict = self.database[x]
        for seqtype in self.seqtypes:
            img = io.imread(filedict[seqtype])
            img = transform.resize(img, (128, 128))
            if not seqtype == 'label1' and not seqtype == 'label2':
                img = img / 255
            path=filedict[seqtype] # slice_ID = path[0].split("/", -1)[2]
            out.append(torch.tensor(img))
        out = torch.stack(out)
        
        flair = torch.unsqueeze(out[0], 0)
        mprage = torch.unsqueeze(out[1], 0)
        pd = torch.unsqueeze(out[2], 0)
        t2 = torch.unsqueeze(out[3], 0)
    
        image = torch.cat((flair, mprage, pd, t2), 0)
        
        label = out[random.randint(4, 5)]
        label = torch.unsqueeze(label, 0)
        
        path = path.replace('\\','/')

        return image.type(torch.FloatTensor), label.type(torch.FloatTensor), path

    def __len__(self):
        return len(self.database)