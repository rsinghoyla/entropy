import os.path
import pickle
from typing import Any, Callable, Optional, Tuple

import numpy as np
from PIL import Image

#from .utils import check_integrity, download_and_extract_archive
from torchvision.datasets.vision import VisionDataset
from torch.utils.data import DataLoader, Dataset # Gives easier dataset managment and creates mini batches
from torchvision.datasets import ImageFolder



#val_dataset = ImageFolder("data/cats-dogs/test_set/test_set/")
#_,test_data,_,test_label = train_test_split(val_dataset.imgs,val_dataset.targets,test_size=0.99,random_state=42)
# ImageLoader Class
import pickle

class ImageLoader(Dataset):
    def __init__(self, dataset, transform=None, bpp_thresh_cat = 100000,bpp_thresh_dog = 100000,page_rank_thresh = None,bpp_thresh2_cat = None,bpp_thresh2_dog = None, sort_on ='size', sort_order = 'rand', proportion = 1.0):
        with open('data/PetImages/train.pkl','rb') as fp:
           self.bpp_dic = pickle.load(fp)
        with open('data/PetImages/page_rank.pkl','rb') as fp:
            self.page_rank_dic = pickle.load(fp)
        
        self.dataset = self.checkChannel(dataset,bpp_thresh_cat,bpp_thresh_dog,page_rank_thresh,bpp_thresh2_cat,bpp_thresh2_dog) # some images are CMYK, Grayscale, check only RGB
        self.sizes = []
        self.pagerank  = []
        print(len(self.dataset))
        for d in self.dataset:
            self.sizes.append(self.bpp_dic[d[0]])
            try:
                self.pagerank.append(self.page_rank_dic[d[0]])
            except KeyError:
                self.pagerank.append(np.mean(list(self.page_rank_dic.values())))
        self.dataset = self.select(sort_order, proportion, sort_on)
        self.transform = transform


    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, item):
        #print(self.dataset[item][0])
        image = Image.open(self.dataset[item][0])
        classCategory = self.dataset[item][1]
        if self.transform:
            image = self.transform(image)
        return image, classCategory
        



    def select(self,sort_order = 'rand', proportion=1.0, sort_on = 'size'):
        num_ = int(proportion *len(self.dataset))
        print("NUM",num_,len(self.sizes))
        if sort_on == 'size':
            use_array = self.sizes
        elif sort_on == 'pagerank':
            use_array = self.pagerank

        _labels = []
        for d in self.dataset:
            _labels.append(d[1])
        if sort_order == 'rand':
            #index = np.random.permutation(len(self.sizes))[:num_]
            index = []
            for l in range(2):
                i = np.where(np.asarray(_labels)==l)[0]
                num_ = int(proportion *len(i))
                index.extend(np.random.permutation(i)[:num_])
        else: 
            index = []
                
            for l in range(2):
                i = np.where(np.asarray(_labels)==l)
                sz = np.asarray(use_array)[i[0]]
                num_ = int(proportion *len(sz))
                if sort_order =='asc':
                    j = np.argsort(sz)[:num_]
                    index.extend(i[0][j])
                elif sort_order == 'desc':
                    j = np.argsort(sz)[::-1][:num_]
                    index.extend(i[0][j])
                if sort_order == 'bal':
                    j = np.argsort(sz)[:int(num_/2)]
                    index.extend(i[0][j])
                    j = np.argsort(sz)[::-1][:int(num_/2)]
                    index.extend(i[0][j])
        data = []    
        for i in index:
            data.append(self.dataset[i])
        #self.stats = np.mean(np.asarray(use_array)[index])
        self.stats = "{:.3f}".format(np.mean(np.asarray(use_array)[index]))+"_"+"{:.3f}".format(np.std(np.asarray(use_array)[index]))
        return data
                    
    def checkChannel(self, dataset,bpp_thresh_cat = 100000,bpp_thresh_dog = 100000,page_rank_thresh = None,bpp_thresh2_cat = None,bpp_thresh2_dog = None):
        datasetRGB = []
        bal = False
        if bpp_thresh2_cat is not None and bpp_thresh2_dog is not None:
            #bpp_thresh2_cat = -1*bpp_thresh2_cat
            #bpp_thresh2_dog = -1*bpp_thresh2_dog
            bal = True
        high = False
        if bpp_thresh_cat < 0 and bpp_thresh_dog <0:
            bpp_thresh_cat = -1*bpp_thresh_cat
            bpp_thresh_dog = -1*bpp_thresh_dog
            high = True
        high2 = False
        prt = page_rank_thresh
        if  page_rank_thresh is not None and page_rank_thresh > 0:
            prt = -1*page_rank_thresh
            high2 = True
        sizes = []
        print(bpp_thresh_cat,bpp_thresh2_cat)
        for index in range(len(dataset)):
            if (Image.open(dataset[index][0]).getbands() == ("R", "G", "B")): # Check Channels
                if bpp_thresh_cat == 100000 or bpp_thresh_dog == 100000:
                    if page_rank_thresh is None:
                        datasetRGB.append(dataset[index])
                        sizes.append(self.bpp_dic[dataset[index][0]])
                    else:
                        if high2:
                            if dataset[index][0] in self.page_rank_dic and self.page_rank_dic[dataset[index][0]] > prt:
                                #print(self.page_rank_dic[dataset[index][0]])
                                datasetRGB.append(dataset[index])
                        else:
                            if dataset[index][0] in self.page_rank_dic and self.page_rank_dic[dataset[index][0]] < prt:
                                datasetRGB.append(dataset[index])
                                
                elif bpp_thresh_cat == 0:
                    if np.random.rand()>bpp_thresh_dog :
                        datasetRGB.append(dataset[index])
                        sizes.append(self.bpp_dic[dataset[index][0]])
                else:
                    #
                    if bal:
                        
                            if ('Cat' in dataset[index][0] and self.bpp_dic[dataset[index][0]]>bpp_thresh_cat  or self.bpp_dic[dataset[index][0]]<bpp_thresh2_cat) or ('Dog' in dataset[index][0] and self.bpp_dic[dataset[index][0]]>bpp_thresh_dog or self.bpp_dic[dataset[index][0]]<bpp_thresh2_dog):
                                datasetRGB.append(dataset[index])
                                sizes.append(self.bpp_dic[dataset[index][0]])
                    else:
                        if high:
                            if ('Cat' in dataset[index][0] and self.bpp_dic[dataset[index][0]]>bpp_thresh_cat) or ('Dog' in dataset[index][0] and self.bpp_dic[dataset[index][0]]>bpp_thresh_dog):
                                datasetRGB.append(dataset[index])
                                sizes.append(self.bpp_dic[dataset[index][0]])
                        else:
                            if ((('Cat' in dataset[index][0] and self.bpp_dic[dataset[index][0]]<bpp_thresh_cat) or ('Dog' in dataset[index][0] and self.bpp_dic[dataset[index][0]]<bpp_thresh_dog))):
                                datasetRGB.append(dataset[index])
                                sizes.append(self.bpp_dic[dataset[index][0]])
        if len(sizes)>0:                
            print('AA',np.histogram(sizes))
        return datasetRGB

class CIFAR10(VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    base_folder = "cifar-10-batches-py"
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = "c58f30108f718f92721af3b95e74349a"
    train_list = [
        ["data_batch_1", "c99cafc152244af753f735de768cd75f"],
        ["data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"],
        ["data_batch_3", "54ebc095f3ab1f0389bbae665268c751"],
        ["data_batch_4", "634d18415352ddfa80567beed471001a"],
        ["data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"],
    ]

    test_list = [
        ["test_batch", "40351d587109b95175f43aff81a1287e"],
    ]
    meta = {
        "filename": "batches.meta",
        "key": "label_names",
        "md5": "5ff9c542aee3614f3951f8cda6e48888",
    }

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)

        self.train = train  # training set or test set

        if download:
            self.download()

        #if not self._check_integrity():
        #    raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.data.append(entry[b"data"].astype('uint8'))
                if b"labels" in entry:
                    self.targets.extend(entry[b"labels"])
                else:
                    self.targets.extend(entry["fine_labels"])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta["filename"])
        #if not check_integrity(path, self.meta["md5"]):
        #    raise RuntimeError("Dataset metadata file not found or corrupted. You can use download=True to download it")
        with open(path, "rb") as infile:
            data = pickle.load(infile, encoding="latin1")
            self.classes = data[self.meta["key"]]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        for filename, md5 in self.train_list + self.test_list:
            fpath = os.path.join(self.root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        #download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"


class CIFAR100(CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """

    base_folder = "cifar-100-python"
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"
    train_list = [
        ["train", "16019d7e3df5f24257cddd939b257f8d"],
    ]

    test_list = [
        ["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"],
    ]
    meta = {
        "filename": "meta",
        "key": "fine_label_names",
        "md5": "7973b15100ade9c7d40fb424638fde48",
    }
