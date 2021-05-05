import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import glob
from scipy.io import loadmat

class SIDDValidDataset(Dataset):
    def __init__(self,path=None):
        self.path = path+'sidd_valid_patches.h5'
        hf = h5py.File(self.path, 'r')
        self.keys = [key for key in hf.keys()]
        hf.close()
    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self,index):
        hf = h5py.File(self.path, 'r')
        data = np.array(hf[self.keys[index]])
        hf.close()
        return (
            torch.tensor(data[0]).float().div(255).permute(2,0,1),
            torch.tensor(data[1]).float().div(255).permute(2,0,1),
        )

class SIDDDataset(Dataset):
    def __init__(self,patches_per_batch=32,patches_per_image=3200,patch_size=80,path=None):
        print('Initializing dataset')
        self.path = path+'/sidd_medium_80_{0}/part{1}.h5'
        self.patches_per_batch = patches_per_batch
        self.patches_per_image = patches_per_image
        self.image_to_batch_ratio = self.patches_per_image//self.patches_per_batch

    def __len__(self):
        return (320*self.patches_per_image)//self.patches_per_batch

    def __getitem__(self,index):
        hf = h5py.File(self.path.format(self.patches_per_image,index//self.image_to_batch_ratio), 'r')
        #except: print(index,self.patches_per_image,self.patches_per_batch)
        data = np.asarray(hf['patches'])
        hf.close()

        #indices = np.random.randint(low=0,high=self.patches_per_image,size=(self.patches_per_batch,))
        start,stop = (index%self.image_to_batch_ratio)*self.patches_per_batch,(index%self.image_to_batch_ratio+1)*self.patches_per_batch
        data = torch.tensor(data[start:stop,:,:,:])
        noisy,clean = data[:,0],data[:,1]
        
        return (
            noisy.float().div(255).permute(0,3,1,2),
            clean.float().div(255).permute(0,3,1,2),
        )

class RENOIRDataset(Dataset):
    def __init__(
        self,
        parent_dir="D:\\malik\\DATASETS\\RENOIR\\",
        cameras = ['Mi3_Aligned','S90_Aligned','T3i_Aligned'],
        camera_info = False
    ):
        parent_dir += "{camera}\\{camera}\\Batch_{batch:03d}\\"
        camera_assign = []
        clean_images = []
        noisy_images = []
        for camera in cameras:
            print("[RENOIR]:",camera)
            for batch_idx in range(1,41):
                parent_dir_now = parent_dir.format(camera=camera,batch=batch_idx)
                clean_images.append( glob.glob(parent_dir_now+"*Reference.bmp")[0])
                noisy_images.append( glob.glob(parent_dir_now+"*Noisy.bmp")[0])
                camera_assign.append(camera)
        
        self.clean_images = clean_images
        self.noisy_images = noisy_images
        self.camera_assign = camera_assign
        self.camera_info = camera_info
    
    def __len__(self): return len(self.clean_images)
    
    def __getitem__(self,index):
        noisy = TF.to_tensor(Image.open(self.noisy_images[index]))
        clean = TF.to_tensor(Image.open(self.clean_images[index]))
        camera = self.camera_assign[index]
        if self.camera_info: return noisy,clean,camera
        else: return noisy,clean

class RINDataset(Dataset):
    def __init__(
        self,
        parent_dir="D:\\malik\\DATASETS\\RIN\\",
        cameras = ['Canon_EOS_5D_Mark3','Nikon_D600','Nikon_D800'],
        camera_info=False,
        crop_size=None,
    ):
        parent_dir += "{camera}\\*\\"
        camera_assign = []
        mat_paths = []
        for camera in cameras:
            print(camera)
            parent_dir_now = parent_dir.format(camera=camera)
            file_list = glob.glob(parent_dir_now+"*.mat")
            mat_paths += file_list
            camera_assign += [camera for _ in range(len(file_list))]
        
        self.mat_paths = mat_paths
        self.camera_assign = camera_assign
        self.camera_info = camera_info
        self.crop_size = crop_size
        
    
    def __len__(self): return len(self.mat_paths)
    
    def _crop(self,image):
        _,W,H = image.shape
        w = W//2; h=H//2; c=self.crop_size//2
        return image[:,(w-c):(w+c),(h-c):(h+c)]
    
    def __getitem__(self,index):
        data = loadmat(self.mat_paths[index])
        clean = TF.to_tensor(data['img_mean']).float()
        noisy = TF.to_tensor(data['img_noisy']).float()
        if self.crop_size is not None:
            noisy = self._crop(noisy)
            clean = self._crop(clean)
        camera = self.camera_assign[index]
        if self.camera_info: return noisy,clean,camera
        else: return noisy,clean