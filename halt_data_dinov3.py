
import random, torch
from PIL import Image
import torchvision.transforms as T
_norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
TTA_TF = [T.Compose([T.Resize((224, 224)), T.ToTensor(), _norm]),
          T.Compose([T.Resize((224, 224)), T.RandomHorizontalFlip(p=1.0), T.ToTensor(), _norm]),
          T.Compose([T.Resize((256, 256)), T.CenterCrop(224), T.ToTensor(), _norm])]
TRAIN_TF = T.Compose([T.RandomResizedCrop(224, scale=(0.5, 1.0), ratio=(0.9, 1.1)), T.RandomHorizontalFlip(0.5), T.RandomVerticalFlip(0.2),
                      T.RandomRotation(30), T.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                      T.ColorJitter(0.3, 0.3, 0.2, 0.1), T.RandomAutocontrast(0.2), T.ToTensor(), _norm])
class ImgViews(torch.utils.data.Dataset):
    def __init__(self, paths, mode, n_aug=48): self.paths, self.mode, self.n_aug = paths, mode, n_aug
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        im = Image.open(self.paths[i]).convert('RGB')
        if self.mode == 'tta': return torch.stack([t(im) for t in TTA_TF])
        random.seed(1000003 + i); torch.manual_seed(1000003 + i)          # deterministic per image, independent of the number of workers
        return torch.stack([TRAIN_TF(im) for _ in range(self.n_aug)])
