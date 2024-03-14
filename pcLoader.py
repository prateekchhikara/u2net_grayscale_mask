import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.image_files = os.listdir(img_dir)
        self.mask_files = os.listdir(mask_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_files[idx])
        mask_path = img_path.replace("images", "annotations").replace(".jpg", ".png")

        # img_path = "/Users/prateekchhikara/Downloads/ADE20K_Dataset/ADEChallengeData2016/images/validation/ADE_val_00001154.jpg"
        # mask_path = "/Users/prateekchhikara/Downloads/ADE20K_Dataset/ADEChallengeData2016/annotations/validation/ADE_val_00001154.png"

        # print("Image path:", img_path)
        # print("Mask path:", mask_path)

        # Read image and mask
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()
        # mask = torch.from_numpy(np.array(mask)).float()

        

        # Apply transformations
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        mask = mask *255

        

        mask_replaced = torch.zeros_like(mask)  # Create a tensor of zeros with the same shape as the original mask
        # Replace pixel values according to the specified mapping
        mask_replaced[mask == 1] = 1
        mask_replaced[mask == 4] = 2
        mask_replaced[mask == 6] = 3
        mask_replaced[mask == 9] = 4
        # For other values, set them to 0
        mask_replaced[(mask != 1) & (mask != 4) & (mask != 6) & (mask != 9)] = 0

        # import pdb; pdb.set_trace()
            

        # # Background mask
        # background_mask = torch.sum(mask_classes, dim=0) == 0
        # print(background_mask)
        # mask_classes[4] = background_mask
        

        # Reshape mask tensor
        return image, mask_replaced

# class CustomDataset(Dataset):
#     def __init__(self, img_dir, mask_dir, transform=None):
#         self.img_dir = img_dir
#         self.mask_dir = mask_dir
#         self.transform = transform

#         self.image_files = os.listdir(img_dir)
#         self.mask_files = os.listdir(mask_dir)

#     def __len__(self):
#         return len(self.image_files)

#     def __getitem__(self, idx):
#         img_path = os.path.join(self.img_dir, self.image_files[idx])

#         mask_path = img_path.replace("images", "annotations").replace(".jpg", ".png")

#         # img_path = "/Users/prateekchhikara/Downloads/ADE20K_Dataset/ADEChallengeData2016/images/validation/ADE_val_00001154.jpg"
#         # mask_path = "/Users/prateekchhikara/Downloads/ADE20K_Dataset/ADEChallengeData2016/annotations/validation/ADE_val_00001154.png"

#         print("Image path:", img_path)
#         print("Mask path:", mask_path)

#         # Read image and mask
#         image = Image.open(img_path).convert("RGB")
#         mask = Image.open(mask_path).convert("L")

#         image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()
#         mask = torch.from_numpy(np.array(mask)).float()

#         # Read image and mask
#         image = Image.open(img_path).convert("RGB")
#         mask = Image.open(mask_path).convert("L")

#         # Apply transformations
#         if self.transform:
#             image = self.transform(image)
#             mask = self.transform(mask)

#         # Extract classes and background mask
#         mask_classes = torch.zeros(4, mask.shape[1], mask.shape[2], dtype=torch.float32)
#         mask_background = torch.ones(mask.shape[1], mask.shape[2], dtype=torch.float32)

#         for i, class_value in enumerate([1, 4, 6, 9]):
#             mask_classes[i] = (mask[0]*255 == class_value).int()

            

#         # # Background mask
#         # background_mask = torch.sum(mask_classes, dim=0) == 0
#         # print(background_mask)
#         # mask_classes[4] = background_mask
        

#         # Reshape mask tensor
#         mask_classes = mask_classes.unsqueeze(1)
#         return image, mask_classes
        

def returnDataLoader():
    img_dir = "/Users/prateekchhikara/Downloads/ADE20K_Dataset/ADEChallengeData2016/images/validation"
    mask_dir = "/Users/prateekchhikara/Downloads/ADE20K_Dataset/ADEChallengeData2016/annotations/validation"

    transform = transforms.Compose([
        transforms.Resize((480, 480)),
        transforms.ToTensor(),
    ])

    dataset = CustomDataset(img_dir, mask_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    return dataloader



def returnDataLoaderCheck():
    # Example usage:
    img_dir = "/Users/prateekchhikara/Downloads/ADE20K_Dataset/ADEChallengeData2016/images/validation"
    mask_dir = "/Users/prateekchhikara/Downloads/ADE20K_Dataset/ADEChallengeData2016/annotations/validation"

    transform = transforms.Compose([
        # transforms.Resize((480, 480)),
        transforms.ToTensor(),
    ])

    dataset = CustomDataset(img_dir, mask_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

    # Iterate through the dataloader
    for images, masks in dataloader:
        print("Image shape:", images.shape)
        print("Mask shape:", masks.shape)
        # import pdb; pdb.set_trace() 

        images = images[0]
        masks = masks[0]

        import pdb; pdb.set_trace()

        from PIL import Image

        # Save the image using Pillow
        image_pil = transforms.ToPILImage()(images)
        image_pil.save("image.png")



        # Save each mask separately using Pillow
        for i in range(5):
            K = masks[i][0]
            mask_pil = transforms.ToPILImage()(K)
            # multiply by 255 to get the original mask back
            import pdb; pdb.set_trace()

            mask_pil.save(f"mask_{i}.png")

        

        


        break  # Print only the first batch for demonstration

if __name__ == "__main__":
    import os
    # set current working directory
    os.chdir("/Users/prateekchhikara/Documents/GitHub/cloth-segmentation")

    returnDataLoaderCheck()
