import torch
import torchvision
from dataset import SegmentationDataset
from torch.utils.data import DataLoader


def save_checkpoint(state, filename="my_checkpoint.pt"):
    print("=> Saving the model ...")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print(" ==> Loading the Model ... !!!")
    model.load_state_dict(checkpoint["state_dict"])


def get_loaders(train_img_path, train_mask_path, val_img_path, val_mask_path,
                batch_size, train_transforms, val_transforms, Num_workers, PIN_MEMORY):
    train_data = SegmentationDataset(train_img_path, train_mask_path, train_transforms)
    val_data = SegmentationDataset(val_img_path, val_mask_path, val_transforms)

    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              num_workers=Num_workers,
                              pin_memory=PIN_MEMORY,
                              shuffle=True)
    val_loader = DataLoader(val_data,
                            batch_size=batch_size,
                            num_workers=Num_workers,
                            pin_memory=PIN_MEMORY,
                            shuffle=False)
    return train_loader, val_loader

def chack_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)

            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2*(preds*y).sum()) / ((preds+y).sum()+1e-8)

    print(f" Got {num_correct}/{num_pixels} with accuracy {num_correct/num_pixels*100:.2f}")
    print(f"Dice Score: {dice_score/len(loader)}")
    model.train()


def save_prediction_as_imgs(loader, model, batches=3, folder="saved_images/", device="cuda"):
    model.eval()
    for ind, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds>0.5).float()

        torchvision.utils.save_image(
            preds, f"{folder}/pred_{ind}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/img_{ind}.png")
        model.train()
        if ind == batches:
            break