from tqdm import tqdm

import torch
import torch.nn as nn

from dataset import get_dataset

def evaluate_pytorch_model(model_path="weights/mobilev2_model.pth"):
    model = torch.load(model_path)
 
    if isinstance(model, torch.nn.DataParallel):
        print(model)
        model = model.module

    _, val_dataset, _ = get_dataset()
    dataloaders = torch.utils.data.DataLoader(
        val_dataset, batch_size=8, shuffle=True, num_workers=4
    )

    running_corrects = 0.0
    for _, (inputs, labels) in tqdm(enumerate(dataloaders)):
        inputs = inputs.cuda()
        labels = labels.cuda()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)
    
    print(f"PyTorch Model Accuracy : {running_corrects / len(val_dataset) * 100}%")


if __name__ == "__main__":
    evaluate_pytorch_model()
