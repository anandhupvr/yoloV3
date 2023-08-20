import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from tqdm import tqdm

from model.darknet import YoloV3
from loss import Yolov3Loss
from utils.data_loader import CocoDataset, VocDataset
import numpy as np
num_class = 80


ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
] 


data_dir = "./data/coco_minitrain_25k"
train_annotation_file = 'train2017.txt'
darknet53_weight_file = "/home/anandhu/Documents/works/yoloV3/weights/darknet53_weights_pytorch.pth"

# VOC config
voc_data_dir = "/home/anandhu/Documents/works/yoloV3/data/VOCdevkit/VOC2007/"
voc_class = 20
num_class = voc_class


def train_fn(train_loader, model, criterion, optimizer, epoch, anchors, device):
    last_loss = 0 # batch loss
    running_loss = 0

    for batch_idx, (x, y) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)):
        x = x.to(device)
        y0, y1, y2 = (y[0].to(device), y[1].to(device), y[2].to(device))
        
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        out = model(x)

        # Compute the loss and its gradients
        loss = ( 
            criterion(out[0], y0, anchors[0])
            + criterion(out[1], y1, anchors[1])
            + criterion(out[2], y2, anchors[2])
           )
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        running_loss += loss.item()

        if (batch_idx % 100 == 99):
            last_loss = running_loss / 100
            print(' batch {} loss: {} '.format(batch_idx + 1, last_loss))            

    return last_loss

def main():

    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    anchors = torch.tensor(ANCHORS).to(device)

    model = YoloV3(num_class).to(device)

    # init darknet53 weights
    model.load_weights("/home/anandhu/Documents/works/yoloV3/darknet53.pth")

    optimizer = optim.Adam(
        model.parameters(), lr=1e-5, weight_decay=1e-4
    )
    criterion = Yolov3Loss()

    # create dataset and data loader
    # dataset = CocoDataset(data_dir, train_annotation_file, ANCHORS)
    dataset = VocDataset(voc_data_dir, voc_class, ANCHORS)

	# Split dataset into train and test subsets
    train_ratio = 0.8
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    test_size = dataset_size - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    num_epochs = 151

    # model.load_weights(darknet53_weight_file)
    running_loss = 0
    last_loss = 0

    for epoch in tqdm(range(num_epochs), desc="Epochs"):

        model.train()

        avg_loss = train_fn(train_loader, model, criterion, optimizer, epoch, anchors, device)

        model.eval()
        running_vloss = 0
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(tqdm(test_loader, leave=False)):
                x = x.to(device)
                y0, y1, y2 = (y[0].to(device), y[1].to(device), y[2].to(device))
                out = model(x)
                vloss = ( 
                    criterion(out[0], y0, anchors[0])
                    + criterion(out[1], y1, anchors[1])
                    + criterion(out[2], y2, anchors[2])
                   )                

                running_vloss += vloss
        avg_vloss = running_vloss / (epoch + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))


        if epoch + 1 in [25, 50, 75, 100, 150]:
            checkpoint = {
                'epoch':epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': running_loss
            }
            torch.save(checkpoint, f"checkpoint_{epoch+1}.pth")







if __name__ == "__main__":
    main()
# x = torch.randn((2, 3, 416, 416))
# target = torch.randn((2, 3, 13, 13, 10))
# model = YoloV3(num_classes)
# loss = Yolov3Loss(num_classes)


# out = model(x)
# anchor = torch.tensor(ANCHORS[0], dtype=torch.float32)
# ls = loss(out[0], target, anchor)


# print(out.shape)
# print(ls.shape)