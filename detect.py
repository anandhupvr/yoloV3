import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms

from model.darknet import YoloV3
from utils.util import nms, cells_to_bboxes, plot_image

def main(weight_file, image_file):
    
    transform = transforms.Compose([
        transforms.Resize((416, 416)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YoloV3(20).to(device)

    # Load checkpoint/weightfile
    checkpoint = torch.load(weight_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # load data

    image = Image.open(image_file).convert("RGB")
    input = transform(image).unsqueeze(0)


    model.eval()
    
    with torch.no_grad():
        input = input.to(device)
        out = model(input)


    ANCHORS = [
        [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
        [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
        [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
    ]  # Note these have been rescaled to be between [0, 1]


    # img, target = dataset.__getitem__(2)

    S = [13, 26, 52]
    scaled_anchors = torch.tensor(ANCHORS) / (
        1 / torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    )
    boxes = []

    for i in range(out[0].shape[1]):
        anchor = scaled_anchors[i]
        anchor = anchor.to(device)
        # print(anchor.shape)
        # print(out[i].shape)
        boxes += cells_to_bboxes(
            out[i], is_preds=True, S=out[i].shape[2], anchors=anchor
        )[0]
    # import pdb; pdb.set_trace()
    boxesn = nms(boxes, 1, 0.5)

    image = image.resize((416, 416), Image.Resampling.LANCZOS)
    plot_image(np.array(image), boxesn, 'voc')






if __name__ == '__main__':
    weight_file = "./checkpoint_150.pth"
    image_file = "./test.jpg"
    main(weight_file, image_file)