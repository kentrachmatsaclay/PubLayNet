import os
import sys
import random
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import transforms
import argparse

import cv2
import numpy as np

from utils import (
    overlay_ann,
    overlay_mask,
)

seed = 1234
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



CATEGORIES2LABELS = {
    0: "bg",
    1: "text",
    2: "title",
    3: "list",
    4: "table",
    5: "figure"
}
SAVE_PATH = "output/"
MODEL_PATH = "model.pth"
def get_instance_segmentation_model(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256

    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )
    return model


def main(argv):
    num_classes = 6
    model = get_instance_segmentation_model(num_classes)
    model.cuda()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", 
        default = MODEL_PATH,
        type = str,
        help = "model checkpoint directory"
    )
    parser.add_argument(
        "--image_path",
        default = None,
        type = str,
        required = True,
        help = "directory of path of image to be passed into model"

    )
    parser.add_argument(
        "--output_path",
        default = None,
        type  = str,
        required = True,
        help = "output directory of model results"
    )

    args = parser.parse_args()

    if os.path.exists(MODEL_PATH):
        checkpoint_path = MODEL_PATH
    else:
        checkpoint_path = args.model_path

    print(checkpoint_path)
    assert os.path.exists(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # NOTE: custom  image
    if len(argv) > 0 and os.path.exists(args.image_path):
        image_path = args.image_path
    else:
        image_path = '/home/z/Downloads/hor02_013.A.A0001.png'

    print(image_path)
    assert os.path.exists(image_path)

    image = cv2.imread(image_path)
    rat = 1000 / image.shape[0]
    image = cv2.resize(image, None, fx=rat, fy=rat)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])
    image = transform(image)

    with torch.no_grad():
        prediction = model([image.cuda()])

    image = torch.squeeze(image, 0).permute(1, 2, 0).mul(255).numpy().astype(np.uint8)
    for pred in prediction:
        for idx, mask in enumerate(pred['masks']):
            if pred['scores'][idx].item() < 0.7:
                continue

            m = mask[0].mul(255).byte().cpu().numpy()
            box = list(map(int, pred["boxes"][idx].tolist()))
            label = CATEGORIES2LABELS[pred["labels"][idx].item()]
            if label == "figure":
                box = pred["boxes"][idx].tolist()  # Get the bounding box
                box = [int(b) for b in box]  # Convert to integer
                cropped_image = image[box[1]:box[3], box[0]:box[2]]  # Crop the region

                # Save the cropped image
                output_filename = os.path.join(args.output_path, f"{os.path.basename(args.image_path)}_figure_{idx}.jpg")
                cv2.imwrite(output_filename, cropped_image)

            score = pred["scores"][idx].item()
            # print(score)
            # image = overlay_mask(image, m)
            image = overlay_ann(image, m, box, label, score)

    # cv2.imwrite('/home/z/research/publaynet/example_images/{}'.format(os.path.basename(image_path)), image)
    if os.path.exists(args.output_path):
        cv2.imwrite(args.output_path+'/{}'.format(os.path.basename(image_path)), image)
    else:
        os.mkdir(args.output_path)
        cv2.imwrite(args.output_path+'/{}'.format(os.path.basename(image_path)), image)

if __name__ == "__main__":
    import sys
    argv = sys.argv[1:]
    main(argv)
