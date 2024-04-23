import cv2
import numpy as np 

def overlay_mask(image, mask, alpha=0.5):
    c = (np.random.random((1, 3)) * 153 + 102).tolist()[0]
 
    mask = np.dstack([mask.astype(np.uint8)] * 3)
    mask = cv2.threshold(mask, 127.5, 255, cv2.THRESH_BINARY)[1]
    inv_mask = 255 - mask

    overlay = image.copy()
    overlay = np.minimum(overlay, inv_mask) 

    color_mask = (mask.astype(np.bool_) * c).astype(np.uint8)
    overlay = np.maximum(overlay, color_mask).astype(np.uint8) 

    image = cv2.addWeighted(image, alpha, overlay, 1 - alpha, 0)
    return image

def overlay_ann(image, mask, box, label, score, alpha=0.5):
    c = np.random.random((1, 3))
    mask_color = (c * 153 + 102).tolist()[0]
    text_color = (c * 183 + 72).tolist()[0]
 
    mask = np.dstack([mask.astype(np.uint8)] * 3)
    mask = cv2.threshold(mask, 127.5, 255, cv2.THRESH_BINARY)[1]
    inv_mask = 255 - mask

    overlay = image.copy()
    overlay = np.minimum(overlay, inv_mask) 

    color_mask = (mask.astype(np.bool_) * mask_color).astype(np.uint8)
        
    overlay = np.maximum(overlay, color_mask).astype(np.uint8) 

    image = cv2.addWeighted(image, alpha, overlay, 1 - alpha, 0)

    # draw on color mask
    cv2.rectangle(
        image,
        (box[0], box[1]),
        (box[2], box[3]),
        mask_color, 1
    )

    (label_size_width, label_size_height), base_line = \
        cv2.getTextSize(
            "{}".format(label),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3, 1
        )

    cv2.rectangle(
        image,
        (box[0], box[1] + 10),
        (box[0] + label_size_width, box[1] + 10 - label_size_height),
        (223, 128, 255),
        cv2.FILLED
    )

    cv2.putText(
        image,
        # "{}: {:.3f}".format(label, score),
        "{}".format(label),
        (box[0], box[1] + 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.3, (0, 0, 0), 1
    )
 
    return image
