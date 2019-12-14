import numpy as np
import cv2


def rle2mask(mask_rle, shape):
    """
    mask_rle: run-length as string formated (start length)ы
    shape: (width,height) of array to return
    Returns numpy array, 1 - mask, 0 - background
    """
    if mask_rle != mask_rle:
        return np.zeros_like(shape)

    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T

#
# def mask2rle(x):
#     dots = np.where(x.T.flatten() == 1)[0]
#     run_lengths = []
#     prev = -2
#     for b in dots:
#         if b > prev + 1:
#             run_lengths.extend((b + 1, 0))
#         run_lengths[-1] += 1
#         prev = b
#     return run_lengths

def mask2rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def out2rle(outputs, i):
    rles = []
    smooth = 1e-15
    prediction = outputs.detach().cpu().softmax(dim=1).numpy()

    for j, sample in enumerate(prediction):
        for val in range(1, 8):
            ch_pred = sample[val]
            shape = shapes[img_names[8*i+j]]
            ch_pred = cv2.resize(ch_pred, (shape[1], shape[0]))
            rles.append(mask2rle(ch_pred.T > 0.5))
    return rles
