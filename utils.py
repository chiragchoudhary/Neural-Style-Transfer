import numpy as np
from PIL import Image


def save_animation(path, content_img_path, style_img_path, reconstruction_type):
    frames = [Image.open(img) for img in (path / f"{content_img_path.stem}_and_{style_img_path.stem}" / reconstruction_type).glob('*.png')]
    frame_one = frames[0]
    frame_one.save(path / f'{content_img_path.stem}_and_{style_img_path.stem}' / reconstruction_type / f'{content_img_path.stem}_and_{style_img_path.stem}_{reconstruction_type}.gif', format="GIF", append_images=frames, save_all=True, duration=100, loop=0)


def deprocess_image(x):
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype("uint8")
    return x
