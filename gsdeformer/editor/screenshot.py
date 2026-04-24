"""
util that captures the screen of the editor
"""
import subprocess
import time
from pathlib import Path

import PIL.Image
import torchvision.io
from loguru import logger as log


def main():
    # using xdotool to find window
    ret = subprocess.run('xdotool search --name "EditorViewer"', capture_output=True, shell=True, check=True)
    wid = ret.stdout.decode('utf-8').strip()
    log.info("located X window ID {}", wid)

    # screenshot using imagemagick
    t = time.time()
    dst_dir = Path("./captures").absolute()
    tmp_capture = dst_dir / f"captured_{t}.png"
    subprocess.run(f"import -window {wid} {tmp_capture}", shell=True, check=True)
    log.info("screenshot captured", tmp_capture)

    # pick out the images
    img = torchvision.transforms.ToTensor()(PIL.Image.open(str(tmp_capture)))
    render_img = img[:, :, :512]
    cage_img = img[:, :, 512:1024]
    torchvision.utils.save_image(render_img, dst_dir / f"capture_render_{t}.png")
    torchvision.utils.save_image(cage_img, dst_dir / f"capture_cage_{t}.png")
    log.info("screenshot processed")


if __name__ == '__main__':
    main()