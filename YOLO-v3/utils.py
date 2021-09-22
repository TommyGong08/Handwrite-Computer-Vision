from PIL import Image


def make_416_image(image_path):
    """
    :param image_path:
    将原图填入其以其最大边的边长的正方形
    :return:
    """
    img = Image.open(image_path)
    w, h = img.size[0], img.size[1]
    max_side = max(w, h)
    mask = Image.new(mode="RGB", size=(max_side, max_side),color=(0, 0, 0))
    mask.paste(img)
    mask.resize(416, 416)
    return mask


if __name__ == '__main__':
    mask = make_416_image('data/image/xxxxx.jpg')
    mask.show()