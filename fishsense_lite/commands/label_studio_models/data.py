import urllib


class Data:
    def __init__(self, img: str):
        self.img = urllib.parse.quote(img)
