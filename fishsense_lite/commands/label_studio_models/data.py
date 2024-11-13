import urllib


class Data:
    def __init__(self, prefix: str, img: str):
        self.img = f"{prefix}{urllib.parse.quote(img)}"
