"""Draw the OSM Context icon (own simple design: map pin over contour rings) -> packaging/osmcontext.ico"""
from pathlib import Path
from PIL import Image, ImageDraw

S = 256
im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
d = ImageDraw.Draw(im)
d.rounded_rectangle((8, 8, S - 8, S - 8), 40, fill=(34, 64, 52, 255))
for i, r in enumerate((104, 80, 56)):
    d.ellipse((128 - r, 150 - r * 0.62, 128 + r, 150 + r * 0.62), outline=(120, 190, 150, 255), width=6)
d.polygon([(128, 214), (84, 118), (172, 118)], fill=(242, 242, 236, 255))
d.ellipse((84, 56, 172, 144), fill=(242, 242, 236, 255))
d.ellipse((108, 80, 148, 120), fill=(34, 64, 52, 255))
im.save(Path(__file__).with_name("osmcontext.ico"), sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)])
