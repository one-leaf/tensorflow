import os
import pygame
from PIL import ImageFont, Image, ImageDraw

def pygame_font(text,curr_dir):
    pygame.init()
    font = pygame.font.Font(os.path.join(curr_dir, "fonts", "simsun.ttc"), 12)
    rtext = font.render(text, True, (0, 0, 0), (255, 255, 255))
    pygame.image.save(rtext, os.path.join(curr_dir, "data", "pygame.png"))

def pil_font(text,curr_dir):
    font = ImageFont.truetype(os.path.join(curr_dir, "fonts", "simsun.ttc"),12,index = 0)
    im = Image.new("RGBA",(300,200),(255, 255, 255))
    draw = ImageDraw.Draw(im)
    draw.text( (0,0), text, font=font, fill="#000000")
    im.save(os.path.join(curr_dir, "data", "pil.png"))

def main():
    curr_dir = os.path.dirname(__file__)
    text = u"901300000中文"
    pygame_font(text,curr_dir)
    pil_font(text,curr_dir)


if __name__ == '__main__':
    main()