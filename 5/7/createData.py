import os
import pygame
from PIL import ImageFont, Image, ImageDraw
from pygame import freetype

def pygame_font(text,curr_dir):
    pygame.init()
    font = pygame.font.Font(os.path.join(curr_dir, "fonts", "simsun.ttc"), 12.5)
    rtext = font.render(text, False, (0, 0, 0), (255, 255, 255))
    pygame.image.save(rtext, os.path.join(curr_dir, "data", "pygame.png"))

def pygame_freetype_font(text,curr_dir):
    pygame.init()
    freetype.init()
    #print(freetype.get_fonts())
    font =  freetype.SysFont("simsun",9)
    #font = freetype.Font(os.path.join(curr_dir, "fonts", "simsun.ttc"), 9)
    font.antialiased = False
    surf = font.render(text, fgcolor=(0, 0, 0), bgcolor=(255, 255, 255), size=12)[0]
    pygame.image.save(surf, os.path.join(curr_dir, "data", "pygame.png"))

def pil_font(text,curr_dir):
    font = ImageFont.truetype(os.path.join(curr_dir, "fonts", "simsun.ttc"),12,index = 0)
    im = Image.new("RGBA",(300,200),(255, 255, 255))
    draw = ImageDraw.Draw(im)
    draw.text( (0,0), text, font=font, fill="#000000")
    im.save(os.path.join(curr_dir, "data", "pil.png"),dpi=(96,96))

def main():
    curr_dir = os.path.dirname(__file__)
    text = u"60片"
    pygame_freetype_font(text,curr_dir)
    pil_font(text,curr_dir)

if __name__ == '__main__':
    main()