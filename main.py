import pygame

from game import Game
import bezier
from robot import Robot

r = Robot()
draw = True
run = True
start = True

pygame.init()
screen = pygame.display.set_mode((1000, 1000))

print("Press Y to use a saved file or Press N to start a new learning fase")

while start:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            start = False
            run = False
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_y:
                with open('q.json', 'r') as inf:
                    r.q = eval(inf.read())
                start = False
            if event.key == pygame.K_n:
                start = False

print(r.q)

while run:
    g = Game((1024, 768), r)
    g.draw_game = draw
    while g.running:
        g.parse_events()
        draw = g.draw_game
        if draw:
            g.draw()
        if g.reset:
            break
        if g.shutdown:
            run = False
