import joblib
import pygame
import bezier

from datetime import *
import time
from game import Game
from robot import Robot
from nn_robot import Wrapper

r = Wrapper()
draw = True
run = True
start = True


def today_at(hr, min=0, sec=0, micros=0):
    n = datetime.now()
    return n.replace(hour=hr, minute=min, second=sec, microsecond=micros)


now = datetime.now()
if now.minute + 30 > 59:
    hour_time = (now.hour + 1, (now.minute + 30) % 60)
else:
    hour_time = (now.hour, now.minute + 30)

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
                r.q = joblib.load("q_table3.sav")
                # with open('q.json', 'r') as inf:
                #     r.q = eval(inf.read())
                start = False
            if event.key == pygame.K_n:
                start = False

print(r.q)

while run:
    g = Game((1024, 768), r)
    g.draw_mode = draw
    now = datetime.now()
    if now > today_at(hour_time[0], hour_time[1]):
        print("Half an hour passed", hour_time)
        now = datetime.now()
        if now.minute + 30 > 59:
            hour_time = (now.hour + 1, (now.minute + 30) % 60)
        else:
            hour_time = (now.hour, now.minute + 30)
        g.save()
    while g.running:
        g.parse_events()
        draw = g.draw_mode
        if draw < 5:
            g.draw()
        if g.reset:
            break
        if g.shutdown:
            run = False