import pygame
import math

from bezier import *


class Game(object):

    def __init__(self, size):
        pygame.init()
        self.w, self.h = size
        self.screen = pygame.display.set_mode(size)
        self.running = True
        with open('track.json', 'r') as inf:
            temp_curves = eval(inf.read())
            self.track = []
            for curve in temp_curves:
                t_curve = []
                for p in curve:
                    t_curve.append(vec2d(p[0], p[1]))
                self.track.append(t_curve)
        A = (0, 100)
        B = (100, 100)
        C = (500, 200)
        D = (500, 300)
        print(self.line_intersection((A, B), (C, D)))

    def parse_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

    def draw(self):
        self.screen.fill((100, 100, 100))

        for curve in self.track:
            if len(curve) < 4:
                continue
            b_points = compute_bezier_points([(x.x, x.y) for x in curve])
            pygame.draw.lines(self.screen, pygame.Color("red"), False, b_points, 2)

        pygame.display.flip()

    def line_intersection(self, line1, line2):
        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        if div == 0:
            raise Exception('lines do not intersect')

        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return x, y