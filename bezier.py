"""
bezier.py - Calculates a bezier curve from control points.

2007 Victor Blomqvist
Released to the Public Domain
"""
import pygame
from pygame.locals import *
import json


class vec2d(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y


gray = (100, 100, 100)
lightgray = (200, 200, 200)
red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)
X, Y, Z = 0, 1, 2


def compute_bezier_points(vertices, numPoints=None):
    if numPoints is None:
        numPoints = 4

    result = []

    b0x = vertices[0][0]
    b0y = vertices[0][1]
    b1x = vertices[1][0]
    b1y = vertices[1][1]
    b2x = vertices[2][0]
    b2y = vertices[2][1]
    b3x = vertices[3][0]
    b3y = vertices[3][1]

    # Compute polynomial coefficients from Bezier points
    ax = -b0x + 3 * b1x + -3 * b2x + b3x
    ay = -b0y + 3 * b1y + -3 * b2y + b3y

    bx = 3 * b0x + -6 * b1x + 3 * b2x
    by = 3 * b0y + -6 * b1y + 3 * b2y

    cx = -3 * b0x + 3 * b1x
    cy = -3 * b0y + 3 * b1y

    dx = b0x
    dy = b0y

    # Set up the number of steps and step size
    numSteps = numPoints - 1  # arbitrary choice
    h = 1.0 / numSteps  # compute our step size

    # Compute forward differences from Bezier points and "h"
    pointX = dx
    pointY = dy

    firstFDX = ax * (h * h * h) + bx * (h * h) + cx * h
    firstFDY = ay * (h * h * h) + by * (h * h) + cy * h

    secondFDX = 6 * ax * (h * h * h) + 2 * bx * (h * h)
    secondFDY = 6 * ay * (h * h * h) + 2 * by * (h * h)

    thirdFDX = 6 * ax * (h * h * h)
    thirdFDY = 6 * ay * (h * h * h)

    # Compute points at each step
    result.append((int(pointX), int(pointY)))

    for i in range(numSteps):
        pointX += firstFDX
        pointY += firstFDY

        firstFDX += secondFDX
        firstFDY += secondFDY

        secondFDX += thirdFDX
        secondFDY += thirdFDY

        result.append((int(pointX), int(pointY)))

    return result


def main():
    pygame.init()
    screen = pygame.display.set_mode((1024, 768))

    curves = []
    ### Control points that are later used to calculate the curve
    control_points = [vec2d(100, 100), vec2d(150, 500), vec2d(450, 500), vec2d(500, 150)]
    curves.append(control_points)

    reward_control_points = [vec2d(0, 0), vec2d(10, 10)]
    rewards = []
    rewards.append(reward_control_points)

    ### The currently selected point
    selected = None

    clock = pygame.time.Clock()

    display_lines = True

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == KEYDOWN:
                if event.key == K_RETURN:
                    display_lines = not display_lines
                elif event.key == K_s:
                    all_curves = []
                    for curve in curves:
                        this_curve = []
                        for point in curve:
                            this_curve.append((point.x, point.y))
                        all_curves.append(this_curve)
                    data = json.dumps(all_curves)
                    f = open("track2.json", "w")
                    f.write(data)
                    f.close()
                    all_rewards = []
                    for line in rewards:
                        this_reward = []
                        for point in line:
                            this_reward.append((point.x, point.y))
                        all_rewards.append(this_reward)
                    data2 = json.dumps(all_rewards)
                    f = open("reward_gates2.json", "w")
                    f.write(data2)
                    f.close()
                    running = False
                elif event.key == K_l:
                    with open('track2.json', 'r') as inf:
                        temp_curves = eval(inf.read())
                        curves = []
                        for curve in temp_curves:
                            t_curve = []
                            for p in curve:
                                t_curve.append(vec2d(p[0], p[1]))
                            curves.append(t_curve)
                    with open('reward_gates2.json', 'r') as inf:
                        temp_curves = eval(inf.read())
                        rewards = []
                        for line in temp_curves:
                            t_curve = []
                            for p in line:
                                t_curve.append(vec2d(p[0], p[1]))
                            rewards.append(t_curve)
            elif event.type == MOUSEBUTTONDOWN and event.button == 1:
                for curve in curves:
                    for p in curve:
                        if abs(p.x - event.pos[X]) < 10 and abs(p.y - event.pos[Y]) < 10:
                            selected = p
                for reward in rewards:
                    for p in reward:
                        if abs(p.x - event.pos[X]) < 10 and abs(p.y - event.pos[Y]) < 10:
                            selected = p
            elif event.type == MOUSEBUTTONDOWN and event.button == 2:
                appended = False
                for i in range(len(rewards)):
                    if len(rewards[i]) < 3:
                        rewards[i].append(vec2d(event.pos[X], event.pos[Y]))
                        appended = True
                        break
                if not appended:
                    new_list = [vec2d(event.pos[X], event.pos[Y])]
                    rewards.append(new_list)
            elif event.type == MOUSEBUTTONDOWN and event.button == 3:
                if len(curves[-1]) < 4:
                    curves[-1].append(vec2d(event.pos[X], event.pos[Y]))
                else:
                    new_list = [vec2d(event.pos[X], event.pos[Y])]
                    curves.append(new_list)
            # elif event.type == MOUSEBUTTONDOWN and event.button == 3:
            #     x,y = pygame.mouse.get_pos()
            #     control_points.append(vec2d(x,y))
            elif event.type == MOUSEBUTTONUP and event.button == 1:
                selected = None

        ### Draw stuff
        screen.fill(gray)

        if selected is not None:
            selected.x, selected.y = pygame.mouse.get_pos()
            pygame.draw.circle(screen, green, (selected.x, selected.y), 10)

        ### Draw control points
        if display_lines:
            for curve in curves:
                for p in curve:
                    pygame.draw.circle(screen, blue, (int(p.x), int(p.y)), 4)

        if display_lines:
            for line in rewards:
                for p in line:
                    pygame.draw.circle(screen, green, (int(p.x), int(p.y)), 4)

        if display_lines:
            for curve in curves:
                ### Draw control "lines"
                if len(curve) < 2:
                    continue
                pygame.draw.lines(screen, lightgray, False, [(x.x, x.y) for x in curve])

        ### Draw bezier curve
        for curve in curves:
            if len(curve) < 4:
                continue
            b_points = compute_bezier_points([(x.x, x.y) for x in curve])
            pygame.draw.lines(screen, pygame.Color("red"), False, b_points, 2)

        for line in rewards:
            if len(line) < 3:
                continue
            pygame.draw.lines(screen, pygame.Color("green"), False, ((line[0].x, line[0].y), (line[1].x, line[1].y)), 2)

        ### Flip screen
        pygame.display.flip()
        clock.tick(100)
        # print clock.get_fps()


if __name__ == '__main__':
    main()