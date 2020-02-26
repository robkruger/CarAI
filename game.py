from __future__ import division

import pygame
import math

from bezier import *
from car import Car
from robot import Robot


class Game(object):

    def __init__(self, size, r):
        pygame.init()
        self.w, self.h = size
        self.screen = pygame.display.set_mode(size)
        self.running = True
        with open('track2.json', 'r') as inf:
            temp_curves = eval(inf.read())
            self.track = []
            for curve in temp_curves:
                t_curve = []
                for p in curve:
                    t_curve.append(vec2d(p[0], p[1]))
                self.track.append(t_curve)
        with open('reward_gates2.json', 'r') as inf:
            temp_lines = eval(inf.read())
            self.checkpoints = []
            for i in range(1, len(temp_lines)):
                t_line = []
                for p in temp_lines[i]:
                    t_line.append((p[0], p[1]))
                self.checkpoints.append(t_line)
        self.car_image_original = pygame.image.load("car.png").convert_alpha()
        self.car_image_original = pygame.transform.scale(self.car_image_original, (25, 16))
        self.car_image = pygame.image.load("car.png").convert_alpha()
        self.car_image = pygame.transform.scale(self.car_image, (25, 16))
        self.car = Car()
        self.car.x = 520
        self.car.y = 650
        self.car.direction = 270
        self.clock = pygame.time.Clock()
        self.finish_line = [(520, 705), (520, 614)]
        self.progress = 1
        self.hit_box = []
        self.color = (0, 0, 255)
        self.detection_lines = []
        self.laps = 0
        self.robot = r
        self.reward = -0.1
        self.reset = False
        self.draw_mode = 0
        self.detected_points = []
        self.rect = None
        self.shutdown = False
        self.detection_lines_to_draw = []

    def parse_events(self):
        keys = pygame.key.get_pressed()
        if self.draw_mode != 4:
            self.clock.tick(100)
        else:
            self.clock.tick()
        delta = max(self.clock.get_time(), 1)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_RETURN:
                    self.draw_mode = (self.draw_mode + 1) % 6
                    print("print drawmode: ", self.draw_mode)
                if event.key == pygame.K_p:
                    self.save()
                    self.running = False
                    self.shutdown = True

        self.reward = -0.1

        action = self.robot.do_action()

        throttle = 0
        if keys[pygame.K_w] or action == 0:
            throttle = 100
        if keys[pygame.K_d] or action == 1:
            self.car.steer(-1, delta)
        if keys[pygame.K_a] or action == 2:
            self.car.steer(1, delta)
        if keys[pygame.K_s] or action == 3:
            throttle = -70
        if action == 4:
            throttle = 100
            self.car.steer(1, delta)
        if action == 5:
            throttle = 100
            self.car.steer(-1, delta)

        self.car.update_speed(throttle, delta)

        rotation_radian = self.car.direction * math.pi / 180
        self.car.x += (1000 / delta) * math.sin(rotation_radian) * self.car.speed
        self.car.y += (1000 / delta) * math.cos(rotation_radian) * self.car.speed

        self.color = (0, 0, 255)
        for curve in self.track:
            if len(curve) < 4:
                continue
            b_points = compute_bezier_points([(x.x, x.y) for x in curve])
            for x in range(len(self.hit_box)):
                for i in range(len(b_points)):
                    if len(self.hit_box) < 4:
                        break
                    if i + 1 == len(b_points):
                        break
                    if self.intersect_segment(self.hit_box[x][0], self.hit_box[x][1], b_points[i], b_points[i + 1]):
                        self.color = (255, 0, 0)
                        self.reward = -1
                        self.reset = True
                    i += 1

        if self.progress > 0:
            for x in range(len(self.hit_box)):
                if len(self.hit_box) < 4:
                    break
                if self.intersect_segment(self.hit_box[x][0], self.hit_box[x][1],
                                          self.checkpoints[self.progress - 1][0],
                                          self.checkpoints[self.progress - 1][1]):
                    self.color = (255, 0, 0)
                    self.progress = (self.progress + 1) % (len(self.checkpoints) + 1)
                    print(self.progress)
                    self.reward = 1
                    break
        else:
            for x in range(len(self.hit_box)):
                if len(self.hit_box) < 4:
                    break
                if self.intersect_segment(self.hit_box[x][0], self.hit_box[x][1], self.finish_line[0],
                                          self.finish_line[1]):
                    self.color = (255, 0, 0)
                    self.progress = (self.progress + 1) % len(self.checkpoints)
                    self.laps += 1
                    print(self.progress, self.laps)
                    self.reward = 1
                    break

        self.car_image, self.rect = self.rot_center(self.car_image_original, self.car.direction)

        self.hit_box = [[(self.car.x + self.rect[0], self.car.y + self.rect[1]),
                         (self.car.x + self.rect[2] + self.rect[0], self.car.y + self.rect[1])],
                        [(self.car.x + self.rect[2] + self.rect[0], self.car.y + self.rect[1]),
                         (self.car.x + self.rect[2] + self.rect[0], self.car.y + self.rect[1] + self.rect[3])],
                        [(self.car.x + self.rect[2] + self.rect[0], self.car.y + self.rect[1] + self.rect[3]),
                         (self.car.x + self.rect[0], self.car.y + self.rect[3] + self.rect[1])],
                        [(self.car.x + self.rect[0], self.car.y + self.rect[1]),
                         (self.car.x + self.rect[0], self.car.y + self.rect[3] + self.rect[1])]]

        self.detection_lines = []
        for i in range(9):
            ra = (i * 40) * math.pi / 180 + rotation_radian
            self.detection_lines.append(
                [(self.car.x, self.car.y), (self.car.x + (math.sin(ra) * 100), self.car.y + (math.cos(ra) * 100))])

        self.detected_points = []
        self.detection_lines_to_draw = []
        for line in self.detection_lines:
            intersect = False
            for curve in self.track:
                if len(curve) < 4:
                    continue
                b_points = compute_bezier_points([(x.x, x.y) for x in curve])
                for i in range(len(b_points)):
                    if intersect:
                        continue
                    if i + 1 == len(b_points):
                        break
                    if self.intersect_segment(line[0], line[1], b_points[i], b_points[i + 1]):
                        point = self.line_intersection(line, (b_points[i], b_points[i + 1]))
                        self.detection_lines_to_draw.append(point)
                        distance = (int(round(self.car.x - point[0], 0)), int(round(self.car.y - point[1], 0)))
                        self.detected_points.append(distance)
                        intersect = True
                    i += 1
            if not intersect:
                self.detected_points.append("n")

    def draw(self):
        self.screen.fill((100, 100, 100))

        if self.draw_mode < 4:
            for curve in self.track:
                if len(curve) < 4:
                    continue
                b_points = compute_bezier_points([(x.x, x.y) for x in curve])
                pygame.draw.lines(self.screen, pygame.Color("red"), False, b_points, 2)

        self.screen.blit(self.car_image, (self.car.x + self.rect[0], self.car.y + self.rect[1]))

        if self.draw_mode < 3:
            pygame.draw.lines(self.screen, pygame.Color("white"), False, self.finish_line, 2)

        if self.draw_mode < 3:
            for checkpoint in self.checkpoints:
                pygame.draw.lines(self.screen, (130, 130, 130), False, checkpoint, 2)

        if self.draw_mode < 1:
            for line in self.hit_box:
                pygame.draw.lines(self.screen, self.color, False, line, 2)

        if self.draw_mode < 2:
            for point in self.detection_lines_to_draw:
                if point == "n":
                    continue
                pygame.draw.lines(self.screen, self.color, False,
                                  [(self.car.x + self.rect[0] + self.rect[2] / 2,
                                    self.car.y + self.rect[1] + self.rect[3] / 2),
                                   (point[0], point[1])], 2)

        self.robot.update(self.reward, self.detected_points)

        pygame.display.flip()

    def rot_center(self, image, angle):

        center = image.get_rect().center
        rotated_image = pygame.transform.rotate(image, angle - 90)
        new_rect = rotated_image.get_rect(center=center)

        return rotated_image, new_rect

    def ccw(self, A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    # Return true if line segments AB and CD intersect
    def intersect_segment(self, A, B, C, D):
        return self.ccw(A, C, D) != self.ccw(B, C, D) and self.ccw(A, B, C) != self.ccw(A, B, D)

    def line_intersection(self, line1, line2):
        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        if div == 0:
            return False

        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return int(x + 0.5), int(y + 0.5)

    def save(self):
        for state in self.robot.q:
            for i in range(len(self.robot.q[state])):
                self.robot.q[state][i] = round(self.robot.q[state][i], 3)
        data = json.dumps(self.robot.q)
        f = open("q.json", "w")
        f.write(data)
        f.close()
