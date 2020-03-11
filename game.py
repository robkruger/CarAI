from __future__ import division

import os
import neatfast

import pygame
import math
import joblib
import numpy as np

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
        # self.car = Car(520, 650, 270)
        self.clock = pygame.time.Clock()
        self.finish_line = [(520, 705), (520, 614)]
        self.progress = 1
        self.hit_box = []
        self.color = (0, 0, 255)
        self.detection_lines = []
        self.laps = 0
        self.robot = r
        self.reset = False
        self.draw_mode = 0
        self.detected_points = []
        self.rect = None
        self.shutdown = False
        self.detection_lines_to_draw = []
        self.game_over = False
        self.action = -1
        self.score_increased = 0
        self.old_lines = None
        self.reward = 0
        self.crossed_checkpoint = False
        pygame.font.init()
        self.font = pygame.font.SysFont("Arial", 25)

    def parse_events(self, car, progress, action, steps):
        self.crossed_checkpoint = False
        # keys = pygame.key.get_pressed()
        delta = 10  # max(self.clock.get_time(), 1)

        self.action = action  # self.robot.do_action()

        throttle = 0
        if self.action == 0:
            throttle = 100
        if self.action == 1:
            car.steer(-1, delta)
        if self.action == 2:
            car.steer(1, delta)
        if self.action == 3:
            throttle = -100
        if self.action == 4:
            throttle = 100
            car.steer(1, delta)
        if self.action == 5:
            throttle = 100
            car.steer(-1, delta)

        self.action = -1

        car.update_speed(throttle, delta)

        rotation_radian = car.direction * math.pi / 180
        car.x += (1000 / delta) * math.sin(rotation_radian) * car.speed
        car.y += (1000 / delta) * math.cos(rotation_radian) * car.speed

        car_image, rect = self.rot_center(self.car_image_original, car.direction)

        hit_box = [[(car.x + rect[0], car.y + rect[1]),
                    (car.x + rect[0] + rect[2], car.y + rect[1])],
                   [(car.x + rect[0] + rect[2], car.y + rect[1]),
                    (car.x + rect[0] + rect[2], car.y + rect[1] + rect[3])],
                   [(car.x + rect[0] + rect[2], car.y + rect[1] + rect[3]),
                    (car.x + rect[0], car.y + rect[1] + rect[3])],
                   [(car.x + rect[0], car.y + rect[1] + rect[3]),
                    (car.x + rect[0], car.y + rect[1])]]

        self.color = (0, 0, 255)
        crossed_checkpoint = False
        if progress > 0:
            for x in range(len(hit_box)):
                if len(hit_box) < 4:
                    break
                if self.intersect_segment(hit_box[x][0], hit_box[x][1],
                                          self.checkpoints[progress - 1][0],
                                          self.checkpoints[progress - 1][1]):
                    self.color = (255, 0, 0)
                    progress = (progress + 1) % (len(self.checkpoints) + 1)
                    crossed_checkpoint = True
                    break
        else:
            for x in range(len(hit_box)):
                if len(hit_box) < 4:
                    break
                if self.intersect_segment(hit_box[x][0], hit_box[x][1], self.finish_line[0],
                                          self.finish_line[1]):
                    self.color = (255, 0, 0)
                    progress = (progress + 1) % len(self.checkpoints)
                    crossed_checkpoint = True
                    # print(progress)
                    break

        game_over = False
        for curve in self.track:
            if len(curve) < 4:
                continue
            b_points = compute_bezier_points([(x.x, x.y) for x in curve])
            for x in range(len(hit_box)):
                for i in range(len(b_points)):
                    if len(hit_box) < 4:
                        break
                    if i + 1 == len(b_points):
                        break
                    if self.intersect_segment(hit_box[x][0], hit_box[x][1], b_points[i], b_points[i + 1]):
                        self.color = (255, 0, 0)
                        game_over = True
                    i += 1

        detection_lines = []
        ra = (270) * math.pi / 180 + rotation_radian
        detection_lines.append(
            [(car.x, car.y), (car.x + (math.sin(ra) * 200), car.y + (math.cos(ra) * 200))])
        ra = (315) * math.pi / 180 + rotation_radian
        detection_lines.append(
            [(car.x, car.y), (car.x + (math.sin(ra) * 200), car.y + (math.cos(ra) * 200))])
        ra = (360) * math.pi / 180 + rotation_radian
        detection_lines.append(
            [(car.x, car.y), (car.x + (math.sin(ra) * 200), car.y + (math.cos(ra) * 200))])
        ra = (45) * math.pi / 180 + rotation_radian
        detection_lines.append(
            [(car.x, car.y), (car.x + (math.sin(ra) * 200), car.y + (math.cos(ra) * 200))])
        ra = (90) * math.pi / 180 + rotation_radian
        detection_lines.append(
            [(car.x, car.y), (car.x + (math.sin(ra) * 200), car.y + (math.cos(ra) * 200))])
        ra = (135) * math.pi / 180 + rotation_radian
        detection_lines.append(
            [(car.x, car.y), (car.x + (math.sin(ra) * 200), car.y + (math.cos(ra) * 200))])
        ra = (180) * math.pi / 180 + rotation_radian
        detection_lines.append(
            [(car.x, car.y), (car.x + (math.sin(ra) * 200), car.y + (math.cos(ra) * 200))])
        ra = (225) * math.pi / 180 + rotation_radian
        detection_lines.append(
            [(car.x, car.y), (car.x + (math.sin(ra) * 200), car.y + (math.cos(ra) * 200))])

        detected_points = []
        detection_lines_to_draw = []
        for line in detection_lines:
            point = (0, 0)
            intersect = False
            distance = 99999999999
            for curve in self.track:
                if len(curve) < 4:
                    continue
                b_points = compute_bezier_points([(x.x, x.y) for x in curve])
                for i in range(len(b_points)):
                    if i + 1 == len(b_points):
                        break
                    if self.intersect_segment(line[0], line[1], b_points[i], b_points[i + 1]):
                        t_point = self.line_intersection(line, (b_points[i], b_points[i + 1]))
                        distance_to_point = (int(round(car.x - t_point[0], 0)), int(round(car.y - t_point[1], 0)))
                        actual_dist = math.sqrt(distance_to_point[0] * distance_to_point[0] + distance_to_point[1] * distance_to_point[1])
                        if actual_dist < distance:
                            distance = actual_dist
                            point = t_point
                        intersect = True
            if not intersect:
                detected_points.append("n")
            else:
                detected_points.append(int(round(distance, 0)))
                detection_lines_to_draw.append(point)

            # self.detected_points[i] = round(min(self.detected_points[i], 100) / 100.0, 2)
        # self.detected_points.append(self.progress)
        # radians = math.atan2(self.checkpoints[self.progress - 1][2][1] - self.car.y,
        #                      self.checkpoints[self.progress - 1][2][0] - self.car.x)
        # degrees = (math.degrees(radians - rotation_radian) + 90) % 360
        # self.detected_points.append(round(degrees / 360, 2))
        state = []
        for point in detected_points:
            if point == 'n':
                state.append(1)
                continue
            state.append(point / 200)
        state.append(car.speed * 10)

        if not crossed_checkpoint:
            steps += 1
        else:
            steps = 0

        return state, game_over, crossed_checkpoint, progress, car_image, rect, steps, hit_box, detection_lines_to_draw

    def draw(self, cars, generation, deaths, max_steps):
        self.screen.fill((100, 100, 100))

        if self.draw_mode < 4:
            for curve in self.track:
                if len(curve) < 4:
                    continue
                b_points = compute_bezier_points([(x.x, x.y) for x in curve])
                pygame.draw.lines(self.screen, pygame.Color("red"), False, b_points, 2)

        for car in cars:
            # 0: car_image, 1: car, 2: rect
            self.screen.blit(car[0], (car[1].x + car[2][0], car[1].y + car[2][1]))
            if self.draw_mode < 2 or self.draw_mode == 4:
                # for point in car[4]:
                #     if point == "n":
                #         continue
                #     pygame.draw.lines(self.screen, self.color, False,
                #                       [(car[1].x + car[2][0] + car[2][2] / 2,
                #                         car[1].y + car[2][1] + car[2][3] / 2),
                #                        (point[0], point[1])], 2)
                for line in car[3]:
                    pygame.draw.lines(self.screen, self.color, False, line, 2)

        for car in deaths:
            # 0: car_image, 1: car, 2: rect
            self.screen.blit(car[0], (car[1].x + car[2][0], car[1].y + car[2][1]))
            if self.draw_mode < 2 or self.draw_mode == 4:
                # for point in car[4]:
                #     if point == "n":
                #         continue
                #     pygame.draw.lines(self.screen, self.color, False,
                #                       [(car[1].x + car[2][0] + car[2][2] / 2,
                #                         car[1].y + car[2][1] + car[2][3] / 2),
                #                        (point[0], point[1])], 2)
                for line in car[3]:
                    pygame.draw.lines(self.screen, self.color, False, line, 2)

        if self.draw_mode < 3:
            pygame.draw.lines(self.screen, pygame.Color("white"), False, self.finish_line, 2)

        if self.draw_mode < 3:
            for checkpoint in self.checkpoints:
                pygame.draw.lines(self.screen, (130, 130, 130), False, (checkpoint[0], checkpoint[1]), 2)

        text = self.font.render("Generation: " + str(generation + start_generation) + "     Max steps: " + str(max_steps), True, (255, 255, 255))
        self.screen.blit(text, (0, 0))

        # self.robot.update(self.reward, self.detected_points, self.progress, round(self.car.speed, 3))

        pygame.display.flip()

    def rot_center(self, image, angle):

        center = image.get_rect().center
        rotated_image = pygame.transform.rotate(image, angle - 90)
        new_rect = rotated_image.get_rect(center=center)

        return rotated_image, new_rect

    def rotate_point(self, cx, cy, angle, point: tuple):
        s = math.sin(angle)
        c = math.cos(angle)

        tx = point[0] - cx
        ty = point[1] - cy

        newx = tx * c - ty * s
        newy = ty * s + ty * c

        tx = newx + cx
        ty = newy + cy

        return tx, ty

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
        j = 0
        for state in self.robot.q:
            for i in range(len(self.robot.q[state])):
                self.robot.q[state][i] = round(self.robot.q[state][i], 3)
                if self.robot.q[state][i] > 0:
                    j += 1
        print(j, len(self.robot.q))
        joblib.dump(self.robot.q, "q_table3.sav", 2)
        # data = json.dumps(self.robot.q)
        # f = open("q.json", "w")
        # f.write(data)
        # f.close()


def fitness_func(genomes, config, best):
    global generation, show_best
    generation += 1
    nets = []
    ge = []
    cars = []

    for _, g in genomes:
        net = neatfast.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        cars.append((Car(520, 650, 270), 1, -1, 0))
        g.fitness = 0
        ge.append(g)

    run = True
    g = Game((1024, 768), None)
    clock = pygame.time.Clock()
    death_positions = []
    max_steps = 60 + (int((generation + start_generation) / 5) * 15)
    step = 0
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_ESCAPE:
                    show_best = not show_best

        step += 1
        all_fitness = []
        for i, genome in enumerate(ge):
            all_fitness.append(genome.fitness)
        clock.tick()
        draw_info = []
        for x, car in enumerate(cars):
            inputs, done, checkpoint, progress, car_image, rect, steps, hitbox, lines = g.parse_events(car[0], car[1], car[2], car[3])
            # if x == np.argmax(all_fitness):
            if x == 0 and show_best:
                draw_info.append((car_image, car[0], rect, hitbox, lines))
            elif not show_best:
                draw_info.append((car_image, car[0], rect, hitbox, lines))

            if done:
                penalty = 2
                ge[x].fitness -= penalty
                nets.pop(x)
                ge.pop(x)
                cars.pop(x)
                # death_positions.append(draw_info[-1])
                continue

            if steps > 200:
                ge[x].fitness -= 10
                nets.pop(x)
                ge.pop(x)
                cars.pop(x)
                continue

            ge[x].fitness -= 0.01

            if checkpoint:
                ge[x].fitness += 5

            output = nets[x].activate(inputs)
            action = np.argmax(output)
            cars[x] = (car[0], progress, action, steps)

        g.draw(draw_info, generation, death_positions, max_steps)

        if len(cars) < 1:
            run = False

        if step > max_steps:
            run = False


generation = -1
start_generation = 129
show_best = False


def run(config_path):
    config = neatfast.config.Config(neatfast.DefaultGenome, neatfast.DefaultReproduction,
                                neatfast.DefaultSpeciesSet, neatfast.DefaultStagnation, config_path)

    # p = neatfast.Population(config)
    p = neatfast.Checkpointer(5).restore_checkpoint("saves/generation-129")

    p.add_reporter(neatfast.Checkpointer(5, 300, "saves/generation-"))

    p.add_reporter(neatfast.StdOutReporter(True))
    stats = neatfast.StatisticsReporter
    # p.add_reporter(stats)

    winner = p.run(fitness_func)
    net = neatfast.nn.FeedForwardNetwork.create(winner, config)
    clock = pygame.time.Clock()

    while 1:
        car = (Car(520, 650, 270), 1, -1, 0)
        draw_info = []
        g = Game((1024, 768), None)

        while 1:
            clock.tick(30)
            draw_info = []
            inputs, done, checkpoint, progress, car_image, rect, steps, hitbox, lines = g.parse_events(car[0], car[1],
                                                                                                       car[2], car[3])
            # if x == np.argmax(all_fitness):
            draw_info.append((car_image, car[0], rect, hitbox, lines))

            if done:
                break

            if steps > 200:
                break

            output = net.activate(inputs)
            action = np.argmax(output)
            car = (car[0], progress, action, steps)
            g.draw(draw_info, -1, [], -1)


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    run(config_path)
