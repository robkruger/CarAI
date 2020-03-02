class Car(object):

    def __init__(self):
        self.x = 0
        self.y = 0
        self.direction = 0
        self.speed = 0
        self.friction = 0.0001

    def update_speed(self, throttle, delta):
        self.speed += (0.1 * (delta / 1000) * (throttle / 100))
        self.speed -= self.friction * (self.speed * 100)
        self.speed = max(self.speed, 0)

    def steer(self, direction, delta):
        self.direction += ((6 * (delta / 1000) / (self.speed / 4 + 0.01)) * direction)
