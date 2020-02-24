from game import Game
import bezier

g = Game((1024, 768))

while g.running:
    g.parse_events()
    g.draw()
