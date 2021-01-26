import gym
import os
import neat
import pickle
from cart import CartPole

def main():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    #run(config_path)
    test(config_path)

def test(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    winner = pickle.load(open("winner.p", 'rb'))

    carts = []
    nets = []
    for _ in range(5):
        net = neat.nn.FeedForwardNetwork.create(winner, config)
        nets.append(net)
        carts.append(CartPole())

    while (len(carts) > 0):
        for x, cart in enumerate(carts):
            cart.render()
            observation, reward, done, info = cart.step()

            if done:
                cart.done()
                carts.pop(x)
                nets.pop(x)
                break

            output = round(nets[x].activate(observation)[0])
            cart.addInstruction(output)


def run(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(train, 50)
    pickle.dump(winner, open("winner.p", "wb"))

def train(parentGenomes, config):
    nets = []
    genomes = []
    carts = []

    for _, genome in parentGenomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        genome.fitness = 0
        genomes.append(genome)
        carts.append(CartPole())

    while (len(carts) > 0):
        for x, cart in enumerate(carts):
            cart.render()
            observation, reward, done, info = cart.step()

            if done:
                cart.done()
                carts.pop(x)
                nets.pop(x)
                genomes.pop(x)
                break

            genomes[x].fitness += reward

            output = round(nets[x].activate(observation)[0])
            cart.addInstruction(output)