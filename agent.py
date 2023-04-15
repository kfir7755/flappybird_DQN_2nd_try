import pygame
import torch
import numpy as np
from flappy import Big_Game
from model import Linear_QNet
import copy

pygame.init()

AGENTS_PER_GEN = 750
# fps = 1000
# clock = pygame.time.Clock()


class Agent:

    def __init__(self):
        self.n_games = 0
        self.model = Linear_QNet(4, 2)

    def get_action(self, state):
        state0 = torch.tensor(state, dtype=torch.float)
        prediction = self.model(state0)
        move = torch.argmax(prediction).item()
        return move

    def calc_fitness(self, old_state, score):
        self.model.fitness = score**2 + 1 / (old_state[0] + 150)


def agents_for_new_gen(agents):
    sum_fitness = sum([agent.model.fitness for agent in agents])
    prob_to_be_parent = [agent.model.fitness / sum_fitness for agent in agents]
    new_gen_agents = [copy.deepcopy(agent) for agent in
                      np.random.choice(agents, size=len(agents) - 10, p=prob_to_be_parent)]
    for i in range(len(new_gen_agents)):
        new_gen_agents[i].model.mutate()
    ind = np.argpartition(prob_to_be_parent, -10)[-10:]
    for i in ind:
        new_gen_agents.append(agents[i])
    return new_gen_agents


def are_all_games_done(game):
    for i in range(AGENTS_PER_GEN):
        if not game.games_over[i]:
            return False
    return True


def train():
    record = 0
    generation = 0
    agents_list = [Agent() for _ in range(AGENTS_PER_GEN)]
    game = Big_Game(AGENTS_PER_GEN)
    added_score = [False] * AGENTS_PER_GEN
    calculated_fitness_this_round = [False] * AGENTS_PER_GEN
    while True:
        record_for_this_gen = 0
        mean_score = 0
        # clock.tick(fps)
        while not are_all_games_done(game):
            old_states = []
            final_moves = []
            for i, agent in enumerate(agents_list):
                # get old state
                old_states.append(game.get_state(i))

                # get move
                final_moves.append(agent.get_action(old_states[i]))

            # perform move and get new state
            dones, scores = game.play_step(moves=final_moves)

            for i in range(AGENTS_PER_GEN):
                if dones[i] and not calculated_fitness_this_round[i]:
                    calculated_fitness_this_round[i] = True
                    agents_list[i].calc_fitness(old_states[i], scores[i])
                    if scores[i] > record:
                        record = scores[i]
                        agents_list[i].model.save()
                    if scores[i] > record_for_this_gen:
                        record_for_this_gen = scores[i]
                    if not added_score[i]:
                        added_score[i] = True
                        mean_score += scores[i]
        mean_score /= AGENTS_PER_GEN
        print("gen:", generation, "record for this gen:", record_for_this_gen, "mean score:", round(mean_score,3), "record:",
              record)
        generation += 1
        agents_list = agents_for_new_gen(agents_list)
        game.reset_game()
        added_score = [False] * AGENTS_PER_GEN
        calculated_fitness_this_round = [False] * AGENTS_PER_GEN


if __name__ == '__main__':
    train()
