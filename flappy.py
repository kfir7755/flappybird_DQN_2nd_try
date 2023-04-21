import pygame
import random

pygame.init()

# clock = pygame.time.Clock()

human_mode = False

screen_width = 864
screen_height = 936

screen = pygame.display.set_mode((screen_width, screen_height), flags=pygame.HIDDEN)
pygame.display.set_caption('Flappy Bird')

# define font
font = pygame.font.SysFont('Bauhaus 93', 60)

# define colours
white = (255, 255, 255)

# game variables
pipe_gap = 150
scroll_speed = 2
dist_between_pipes = 300

# load images
bg = pygame.image.load('img/bg.png').convert_alpha()
ground_img = pygame.image.load('img/ground.png').convert_alpha()
button_img = pygame.image.load('img/restart.png').convert_alpha()


def draw_text(text, font, text_col, x, y):
    img = font.render(text, True, text_col)
    screen.blit(img, (x, y))


def draw_text(text, font, text_col, x, y):
    img = font.render(text, True, text_col)
    screen.blit(img, (x, y))


class Bird(pygame.sprite.Sprite):

    def __init__(self, x, y, game, i):
        pygame.sprite.Sprite.__init__(self)
        self.images = []
        self.index = 0
        self.i = i
        self.counter = 0
        for num in range(1, 4):
            img = pygame.image.load(f"img/bird{num}.png").convert_alpha()
            self.images.append(img)
        self.image = self.images[self.index]
        self.rect = self.image.get_rect()
        self.rect.center = [x, y]
        self.vel = 0
        self.game = game
        self.clicked = False

    def update(self, move=None):

        if self.game.flyings[self.i]:
            # apply gravity
            self.vel += 0.5
            if self.vel > 15:
                self.vel = 15
            if self.vel < -35:
                self.vel = -35
            if self.rect.bottom < 768:
                self.rect.y += int(self.vel)

        if not self.game.games_over[self.i]:
            # jump
            if move is None:
                move = pygame.mouse.get_pressed()[0]
            if move == 1 and not self.clicked:
                self.clicked = True
                if self.vel <= 0:
                    self.vel -= 15
                self.vel = -10
            if move == 0:
                self.clicked = False

            # handle the animation
            flap_cooldown = 5
            self.counter += 1

            if self.counter > flap_cooldown:
                self.counter = 0
                self.index += 1
                if self.index >= len(self.images):
                    self.index = 0
                self.image = self.images[self.index]

            # rotate the bird
            self.image = pygame.transform.rotate(self.images[self.index], self.vel * -2)
        else:
            # point the bird at the ground
            self.image = pygame.transform.rotate(self.images[self.index], -90)


class Pipe(pygame.sprite.Sprite):

    def __init__(self, x, y, position):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load("img/pipe.png").convert_alpha()
        self.rect = self.image.get_rect()
        # position variable determines if the pipe is coming from the bottom or top
        # position 1 is from the top, -1 is from the bottom
        if position == 1:
            self.image = pygame.transform.flip(self.image, False, True)
            self.rect.bottomleft = [x, y - int(pipe_gap / 2)]
        elif position == -1:
            self.rect.topleft = [x, y + int(pipe_gap / 2)]

    def update(self):
        self.rect.x -= scroll_speed
        if self.rect.right < 0:
            self.kill()


class Button:
    def __init__(self, x, y, image):
        self.image = image
        self.rect = self.image.get_rect()
        self.rect.topleft = (x, y)

    def draw(self):
        action = False

        # get mouse position
        pos = pygame.mouse.get_pos()

        # check mouseover and clicked conditions
        if self.rect.collidepoint(pos):
            if pygame.mouse.get_pressed()[0] == 1:
                action = True

        # draw button
        screen.blit(self.image, (self.rect.x, self.rect.y))
        return action


# class Game:
#
#     def __init__(self):
#         # define game variables
#         self.ground_scroll = 0
#         self.game_over = False
#         self.last_pipe = -1
#         self.score = 0
#         self.pass_pipe = False
#         self.pipe_group = pygame.sprite.Group()
#         self.bird_group = pygame.sprite.Group()
#         self.flappy = Bird(100, int(screen_height / 2), self)
#         self.bird_group.add(self.flappy)
#         if human_mode:
#             self.flying = False
#         else:
#             self.flying = True
#
#     def reset_game(self):
#         self.pipe_group.empty()
#         self.flappy.rect.x = 100
#         self.flappy.rect.y = int(screen_height / 2)
#         self.score = 0
#         self.game_over = False
#         if not human_mode:
#             self.flying = True
#
#     def get_state(self):
#         if len(self.pipe_group) > 0:
#             bird_y_loc = self.flappy.rect.y
#             x_dist_pipe_bird = self.pipe_group.sprites()[0].rect.left - self.flappy.rect.right
#             bot_pipe_y_loc = self.pipe_group.sprites()[0].rect.top - bird_y_loc
#             top_pipe_y_loc = self.pipe_group.sprites()[1].rect.bottom - bird_y_loc
#             return x_dist_pipe_bird / 1000, 2 * bot_pipe_y_loc / screen_height, 2 * top_pipe_y_loc / screen_height, self.flappy.vel / 10
#         return 0.734, 2 * 134 / screen_height, -16, 0.05
#
#     def play_game(self):
#         if not human_mode:
#             # create restart button instance
#             button = Button(screen_width // 2 - 50, screen_height // 2 - 100, button_img)
#
#             run = True
#             while run:
#                 run = self.play_step(button=button)[0]
#             pygame.quit()
#
#     def play_step(self, move=None, button=None):
#         run = True
#         # clock.tick(fps)
#         reward = 0
#         # draw background
#         screen.blit(bg, (0, 0))
#
#         self.pipe_group.draw(screen)
#         self.bird_group.draw(screen)
#         self.bird_group.update(move)
#
#         # draw and scroll the ground
#         screen.blit(ground_img, (self.ground_scroll, 768))
#
#         # check the score
#         if len(self.pipe_group) > 0:
#             if self.bird_group.sprites()[0].rect.left > self.pipe_group.sprites()[0].rect.left \
#                     and self.bird_group.sprites()[0].rect.right < self.pipe_group.sprites()[0].rect.right \
#                     and not self.pass_pipe:
#                 self.pass_pipe = True
#                 reward = 100
#             if self.pass_pipe:
#                 if self.bird_group.sprites()[0].rect.left > self.pipe_group.sprites()[0].rect.right:
#                     self.score += 1
#                     self.pass_pipe = False
#         draw_text(str(self.score), font, white, int(screen_width / 2), 20)
#
#         # look for collision
#         if pygame.sprite.groupcollide(self.bird_group, self.pipe_group, False,
#                                       False) or self.flappy.rect.top < 0:
#             self.game_over = True
#         # once the bird has hit the ground it's game over and no longer flying
#         if self.flappy.rect.bottom >= 768:
#             self.game_over = True
#             self.flying = False
#
#         if self.flying and not self.game_over:
#             # generate new pipes
#             if self.last_pipe < 0:
#                 pipe_height = random.randint(-200, 200)
#                 btm_pipe = Pipe(screen_width, int(screen_height / 2) + pipe_height, -1)
#                 top_pipe = Pipe(screen_width, int(screen_height / 2) + pipe_height, 1)
#                 self.pipe_group.add(btm_pipe)
#                 self.pipe_group.add(top_pipe)
#                 self.last_pipe = dist_between_pipes
#             self.last_pipe -= scroll_speed
#             self.pipe_group.update()
#             self.ground_scroll -= scroll_speed
#             if abs(self.ground_scroll) > 35:
#                 self.ground_scroll = 0
#
#         # check for game over and reset
#         if self.game_over:
#             reward = -10
#             if button is not None:
#                 if button.draw():
#                     self.reset_game()
#
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 run = False
#             if event.type == pygame.MOUSEBUTTONDOWN and not self.flying and not self.game_over and move is None:
#                 self.flying = True
#             elif move == 1 and not self.flying and not self.game_over:
#                 self.flying = True
#         pygame.display.update()
#         return run, reward, self.game_over, self.score


class Big_Game:
    def __init__(self, n):
        # define game variables
        self.ground_scroll = 0
        self.screen = pygame.display.set_mode((screen_width, screen_height), flags=pygame.SHOWN)
        self.is_shown = True
        pygame.display.set_caption('Flappy Bird')
        self.games_over = {}
        self.last_pipe = dist_between_pipes
        self.scores = {}
        self.passes_pipe = [False] * n
        self.pipe_group = pygame.sprite.Group()
        pipe_height = random.randint(-200, 200)
        btm_pipe = Pipe(screen_width, int(screen_height / 2) + pipe_height, -1)
        top_pipe = Pipe(screen_width, int(screen_height / 2) + pipe_height, 1)
        self.pipe_group.add(btm_pipe)
        self.pipe_group.add(top_pipe)
        self.birds_group = []
        for i in range(n):
            self.birds_group.append(pygame.sprite.Group())
            self.games_over[i] = False
            self.scores[i] = 0
        self.flappies = []
        for i in range(n):
            self.flappies.append(Bird(100, int(screen_height / 2), self, i))
            self.birds_group[i].add(self.flappies[i])
        self.n = n
        if human_mode:
            self.flyings = [False] * n
        else:
            self.flyings = [True] * n

    def change_screen_condition(self):
        if self.is_shown:
            self.is_shown = False
            self.screen = pygame.display.set_mode((screen_width, screen_height), flags=pygame.HIDDEN)
        else:
            self.is_shown = True
            self.screen = pygame.display.set_mode((screen_width, screen_height), flags=pygame.SHOWN)

    def findMinNotDone(self):
        for i in range(self.n):
            if not self.games_over[i]:
                return i
        return None

    def reset_game(self):
        self.pipe_group.empty()
        self.last_pipe = pipe_gap * 2
        pipe_height = random.randint(-200, 200)
        btm_pipe = Pipe(screen_width, int(screen_height / 2) + pipe_height, -1)
        top_pipe = Pipe(screen_width, int(screen_height / 2) + pipe_height, 1)
        self.pipe_group.add(btm_pipe)
        self.pipe_group.add(top_pipe)
        for i in range(self.n):
            self.flappies[i].rect.x = 100
            self.flappies[i].rect.y = int(screen_height / 2)
            self.scores[i] = 0
            self.games_over[i] = False
            if not human_mode:
                self.flyings[i] = True

    def get_state(self, i):
        if len(self.pipe_group) > 0:
            bird_y_loc = self.flappies[i].rect.y
            x_dist_pipe_bird = self.pipe_group.sprites()[0].rect.left - self.flappies[i].rect.right
            bot_pipe_y_loc = self.pipe_group.sprites()[0].rect.top - bird_y_loc
            top_pipe_y_loc = self.pipe_group.sprites()[1].rect.bottom - bird_y_loc
            return x_dist_pipe_bird / 500, 10 * bot_pipe_y_loc / screen_height, 5 * top_pipe_y_loc / screen_height, \
                   self.flappies[i].vel / 35
        return None

    def play_step(self, moves, agents_alive):
        # draw background
        if self.is_shown:
            screen.blit(bg, (0, 0))
            self.pipe_group.draw(screen)
            # draw and scroll the ground
            screen.blit(ground_img, (self.ground_scroll, 768))
            for i in agents_alive:
                self.birds_group[i].draw(screen)
                self.birds_group[i].update(moves[i])
        else:
            for i in agents_alive:
                self.birds_group[i].update(moves[i])

        # check the score
        if len(self.pipe_group) > 0:
            for i in agents_alive:
                if self.birds_group[i].sprites()[0].rect.left > self.pipe_group.sprites()[0].rect.left \
                        and self.birds_group[i].sprites()[0].rect.right < self.pipe_group.sprites()[0].rect.right \
                        and not self.passes_pipe[i]:
                    self.passes_pipe[i] = True
                if self.passes_pipe[i]:
                    if self.birds_group[i].sprites()[0].rect.left > self.pipe_group.sprites()[0].rect.right:
                        self.scores[i] += 1
                        self.passes_pipe[i] = False
        if self.is_shown:
            draw_text(str(max(self.scores.values())), font, white, int(screen_width / 2), 20)
            draw_text(str(sum([1 if not self.games_over[i] else 0 for i in range(self.n)])), font, white,
                      int(3 * screen_width / 4), 20)

        # look for collision
        for i in agents_alive:
            if pygame.sprite.groupcollide(self.birds_group[i], self.pipe_group, False, False) \
                    or self.flappies[i].rect.top < 0:
                self.games_over[i] = True
            # once the bird has hit the ground it's game over and no longer flying
            if self.flappies[i].rect.bottom >= 768:
                self.games_over[i] = True
                self.flyings[i] = False

            if self.flyings[i] and not self.games_over[i]:
                # generate new pipes
                if self.last_pipe < 0:
                    pipe_height = random.randint(-200, 200)
                    btm_pipe = Pipe(screen_width, int(screen_height / 2) + pipe_height, -1)
                    top_pipe = Pipe(screen_width, int(screen_height / 2) + pipe_height, 1)
                    self.pipe_group.add(btm_pipe)
                    self.pipe_group.add(top_pipe)
                    self.last_pipe = dist_between_pipes
        self.last_pipe -= scroll_speed
        self.pipe_group.update()

        if abs(self.ground_scroll) > 35 and self.findMinNotDone() is not None:
            self.ground_scroll -= scroll_speed
            self.ground_scroll = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        for i in agents_alive:
            if moves[i] == 1 and not self.flyings[i] and not self.games_over[i]:
                self.flyings[i] = True
        if self.is_shown:
            pygame.display.update()
        return self.games_over, self.scores
