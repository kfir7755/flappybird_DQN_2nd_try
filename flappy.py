import pygame
# from pygame.locals import *
import random

pygame.init()

clock = pygame.time.Clock()
fps = 60

human_mode = False

screen_width = 864
screen_height = 936

screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('Flappy Bird')

# define font
font = pygame.font.SysFont('Bauhaus 93', 60)

# define colours
white = (255, 255, 255)

# game variables
pipe_gap = 150
scroll_speed = 4

# load images
bg = pygame.image.load('img/bg.png')
ground_img = pygame.image.load('img/ground.png')
button_img = pygame.image.load('img/restart.png')


def draw_text(text, font, text_col, x, y):
    img = font.render(text, True, text_col)
    screen.blit(img, (x, y))


def draw_text(text, font, text_col, x, y):
    img = font.render(text, True, text_col)
    screen.blit(img, (x, y))


class Game:

    def __init__(self):
        # define game variables
        self.ground_scroll = 0
        self.game_over = False
        self.pipe_frequency = 1500  # milliseconds
        self.last_pipe = pygame.time.get_ticks() - self.pipe_frequency
        self.score = 0
        self.pass_pipe = False
        if human_mode:
            self.flying = False
        else:
            self.flying = True

    def reset_game(self):
        pipe_group.empty()
        flappy.rect.x = 100
        flappy.rect.y = int(screen_height / 2)
        self.score = 0
        if not human_mode:
            self.flying = True


class Bird(pygame.sprite.Sprite):

    def __init__(self, x, y):
        pygame.sprite.Sprite.__init__(self)
        self.images = []
        self.index = 0
        self.counter = 0
        for num in range(1, 4):
            img = pygame.image.load(f"img/bird{num}.png")
            self.images.append(img)
        self.image = self.images[self.index]
        self.rect = self.image.get_rect()
        self.rect.center = [x, y]
        self.vel = 0
        self.clicked = False

    def update(self):

        if game.flying:
            # apply gravity
            self.vel += 0.5
            if self.vel > 8:
                self.vel = 8
            if self.rect.bottom < 768:
                self.rect.y += int(self.vel)

        if not game.game_over:
            # jump
            if pygame.mouse.get_pressed()[0] == 1 and not self.clicked:
                self.clicked = True
                self.vel = -10
            if pygame.mouse.get_pressed()[0] == 0:
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
        self.image = pygame.image.load("img/pipe.png")
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


def get_state(flappy_bird, pipes):
    if len(pipes) > 0:
        bird_y_loc = flappy_bird.rect.y
        x_dist_pipe_bird = pipes.sprites()[0].rect.left - flappy.rect.right
        bot_pipe_y_loc = pipes.sprites()[0].rect.top - bird_y_loc
        top_pipe_y_loc = pipes.sprites()[1].rect.bottom - bird_y_loc
        return x_dist_pipe_bird, bot_pipe_y_loc, top_pipe_y_loc, flappy_bird.vel
    return None


pipe_group = pygame.sprite.Group()
bird_group = pygame.sprite.Group()

flappy = Bird(100, int(screen_height / 2))

bird_group.add(flappy)
game = Game()

if not human_mode:
    # create restart button instance
    button = Button(screen_width // 2 - 50, screen_height // 2 - 100, button_img)

    run = True
    while run:

        clock.tick(fps)

        # draw background
        screen.blit(bg, (0, 0))

        pipe_group.draw(screen)
        bird_group.draw(screen)
        bird_group.update()

        # draw and scroll the ground
        screen.blit(ground_img, (game.ground_scroll, 768))

        # check the score
        if len(pipe_group) > 0:
            if bird_group.sprites()[0].rect.left > pipe_group.sprites()[0].rect.left \
                    and bird_group.sprites()[0].rect.right < pipe_group.sprites()[0].rect.right \
                    and not game.pass_pipe:
                game.pass_pipe = True
            if game.pass_pipe:
                if bird_group.sprites()[0].rect.left > pipe_group.sprites()[0].rect.right:
                    game.score += 1
                    game.pass_pipe = False
        draw_text(str(game.score), font, white, int(screen_width / 2), 20)

        # look for collision
        if pygame.sprite.groupcollide(bird_group, pipe_group, False, False) or flappy.rect.top < 0:
            game.game_over = True
        # once the bird has hit the ground it's game over and no longer flying
        if flappy.rect.bottom >= 768:
            game.game_over = True
            flying = False

        if game.flying and not game.game_over:
            # generate new pipes
            time_now = pygame.time.get_ticks()
            if time_now - game.last_pipe > game.pipe_frequency:
                pipe_height = random.randint(-100, 100)
                btm_pipe = Pipe(screen_width, int(screen_height / 2) + pipe_height, -1)
                top_pipe = Pipe(screen_width, int(screen_height / 2) + pipe_height, 1)
                pipe_group.add(btm_pipe)
                pipe_group.add(top_pipe)
                game.last_pipe = time_now

            pipe_group.update()

            game.ground_scroll -= scroll_speed
            if abs(game.ground_scroll) > 35:
                game.ground_scroll = 0

        # check for game over and reset
        if game.game_over:
            if button.draw():
                game.game_over = False
                game.reset_game()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.MOUSEBUTTONDOWN and not game.flying and not game.game_over:
                game.flying = True

        pygame.display.update()
    pygame.quit()
