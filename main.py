import pygame
import numpy as np
import math

import AI
from AI import Agent
import gym

import matplotlib.pyplot as plt
import cv2


# toto = np.array([1, 2])
# print(toto)
# toto = toto[np.newaxis, :]
# print(toto)
# quit()

"""
env = gym.make('LunarLander-v2', render_mode="rgb_array")
n_games = 50000
agent = Agent(gamma=0.99, epsilon=1.0, alpha=0.0005, inputDims=8, nbActions=4,
                           mem_size=1000000, batchSize=64, epsilon_end=0.01)
scores = []
eps_history = []

nbScoreNegatif = 0
nbScorePositif = 0

for i in range(n_games):
    done = False
    score = 0
    observationTuple = env.reset()
    observation = observationTuple[0]

    hasExit = False
    nbStep = 0
    while not done:
        #print(observation)
        action = agent.choose_action(observation)
        observation_, reward, done, truncated, info = env.step(action)

        #frameBuffer = frameBuffer[:, :, ::-1].copy()
        frameBuffer = env.render()
        cv2.imshow('Univers expansion', frameBuffer)
        cv2.waitKey(1)

        #plt.imshow(env.render())
        #plt.show()
        #plt.draw()
        #plt.pause(0.00001)

        score += reward
        agent.remember(observation, action, reward, observation_, done)
        observation = observation_
        agent.learn()
        print("Game ", i, " Step ", nbStep, " Score is currently at ", score, " score +/- ", nbScorePositif, " ", nbScoreNegatif)
        nbStep += 1

    if not hasExit:
        if score > 0:
            nbScorePositif += 1
        else:
            nbScoreNegatif += 1

    eps_history.append(agent.epsilon)
    scores.append(score)

    avg_score = np.mean(scores[max(0, i-100):(i+1)])
    print('episode ', i, 'score %.2f' % score, 'avg score %.2f' % avg_score)

env.close()

quit()
"""


# go backward (frener), remettre toutes les actions
# 2 rayons de plus a l'avant et deux à l'arriere
# enlever l'env custom



pygame.init()

# Set up the drawing window
SCREEN_WIDTH = 1024
SCREEN_HEIGHT = 768
MAX_DIST = math.sqrt(SCREEN_WIDTH*SCREEN_WIDTH + SCREEN_HEIGHT*SCREEN_HEIGHT)
AI_RANGE = MAX_DIST / 5.0
MAX_VELOCITY = 150.0
FPS = 15 #1000.0 / 0.055
PERIODE_MS = (1 / FPS) * 1000
DEGRE_TO_RAD = 0.0174532
RAY_SIZE = 1000
NB_CHECKPOINTS = 20

SCREEN = pygame.display.set_mode([SCREEN_WIDTH, SCREEN_HEIGHT])
SPRITE_WIDTH = 25
SPRITE_HEIGHT = 17
CAR_SPRITE = pygame.image.load('redCar.png').convert_alpha()
CAR_SPRITE = pygame.transform.scale(CAR_SPRITE, (SPRITE_WIDTH, SPRITE_HEIGHT))
CAR_STARTING_POS = np.array([100, (SCREEN_HEIGHT / 2) + 10])
CAR_STARTING_DIR = np.array([-1.0, 0.0])
CAR_STARTING_ANGLE = 90.0


# Car possible actions
INPUT_FORWARD   = 0b1000
INPUT_BACKWARD  = 0b0100
INPUT_LEFT      = 0b0010
INPUT_RIGHT     = 0b0001
INPUT_FORWARD_RIGHT = INPUT_FORWARD | INPUT_RIGHT
INPUT_FORWARD_LEFT = INPUT_FORWARD | INPUT_LEFT
INPUT_BACKWARD_RIGHT = INPUT_BACKWARD | INPUT_RIGHT
INPUT_BACKWARD_LEFT = INPUT_BACKWARD | INPUT_LEFT
#ALL_INPUTS = [INPUT_FORWARD, INPUT_BACKWARD, INPUT_LEFT, INPUT_RIGHT, INPUT_FORWARD_RIGHT, INPUT_FORWARD_LEFT, INPUT_BACKWARD_RIGHT, INPUT_BACKWARD_LEFT]
#ALL_INPUTS = [INPUT_FORWARD, INPUT_LEFT, INPUT_RIGHT, INPUT_FORWARD_RIGHT, INPUT_FORWARD_LEFT, INPUT_BACKWARD]
ALL_INPUTS = [INPUT_FORWARD, INPUT_LEFT, INPUT_RIGHT, INPUT_FORWARD_RIGHT, INPUT_FORWARD_LEFT]
NB_ACTION = len(ALL_INPUTS)
# Car sensors
#SENSORS = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0], [1.0, 1.0], [-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0]])
#SENSORS = np.array([[-1.0, 0.0], [0.0, 1.0], [0.0, -1.0], [-1.0, -1.0], [-1.0, 1.0], [-0.5, 1.0], [0.5, 1.0], [-0.5, -1.0], [0.5, -1.0], [-1.0, -0.5], [-1.0, 0.5]])
SENSORS = np.array([[-1.0, 0.0], [0.0, 1.0], [0.0, -1.0], [-1.0, -1.0], [-1.0, 1.0], [-1.0, -0.5], [-1.0, 0.5]])
NB_DATA_INPUT = 2 # Velocity, dist to next checkpoint
ENV_SIZE = len(SENSORS) + NB_DATA_INPUT

USE_AI = True
LOAD_MODEL = False
MODEL_NAME = 'dqn_model_3.h5'
DRAW_DEBUG = False



USE_TRACK_DEBUG = True
##TRACK1_DEBUG = [(24, 338), (28, 238), (43, 160), (78, 114), (132, 90), (232, 59), (289, 73), (321, 114), (335, 157), (351, 210), (394, 225), (445, 212), (467, 172), (475, 119), (485, 76), (513, 57), (632, 44), (706, 78), (728, 159), (728, 236), (718, 334), (689, 402), (609, 467), (507, 582), (342, 570), (306, 465), (280, 371), (208, 378), (178, 496), (132, 575), (60, 575), (34, 508), (24, 338)]
##TRACK2_DEBUG = [(106, 313), (115, 239), (146, 193), (192, 188), (235, 207), (285, 250), (309, 293), (347, 317), (423, 319), (485, 307), (525, 258), (542, 195), (567, 146), (621, 153), (635, 187), (632, 249), (614, 313), (581, 379), (548, 430), (503, 474), (439, 481), (401, 413), (377, 361), (338, 324), (293, 297), (255, 289), (222, 301), (179, 324), (155, 364), (118, 459), (106, 313)]
#TRACK1_DEBUG = [(3, 298), (17, 183), (92, 85), (206, 31), (381, 11), (564, 20), (685, 88), (765, 221), (785, 325), (753, 448), (686, 539), (561, 580), (405, 588), (211, 580), (103, 518), (20, 414), (3, 298)]
#TRACK2_DEBUG = [(176, 310), (206, 223), (285, 175), (416, 169), (544, 188), (601, 265), (607, 339), (566, 417), (475, 437), (360, 440), (273, 419), (215, 372), (176, 310)]
##CHECKPOINTS_DEBUG = [(2, 215), (208, 270), (66, 80), (224, 235), (160, 26), (278, 215), (254, 12), (323, 191), (357, 6), (367, 197), (468, 7), (453, 197), (596, 20), (498, 192), (658, 48), (557, 225), (762, 166), (575, 262), (795, 293), (585, 317), (774, 434), (576, 366), (700, 560), (535, 388), (580, 591), (522, 405), (485, 593), (482, 424), (397, 594), (404, 426), (287, 594), (341, 426), (171, 573), (298, 412), (86, 522), (258, 385), (33, 468), (223, 357), (6, 354), (207, 322)]
#CHECKPOINTS_DEBUG = [(160, 26), (278, 215), (254, 12), (323, 191), (357, 6), (367, 197), (468, 7), (453, 197), (596, 20), (498, 192), (658, 48), (557, 225), (762, 166), (575, 262), (795, 293), (585, 317), (774, 434), (576, 366), (700, 560), (535, 388), (580, 591), (522, 405), (485, 593), (482, 424), (397, 594), (404, 426), (287, 594), (341, 426), (171, 573), (298, 412), (86, 522), (258, 385), (33, 468), (223, 357), (6, 354), (207, 322)]

TRACK1_DEBUG = [(56, 399), (55, 357), (70, 316), (91, 298), (124, 276), (147, 250), (145, 219), (136, 192), (121, 181), (96, 161), (95, 116), (102, 80), (125, 60), (152, 41), (208, 27), (292, 19), (388, 17), (437, 31), (487, 41), (536, 50), (598, 48), (654, 34), (701, 22), (765, 25), (829, 30), (885, 52), (948, 92), (980, 143), (975, 195), (941, 236), (892, 256), (818, 269), (535, 332), (513, 362), (513, 386), (536, 401), (565, 401), (907, 343), (957, 370), (975, 438), (970, 529), (941, 625), (904, 709), (796, 754), (684, 757), (539, 648), (394, 743), (323, 758), (149, 735), (98, 672), (55, 524), (56, 399)]
TRACK2_DEBUG = [(141, 405), (156, 365), (187, 344), (221, 309), (240, 268), (241, 222), (229, 166), (201, 139), (215, 122), (269, 106), (339, 98), (493, 130), (571, 140), (664, 126), (714, 106), (780, 112), (814, 130), (806, 162), (773, 178), (463, 258), (396, 318), (383, 399), (409, 457), (467, 487), (585, 488), (807, 453), (855, 468), (860, 524), (830, 620), (757, 672), (540, 525), (338, 651), (236, 638), (174, 580), (141, 405)]
#CHECKPOINTS_DEBUG = []#[(60, 293), (200, 371), (114, 233), (270, 242), (96, 58), (249, 142), (341, 4), (345, 125), (530, 31), (525, 145), (756, 9), (735, 120), (778, 153), (999, 177), (678, 190), (720, 326), (421, 280), (564, 366), (554, 388), (523, 507), (674, 368), (704, 488), (807, 343), (813, 479), (998, 437), (842, 506), (967, 635), (819, 602), (723, 762), (741, 652), (547, 684), (630, 559), (518, 693), (423, 574), (350, 760), (325, 637), (125, 737), (230, 615), (43, 548), (179, 522)]
CHECKPOINTS_DEBUG = [(40, 364), (176, 376), (113, 270), (195, 349), (133, 252), (248, 280), (131, 203), (255, 199), (78, 124), (228, 144), (246, 11), (274, 119), (401, 12), (401, 119), (525, 37), (523, 139), (644, 27), (675, 128), (925, 64), (794, 128), (798, 161), (859, 272), (554, 226), (584, 323), (376, 406), (534, 375), (778, 354), (791, 456), (846, 486), (981, 473), (828, 606), (939, 647), (756, 660), (763, 760), (538, 516), (539, 663), (334, 639), (349, 752), (198, 585), (86, 642)]

def segments_intersect(seg1_start, seg1_end, seg2_start, seg2_end):
    """
    Detect if two 2D line segments intersect.

    Arguments:
    seg1_start: tuple (x,y) containing the coordinates of the start point of segment 1
    seg1_end: tuple (x,y) containing the coordinates of the end point of segment 1
    seg2_start: tuple (x,y) containing the coordinates of the start point of segment 2
    seg2_end: tuple (x,y) containing the coordinates of the end point of segment 2

    Returns:
    The coordinates of the intersection point if the segments intersect, None otherwise.
    """

    def orientation(p, q, r):
        # Function to find the orientation of a triplet of points (p, q, r)
        # Returns:
        # 0 if p, q, r are collinear
        # 1 if p, q, r turn in counterclockwise direction
        # 2 if p, q, r turn in clockwise direction
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0:
            return 0
        return 1 if val > 0 else 2

    # Find the orientations needed to determine the intersection
    o1 = orientation(seg1_start, seg1_end, seg2_start)
    o2 = orientation(seg1_start, seg1_end, seg2_end)
    o3 = orientation(seg2_start, seg2_end, seg1_start)
    o4 = orientation(seg2_start, seg2_end, seg1_end)

    # General case
    if o1 != o2 and o3 != o4:
        # Find the intersection of the two lines
        x1, y1 = seg1_start
        x2, y2 = seg1_end
        x3, y3 = seg2_start
        x4, y4 = seg2_end
        x_num = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4))
        y_num = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4))
        denom = ((x1 - x2) * (y3 - y4)) - ((y1 - y2) * (x3 - x4))
        x_int = x_num / denom
        y_int = y_num / denom
        return (x_int, y_int)

    # If the segments do not intersect, return None
    return None



class HitBox:
    def __init__(self, xSize, ySize, center, dir):
        self.xSize = xSize
        self.ySize = ySize
        self.center = np.array([])
        self.frontLeft = np.array([])
        self.backRight = np.array([])
        self.frontRight = np.array([])
        self.backLeft = np.array([])

        self.build(center, dir)

    def build(self, center, dir):
        leftDir = np.array([dir[1], -dir[0]])
        self.center = center
        self.frontLeft = self.center + dir * (self.ySize / 2) + leftDir * (self.xSize / 2)
        self.backRight = self.center - dir * (self.ySize / 2) - leftDir * (self.xSize / 2)
        self.frontRight = self.center + dir * (self.ySize / 2) - leftDir * (self.xSize / 2)
        self.backLeft = self.center - dir * (self.ySize / 2) + leftDir * (self.xSize / 2)



class Car:
    def __init__(self):
        self.reset()
        self.sprite = CAR_SPRITE
        self.mass = 1
        self.rect = self.sprite.get_rect(center=self.pos)
        self.raycast = []
        for i in range(len(SENSORS)):
            self.raycast.append(SENSORS[i])
        self.hitBox = HitBox(SPRITE_HEIGHT, SPRITE_WIDTH, self.pos, self.dir)
        self.distWithNextCheckPoint = np.inf

        # Init first orientation
        self.turn(0)

        # AI agent
        self.env = AI.CustomEnv(NB_ACTION, ENV_SIZE, 1.0)
        self.agent = Agent(gamma=0.99, epsilon=1.0, alpha=0.0005, inputDims=ENV_SIZE, nbActions=NB_ACTION,
                           mem_size=1000000, batchSize=64, epsilon_end=0.01)
        #self.agent.loadModel()
        self.scores = []
        self.eps_history = []


    def reset(self):
        self.accel = np.array([0.0, 0.0])
        self.velocity = np.array([0.0, 0.0])
        self.pos = CAR_STARTING_POS
        self.accelForce = 0
        self.angle = CAR_STARTING_ANGLE
        self.dir = CAR_STARTING_DIR
        self.raycastResults = []
        self.prevRaycastResults = []
        for i in range(len(SENSORS)):
            self.raycastResults.append(float(math.inf))
            self.prevRaycastResults.append(float(math.inf))


    def move(self, input, dt):
        if bool(input & INPUT_FORWARD):
            self.moveForward(dt)
        if bool(input & INPUT_BACKWARD):
            self.moveBackward(dt)
        if bool(input & INPUT_LEFT):
            self.turnLeft(dt)
        if bool(input & INPUT_RIGHT):
            self.turnRight(dt)


    def draw(self, screen):
        screen.blit(self.sprite, self.rect)

    def moveForward(self, dt):
        self.accel = 5000.0 * dt * self.dir

    def moveBackward(self, dt):
        self.accel = -5000.0 * dt * self.dir

    def turn(self, theta):
        self.angle = self.angle + theta
        if (self.angle > 360):
            self.angle = self.angle - 360
        elif (self.angle < 0):
            self.angle = self.angle + 360

        #theta = theta * DEGRE_TO_RAD
        theta = self.angle * DEGRE_TO_RAD

        # Compute rotation matrix and rotate
        rotationMatrix = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
        #self.dir = rotationMatrix.dot(self.dir)
        self.dir = rotationMatrix.dot(CAR_STARTING_DIR)

        # Normalize it to be sure it stay a unit vector over time
        norm = np.linalg.norm(self.dir)
        self.dir = self.dir / norm

        # Same for all raycast
        for i in range(len(self.raycast)):
            sensor = np.array([SENSORS[i][0], SENSORS[i][1]])
            self.raycast[i] = rotationMatrix.dot(sensor)
            norm2 = np.linalg.norm(self.raycast[i])
            self.raycast[i] = self.raycast[i] / norm2

        # Get angle from dir vector and rotate sprite
        #angle = np.arctan2(self.dir[0], self.dir[1])
        #angle = angle / DEGRE_TO_RAD
        #angle = angle - 90  # Beark
        #self.sprite = pygame.transform.rotate(CAR_SPRITE, angle)
        self.sprite = pygame.transform.rotate(CAR_SPRITE, -self.angle + 180)


    def turnLeft(self, dt):
        self.turn(-100.0 * dt)

    def turnRight(self, dt):
        self.turn(100.0 * dt)

    def update(self, deltaTsec):
        frictionCoef = 0.02
        normV = np.linalg.norm(self.velocity)
        friction = np.array([0, 0])
        if normV > 0:
            vDir = self.velocity / normV
            friction = normV * normV * frictionCoef * vDir * -1
        sommeForces = self.accel + friction
        self.accel = np.array([0, 0])
        accel = sommeForces / self.mass
        self.velocity = self.velocity + accel * deltaTsec
        newNormV = np.linalg.norm(self.velocity)
        # Limit speed
        if newNormV > MAX_VELOCITY:
            coef = MAX_VELOCITY / newNormV
            self.velocity = self.velocity * coef

        self.pos = self.pos + self.velocity * deltaTsec
        self.rect = self.sprite.get_rect(center=self.pos)

        # Update hitbox
        self.hitBox.build(self.pos, self.dir)

    def raycastSensors(self, track1, track2, nextCheckPoint):
        self.raycastResults.clear()
        for i in range(len(SENSORS)):
            self.raycastResults.append(float(math.inf))

        def raycast1track(track, ray, indexRay):
            for j in range(len(track)):
                if j > 0:
                    endPos = self.pos + ray * RAY_SIZE
                    intersect = segments_intersect(self.pos, endPos, track[j - 1], track[j])
                    if intersect != None:
                        dist = np.linalg.norm(intersect - self.pos)
                        if dist < self.raycastResults[indexRay]:
                            self.raycastResults[indexRay] = dist

        # Raycast with next checkpoint
        #endPos = self.pos + self.raycast[1] * RAY_SIZE
        #intersect = segments_intersect(self.pos, endPos, nextCheckPoint[0], nextCheckPoint[1])
        #if intersect != None:
            #self.distWithNextCheckPoint = np.linalg.norm(intersect - self.pos)
        #else:
            #self.distWithNextCheckPoint = np.inf
        cpMid = np.array(nextCheckPoint[0]) + np.array(nextCheckPoint[1])
        cpMid = cpMid / 2
        self.distWithNextCheckPoint = np.linalg.norm(cpMid - self.pos)

        # Raycast with both track sides
        for i in range(len(self.raycast)):
            self.prevRaycastResults[i] = self.raycastResults[i]
            raycast1track(track1, self.raycast[i], i)
            raycast1track(track2, self.raycast[i], i)


class Game:
    def __init__(self):
        self.running = True
        self.screen = SCREEN
        self.track1 = []
        self.track1Completed = False
        self.track2 = []
        self.track2Completed = False
        self.car = Car()
        self.prevTick = pygame.time.get_ticks()
        self.checkPointsCompleted = False
        self.checkPoints = []
        self.mouseCursor = np.array([0, 0])
        self.checkpointIndex = 0

        if USE_TRACK_DEBUG:
            self.track1 = TRACK1_DEBUG
            self.track2 = TRACK2_DEBUG
            self.checkPoints = CHECKPOINTS_DEBUG
            self.track1Completed = True
            self.track2Completed = True
            self.checkPointsCompleted = True


    def hitNextCheckPoint(self):
        # Check only 2 sides of the hitbox
        # No need to check for the all 4 sides
        # Still two perpendicular sides are needed to ensure we don't "jump" over the checkpoint in one frame

        def increaseIndex():
            self.checkpointIndex = self.checkpointIndex + 1
            idTmp1, idTmp2 = self.getCheckpointIndexes(self.checkpointIndex)

            if idTmp2 == -1 or idTmp1 == -1:
                self.checkpointIndex = 0

        id1, id2 = self.getCheckpointIndexes(self.checkpointIndex)
        if id2 != -1 and id2 != -1:
            intersect = segments_intersect(self.car.hitBox.frontLeft, self.car.hitBox.backLeft, self.checkPoints[id1], self.checkPoints[id2])
            if intersect != None:
                increaseIndex()
                return True
            intersect = segments_intersect(self.car.hitBox.frontLeft, self.car.hitBox.frontRight, self.checkPoints[id1], self.checkPoints[id2])
            if intersect != None:
                increaseIndex()
                return True
        return False


    def addDotToTrack(self, pos, track, isTrackCompleted):
        if not isTrackCompleted:
            if len(track) > 0:
                p1 = np.array(track[0])
                p2 = np.array(pos)
                v = p2 - p1
                distToStart = np.linalg.norm(v)
                if distToStart < 15:
                    pos = track[0]
                    isTrackCompleted = True
            track.append(pos)
        return isTrackCompleted


    def handleEvents(self, dt):
        for event in pygame.event.get():
            pos = pygame.mouse.get_pos()
            self.mouseCursor = pos
            if event.type == pygame.MOUSEBUTTONDOWN:
                if not self.track1Completed:
                    self.track1Completed = self.addDotToTrack(pos, self.track1, self.track1Completed)
                elif not self.track2Completed:
                    self.track2Completed = self.addDotToTrack(pos, self.track2, self.track2Completed)
                elif not self.checkPointsCompleted:
                    self.checkPoints.append(pos)
                    if len(self.checkPoints) >= (NB_CHECKPOINTS * 2):
                        self.checkPointsCompleted = True


            if event.type == pygame.QUIT:
                self.running = False

        keyPressed = pygame.key.get_pressed()
        if keyPressed[pygame.K_LEFT]:
            self.car.move(INPUT_LEFT, dt)
        if keyPressed[pygame.K_RIGHT]:
            self.car.move(INPUT_RIGHT, dt)
        if keyPressed[pygame.K_UP]:
            self.car.move(INPUT_FORWARD, dt)
        if keyPressed[pygame.K_DOWN]:
            self.car.move(INPUT_BACKWARD, dt)

    # Collision with hitbox
    def collideWithTrack(self, track):
        for j in range(len(track)):
            if j > 0:
                intersect = segments_intersect(self.car.hitBox.backLeft, self.car.hitBox.frontLeft, track[j - 1], track[j])
                if intersect != None:
                    return True
                intersect = segments_intersect(self.car.hitBox.frontLeft, self.car.hitBox.frontRight, track[j - 1], track[j])
                if intersect != None:
                    return True
                intersect = segments_intersect(self.car.hitBox.frontRight, self.car.hitBox.backRight, track[j - 1], track[j])
                if intersect != None:
                    return True
                intersect = segments_intersect(self.car.hitBox.backRight, self.car.hitBox.backLeft, track[j - 1], track[j])
                if intersect != None:
                    return True
        return False

    def isCarDead(self):
        if self.collideWithTrack(self.track1):
            return True
        if self.collideWithTrack(self.track2):
            return True
        return False


    def resetAfterDeath(self):
        # Reset car's physics
        self.car.reset()
        # Reset car's first orientation
        self.car.turn(0)
        # Reset car's next checkpoint
        self.checkpointIndex = 0
        # Reset raycast angle

    def observeEnv(self):
        observ = np.zeros((ENV_SIZE, ), dtype=np.float32)
        id = 0
        #observ[id] = self.car.pos[0] / float(SCREEN_WIDTH)
        #id += 1
        #observ[id] = self.car.pos[1] / float(SCREEN_HEIGHT)
        #id += 1
        #observ[id] = self.car.angle / 360.0
        #id += 1
        observ[id] = min(np.linalg.norm(self.car.velocity) / MAX_VELOCITY, 1.0)
        id += 1
        observ[id] = min(self.car.distWithNextCheckPoint / AI_RANGE, 1.0)

        #print(ENV_SIZE + NB_DATA_INPUT)
        for i in range(ENV_SIZE - NB_DATA_INPUT):
            id += 1
            #print(id)
            observ[id] = min(self.car.raycastResults[i] / AI_RANGE, 1.0)
            #log = math.log(self.car.raycastResults[i], 10)
            #clamped = 1 - math.exp(-(self.car.raycastResults[i] / 100.0))
            #print("observ[id] ", self.car.raycastResults[i], " ", old, " ", observ[id])

        return observ


    def getCheckpointIndexes(self, index):
        id1 = (self.checkpointIndex * 2)
        id2 = id1 + 1
        if id2 >= len(self.checkPoints) or id1 < 0:
            return -1, -1

        return id1, id2


    def draw(self, isdead=False):
        # Fill the background with white
        self.screen.fill((255, 255, 255))

        # Draw track
        for i in range(len(self.track1)):
            if i > 0:
                pygame.draw.line(self.screen, color='grey', start_pos=self.track1[i - 1], end_pos=self.track1[i],
                                 width=4)
        if not self.track1Completed and len(self.track1) > 0:
            pygame.draw.line(self.screen, color='grey', start_pos=self.track1[-1], end_pos=self.mouseCursor, width=4)
        for i in range(len(self.track2)):
            if i > 0:
                pygame.draw.line(self.screen, color='grey', start_pos=self.track2[i - 1], end_pos=self.track2[i],
                                 width=4)
        if not self.track2Completed and len(self.track2) > 0:
            pygame.draw.line(self.screen, color='grey', start_pos=self.track2[-1], end_pos=self.mouseCursor, width=4)

        # Draw all checkpoints
        if DRAW_DEBUG:
            drawAllCheckpoint = True
            if drawAllCheckpoint:
                for i in range(len(self.checkPoints)):
                    if i > 0 and (i % 2) != 0:
                        pygame.draw.line(self.screen, color='green', start_pos=self.checkPoints[i - 1],
                                         end_pos=self.checkPoints[i], width=4)
                if not self.checkPointsCompleted and len(self.checkPoints) > 0 and (len(self.checkPoints) % 2) != 0:
                    pygame.draw.line(self.screen, color='green', start_pos=self.checkPoints[-1],
                                     end_pos=self.mouseCursor, width=4)
            else:
                id1, id2 = self.getCheckpointIndexes(self.checkpointIndex)
                if id1 != -1 and id2 != -1:
                    pygame.draw.line(self.screen, color='green',
                                     start_pos=self.checkPoints[id1],
                                     end_pos=self.checkPoints[id2], width=4)

        # Draw the car
        self.car.draw(self.screen)

        # Draw debug dir vector
        if DRAW_DEBUG:
            end = self.car.pos + self.car.dir * 100
            pygame.draw.line(self.screen, color='black', start_pos=self.car.pos, end_pos=end, width=4)

        # Draw debug hitbox
        if DRAW_DEBUG:
            colorHb = 'black'
            if isdead:
                colorHb = 'red'
            pygame.draw.line(self.screen, color=colorHb, start_pos=self.car.hitBox.backLeft,
                             end_pos=self.car.hitBox.frontLeft, width=2)
            pygame.draw.line(self.screen, color=colorHb, start_pos=self.car.hitBox.backLeft,
                             end_pos=self.car.hitBox.backRight, width=2)
            pygame.draw.line(self.screen, color=colorHb, start_pos=self.car.hitBox.frontRight,
                             end_pos=self.car.hitBox.frontLeft, width=2)
            pygame.draw.line(self.screen, color=colorHb, start_pos=self.car.hitBox.frontRight,
                             end_pos=self.car.hitBox.backRight, width=2)

        # Draw debug raycast and collisions
        if DRAW_DEBUG:
            for i in range(len(self.car.raycast)):
                pygame.draw.line(self.screen, color='black', start_pos=self.car.pos,
                                 end_pos=self.car.pos + self.car.raycast[i] * RAY_SIZE, width=2)
                if i < len(self.car.raycastResults):
                    collisionPoint = self.car.pos + self.car.raycast[i] * self.car.raycastResults[i]
                    pygame.draw.circle(self.screen, color='black', center=collisionPoint, radius=5, width=3)

        #if self.car.distWithNextCheckPoint != np.inf:
        #   collisionPoint = self.car.pos + self.car.raycast[1] * self.car.distWithNextCheckPoint
        #   pygame.draw.circle(self.screen, color='blue', center=collisionPoint, radius=5, width=3)

        # Print track 1 & 2
        #if self.track1Completed and self.track2Completed and self.checkPointsCompleted:
        #    print(self.track1)
        #    print(self.track2)
        #    print(self.checkPoints)
        #    self.running = False

        # Flip the display
        pygame.display.flip()


    def mainLoop(self):

        # Build track
        while (not self.track1Completed) or (not self.track2Completed) or (not self.checkPointsCompleted):
            self.handleEvents(0.0)
            self.draw()

        if LOAD_MODEL:
            self.car.agent.loadModel(MODEL_NAME)
            self.car.agent.epsilon = 0.0

        observation = self.car.env.reset()
        observation = self.observeEnv()
        score = 0

        lastDistToCP = 0.0
        isdead = False
        nbIterations = 0

        while self.running:
            # Handle FPS
            ticks = pygame.time.get_ticks()
            deltaT = ticks - self.prevTick
            waitTime = PERIODE_MS - deltaT
            if waitTime < 0:
                waitTime = 0
            pygame.time.wait(int(waitTime))
            self.prevTick = pygame.time.get_ticks()

            deltaTimeSec = (deltaT + waitTime) / 1000
            #print(waitTime, " ", deltaTimeSec, " ", PERIODE_MS)

            deltaTimeSec = PERIODE_MS / 1000 # To be sure IA is trained in the same conditions

            # dt = 0.0666667

            if isdead:
                self.resetAfterDeath()
                done = True


            if USE_AI:
                # IA choose an action and move the car
                action = 0
                if not LOAD_MODEL:
                    action = self.car.agent.choose_action(observation)
                else:
                    observation = self.observeEnv()
                    action = self.car.agent.choose_action(observation)

                self.car.move(ALL_INPUTS[action], deltaTimeSec)

            done = False

            # Update physics, events & game
            self.handleEvents(deltaTimeSec)
            self.car.update(deltaTimeSec)
            id1, id2 = self.getCheckpointIndexes(self.checkpointIndex)
            self.car.raycastSensors(self.track1, self.track2, (self.checkPoints[id1], self.checkPoints[id2]))
            hasHitCheckPoint = self.hitNextCheckPoint()

            # Check if the car hit the wall
            isdead = self.isCarDead()
            if isdead:
                done = True

            # IA learn from the consequences of its action
            #new_observation, reward, done, info = self.car.env.step(action, isdead, hasHitCheckPoint, self.car.angle)

            if USE_AI:
                reward = 0.0
                if isdead:
                    reward -= 100.0
                if hasHitCheckPoint:
                    reward += 10.0

                if self.car.distWithNextCheckPoint < lastDistToCP:
                    reward += 1.0
                else:
                    reward -= 1.0

                #if ALL_INPUTS[action] == INPUT_BACKWARD:
                #    reward -= 1

                lastDistToCP = self.car.distWithNextCheckPoint

                if not LOAD_MODEL:
                    new_observation = self.observeEnv() # Added
                    score += reward
                    self.car.agent.remember(observation, action, reward, new_observation, done)
                    observation = new_observation

                    self.car.agent.learn()

                    nbIterations += 1
                    if (nbIterations % 2500) == 0 and nbIterations > 0:
                        self.car.agent.saveModel()
                        print("Model saved")
                    print("Iteration ", nbIterations, " score: ", score)


            self.draw(isdead)



game = Game()
game.mainLoop()

# Done! Time to quit.
pygame.quit()
