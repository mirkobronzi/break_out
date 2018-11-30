import logging
import random
import time
from collections import namedtuple

import gym

logger = logging.getLogger(__name__)


MoveResult = namedtuple(
    'MoveResult', ['frames', 'next_frames', 'action', 'reward', 'is_done'])


class BreakOut:

    def __init__(self, render=False, n_actions=4):
        self.env = gym.make('BreakoutDeterministic-v4')
        self.env.reset()
        self.render = render
        self.n_actions = n_actions
        self._tot_steps = 0
        self._games_played = 0
        self.lives = 0

    def perform_action(self, action, force_render=False):
        next_frame, reward, is_done, info = self.env.step(action)
        # if self.lives != info['ale.lives']:
        #     # throw the ball
        #     next_frame, reward, is_done, info = self.env.step(1)
        #     self.lives = info['ale.lives']
        if force_render or self.render:
            self.make_render()
        return reward, next_frame, is_done

    def choose_best_action(self, frames):
        return random.randint(0, self.n_actions - 1)

    def choose_and_perform_actions(self, frames):
        action = self.choose_best_action(frames)
        rewards, next_frames, is_done = self.perform_action(action)
        move_result = MoveResult(frames, next_frames, action, rewards,
                                 is_done)
        return move_result

    def make_render(self):
        self.env.render()
        time.sleep(0.05)

    def maybe_render(self):
        if self.render:
            self.make_render()

    def status(self):
        raise NotImplementedError()

    def prepare_frames(self, frames):
        return frames

    def prepare_frame(self, frames):
        return frames

    def _get_first_frames(self):
        self.env.reset()
        self.lives = 0
        # get first frames by playing no action (action=0)
        last_frames = []
        for _ in range(self.n_actions):
            _, last_frame, _ = self.perform_action(0)
            last_frames.append(last_frame)
        last_frames = self.prepare_frames(last_frames)
        return last_frames

    def play_one_game(self, max_moves=99999999):
        frames = self._get_first_frames()
        is_done = False
        count = 0
        while not is_done and count < max_moves:
            action = self.choose_best_action(frames)
            _, next_frame, is_done = self.perform_action(
                action, force_render=True)
            print('{}({}) '.format(count, action), end=' ', flush=True)
            frames = frames[1:] + [self.prepare_frame(next_frame)]
            count += 1
        logger.info('game ended')


def main():
    logging.basicConfig(level=logging.INFO)
    bo = BreakOut()
    bo.play_one_game()


if __name__ == "__main__":
    main()
