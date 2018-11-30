import argparse
import logging
import random
from collections import namedtuple, deque

import numpy as np
from PIL import Image


from break_out import BreakOut, MoveResult
from models.model_keras import ModelKeras
from models.model_pytorch import ModelPyTorch

logger = logging.getLogger(__name__)


MoveBatch = namedtuple(
    'MoveBatch', ['frame_batch', 'next_frame_batch', 'action_batch',
                  'reward_batch', 'is_done_batch'])

Y_SHAPE = 105
X_SHAPE = 80


def one_hot(action, n_actions):
    b = np.zeros(n_actions)
    b[action] = 1
    return b


def _prepare_batch_for_fit(batch, n_actions):
    frame_batch = np.stack([x.frames for x in batch])
    next_frame_batch = np.stack([x.next_frames for x in batch])
    action_batch  = np.stack([one_hot(x.action, n_actions) for x in batch])
    reward_batch = np.stack([x.reward for x in batch])
    is_done_batch = np.stack([x.is_done for x in batch])
    return MoveBatch(frame_batch, next_frame_batch, action_batch, reward_batch,
                     is_done_batch)


class BreakOutML(BreakOut):

    def __init__(self, model, render=False,  n_actions=4, gamma=0.99,
                 batch_size=32, debug=False, epsilon_annihilation=20000):
        super(__class__, self).__init__(render, n_actions)
        self.model = model
        self.frames = None
        self._random_actions = 0
        self._model_actions = 0
        self.actions = {0: 0, 1: 0, 2: 0, 3: 0}
        self.gamma = gamma
        logger.info('gamma is {}'.format(gamma))
        self.batch_size = batch_size
        self.debug = debug
        self.epsilon_annihilation = epsilon_annihilation
        self.min_epsilon = 0.05
        self._total_reward = 0
        self._total_moves = 0
        self._last_n_actions = deque()
        self.same_model = 0
        self.diff_model = 0

    def _show_frames(self, move):
        logger.info('reward {}'.format(move.reward))
        logger.info('action {}'.format(move.action))
        self.env.render()
        logger.info('start frames')
        for i in range(self.n_actions):
            img1 = Image.fromarray(move.frames[i], 'L')
            img1.show()
        input()
        logger.info('end frames')
        for i in range(self.n_actions):
            img1 = Image.fromarray(move.next_frames[i], 'L')
            img1.show()
        input()

    def _fit_batch(self, batch):
        assert batch.frame_batch.shape[0] == self.batch_size, \
            "frames has shape {} - but batch size is {}".format(
                batch.frame_batch.shape, self.batch_size
            )
        next_q_values = self.model.predict(
            batch.next_frame_batch,
            np.ones([self.batch_size, self.n_actions]))
        next_q_values[batch.is_done_batch] = 0
        q_values = batch.reward_batch + (
                self.gamma * np.max(next_q_values, axis=1))

        target = np.expand_dims(q_values, axis=1) * batch.action_batch

        before = self.model.predict(batch.frame_batch, batch.action_batch)
        self.model.fit(
            batch.frame_batch, batch.action_batch,
            target
        )
        after = self.model.predict(batch.frame_batch, batch.action_batch)
        if np.array_equal(before, after):
            self.same_model += 1
        else:
            self.diff_model += 1

    def get_epsilon_for_iteration(self):
        current_epsilon = max(
            1 - (float(self._tot_steps) / self.epsilon_annihilation),
            self.min_epsilon)
        return min(
            current_epsilon,
            1.0)

    def prepare_frames(self, frames):
        result = []
        for frame in frames:
            result.append(self.prepare_frame(frame))
        return result

    def prepare_frame(self, frame):
        half_res_frame = frame[::2, ::2]
        removed_score = half_res_frame[-Y_SHAPE:, :, :]
        grayscale_frame = np.mean(removed_score, axis=2).astype(np.uint8)
        return grayscale_frame

    def prepare_reward(self, reward):
        modified_reward = reward
        # if all([x == self._last_n_actions[0] for x in self._last_n_actions]):
        #     modified_reward -= 1
        # return modified_reward
        return np.sign(modified_reward)

    def choose_best_action(self, frames):
        # FIXME: should we go batch wise? for now it is just one element..
        frames = np.expand_dims(frames, axis=0)
        return np.argmax(
            self.model.predict(frames, np.ones([1, self.n_actions])))

    def _store_action(self, action):
        # keeping track of the last n actions
        assert len(self._last_n_actions) <= 4
        if len(self._last_n_actions) == 4:
            self._last_n_actions.popleft()
        self._last_n_actions.append(action)

    def choose_and_perform_actions(self, frames, epsilon=None):
        if epsilon is None:
            epsilon = self.get_epsilon_for_iteration()
        if random.random() < epsilon:
            action = random.randint(0, self.n_actions - 1)
            self._random_actions += 1
        else:
            action = self.choose_best_action(frames)
            self._model_actions += 1
        self._store_action(action)
        self.actions[action] += 1
        reward, next_frame, is_done = self.perform_action(action)
        next_frames = frames[1:] + [self.prepare_frame(next_frame)]
        reward = self.prepare_reward(reward)
        move_result = MoveResult(frames, next_frames, action, reward,
                                 is_done)
        if self.debug and (move_result.reward > 0) and self._games_played > 100:
            self._show_frames(move_result)
        return move_result

    def _get_batch(self, frames):
        batch = []
        for _ in range(self.batch_size):
            move = self.choose_and_perform_actions(frames)
            self._total_reward += move.reward
            self._total_moves += 1
            self._tot_steps += 1
            frames = move.next_frames
            batch.append(move)
            if move.is_done:
                frames = self._get_first_frames()
                self._games_played += 1
                self.print_stats()
        return batch, frames

    def print_stats(self):
        logger.info(
            "done game {:3d} / steps {:6d} | avg reward {:2.3f} - "
            "avg moves {:3.2f} | total: epsilon {:.3f} - rand {:6d} / "
            " model {:6d} - actions {} - same {} / diff {}".format(
                self._games_played, self._tot_steps,
                self._total_reward / self._games_played,
                self._total_moves / self._games_played,
                self.get_epsilon_for_iteration(),
                self._random_actions, self._model_actions,
                self.compute_action_percentage(),
                self.same_model, self.diff_model))

    def compute_action_percentage(self):
        tot_actions = sum(self.actions.values())
        action_percentages = ["{}: {:2.0f}%".format(
            k, (v / tot_actions) * 100) for k, v in self.actions.items()]
        return ' | '.join(action_percentages)

    def train_n_steps(self, steps):
        frames = self._get_first_frames()
        while self._tot_steps <= steps:
            batch, frames = self._get_batch(frames)
            batch = _prepare_batch_for_fit(batch, self.n_actions)
            self._fit_batch(batch)
            self.maybe_render()

    def save(self, path):
        self.model.save(path)


def main():
    parser = argparse.ArgumentParser(description='atari ML')
    define_options(parser)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    model = get_models(args)
    boml = BreakOutML(model=model, debug=args.debug,
                      n_actions=args.n_actions, render=args.render,
                      epsilon_annihilation=args.epsilon_annihilation,
                      batch_size=args.batch_size, gamma=args.gamma)
    if args.play:
        boml.play_one_game()
    else:
        boml.train_n_steps(args.steps)
        boml.save(args.model_path)


def get_models(args):
    if args.model_type == 'keras':
        model = ModelKeras(args.n_actions, X_SHAPE, Y_SHAPE, args.batch_size,
                           args.model_path)
    elif args.model_type == 'pytorch':
        model = ModelPyTorch(args.n_actions, X_SHAPE, Y_SHAPE, args.batch_size,
                             args.model_path)
    else:
        raise ValueError('mode type {} not supported'.format(args.model_type))
    return model


def define_options(parser):
    parser.add_argument('--model-path',
                        help='will save the model to this file - if '
                             'already present, will load from it too',
                        required=True),
    parser.add_argument('--model-type',
                        help='model type - keras or pytorch',
                        default='keras'),
    parser.add_argument('--steps',
                        help='for how many steps to train',
                        default=50000,
                        type=int)
    parser.add_argument('--n-actions',
                        help='how many frames/actions in one state',
                        default=4,
                        type=int)
    parser.add_argument('--epsilon-annihilation',
                        default=200000,
                        type=int)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--play',
                        help='will play a game',
                        action='store_true')
    parser.add_argument('--render',
                        help='will render (even during train)',
                        action='store_true')
    parser.add_argument('--batch-size',
                        default=32,
                        type=int)
    parser.add_argument('--debug',
                        action='store_true')


if __name__ == "__main__":
    main()
