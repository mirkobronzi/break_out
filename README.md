# what is this

goal: try to replicate the results of a RL agent playing break-out

see here for info/context: https://deepmind.com/research/publications/playing-atari-deep-reinforcement-learning/

# instructions

this project provides 3 way to play the breakout game:

randomly (break_out.py)

using Q-learning (break_out_ml.py)- learning while playing (i.e., training
will process data in the same order as in a game - which does not provide
good results)

using Q-learning and memory reply organized by cycles (break_out_ml.py) - i.e.,
the system will play N moves, then shuffle, then train

using Q-learning and memory reply randomly updated
(break_out_ml.py --memory-updates) - i.e., the system will fill a memory, then
play one step starting from a random state from the memory, and it will put the
result in a random position in the memory

use --help to see the various options


