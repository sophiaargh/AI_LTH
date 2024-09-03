import gym
import random
import requests
import numpy as np
import argparse
import sys
import math
import copy
from gym_connect_four import ConnectFourEnv

env: ConnectFourEnv = gym.make("ConnectFour-v0")

SERVER_ADDRESS = "https://vilde.cs.lth.se/edap01-4inarow/"
API_KEY = 'nyckel'
STIL_ID = ["vi07313ba-s", "so4816ko-s"]  # fill this list with your stil-id's


def call_server(move):
    res = requests.post(SERVER_ADDRESS + "move",
                        data={
                            "stil_id": STIL_ID,
                            "move": move,
                            # -1 signals the system to start a new game. any running game is counted as a loss
                            "api_key": API_KEY,
                        })
    # For safety some respose checking is done here
    if res.status_code != 200:
        print("Server gave a bad response, error code={}".format(res.status_code))
        exit()
    if not res.json()['status']:
        print("Server returned a bad status. Return message: ")
        print(res.json()['msg'])
        exit()
    return res


def check_stats():
    res = requests.post(SERVER_ADDRESS + "stats",
                        data={
                            "stil_id": STIL_ID,
                            "api_key": API_KEY,
                        })

    stats = res.json()
    return stats


"""
You can make your code work against this simple random agent
before playing against the server.
It returns a move 0-6 or -1 if it could not make a move.
To check your code for better performance, change this code to
use your own algorithm for selecting actions too
"""


def opponents_move(env):
    env.change_player()  # change to oppoent
    avmoves = env.available_moves()
    if not avmoves:
        env.change_player()  # change back to student before returning
        return -1

    # TODO: Optional? change this to select actions with your policy too
    # that way you get way more interesting games, and you can see if starting
    # is enough to guarrantee a win
    action = random.choice(list(avmoves))
    # action = input("Give a move: ", avmoves)

    state, reward, done, _ = env.step(action)
    if done:
        if reward == 1:  # reward is always in current players view
            reward = -1
    env.change_player()  # change back to student before returning
    return state, reward, done


def evaluate_block(block):
    """
    Determines all possible 4 token sections of the board, and scores
    each block based on how many sequential tokens are in each block

    block - array containing the number of tokens placed on the board
    token - player piece
    """
    score = 0

    if np.count_nonzero(block == 1) == 4:
        score += 9999
    # three tokens are in sequence
    elif np.count_nonzero(block == 1) == 3 and np.count_nonzero(block == 0) == 1:
        score += 5
    # two tokens are in sequence
    elif np.count_nonzero(block == 1) == 2 and np.count_nonzero(block == 0) == 2:
        score += 2

    # Punish player because left opponent open to win
    if np.count_nonzero(block == -1) == 3 and np.count_nonzero(block == 0) == 1:
        score -= 5

    return score


def score_token_position(state, token):
    """
    calculates the score based on how many tokens are connected
    token - determines whether the player is human or ai
    state - current state of the board
    """
    score = 0
    row_size = state.shape[0]
    column_size = state.shape[1]

    # Give higher weight to token placed in center column
    # Want to give them preference due to
    # advantage of higher chance of winning
    center_col = state[:, 3]
    center_count = np.count_nonzero(center_col == token)
    score += center_count * 6

    # horizontal
    for row in range(row_size):
        row_list = state[row]
        for column in range(column_size - 3):
            block = row_list[column:column + 4]
            score += evaluate_block(block)

    # vertical
    for column in range(column_size):
        column_list = state[:, column]
        for row in range(row_size - 3):
            block = column_list[row:row + 4]
            score += evaluate_block(block)

    # diagonal
    for row in range(row_size - 3):
        for column in range(column_size - 3):
            block = [state[row + token_sequence_size][column + token_sequence_size] for token_sequence_size in range(4)]
            score += evaluate_block(block)

    # reverse diagonal
    for row in range(row_size - 3):
        for column in range(column_size - 3):
            block = [state[row + 3 - token_sequence_size][column + token_sequence_size] for token_sequence_size in
                     range(4)]
            score += evaluate_block(block)

    return score


def student_move():
    """
    TODO: Implement your min-max alpha-beta pruning algorithm here.
    Give it whatever input arguments you think are necessary
    (and change where it is called).
    The function should return a move from 0-6
    """
    best_move, value = min_max(env, 5, -math.inf, math.inf, True)
    return best_move


def min_max(env, depth, alpha, beta, max_player):
    """
    Min-max alpha-beta pruning algorithm

    depth: how many moves ahead we want to search
    max_player: flag that represents if its players turn
    env: current state of the game
    """
    player_moves = env.available_moves()

    # terminal node: opponent wins, player wins, or no more pieces
    if depth == 0 or len(player_moves) == 0 or env.is_win_state():
        return None, score_token_position(env.board, True)

    random.shuffle(list(player_moves))

    if max_player:
        max_value = -math.inf
        # Use center column as default
        best_move = 3
        for move in player_moves:
            new_env = copy.deepcopy(env)
            new_env.step(move)
            new_env.change_player()

            # no need for the move that will be returned with this function call as its not needed hence
            # why it is set to a temp variable
            _, value = min_max(new_env, depth - 1, alpha, beta, False)
            if max_value < value:
                max_value = value
                best_move = move
            alpha = max(alpha, value)
            if beta <= alpha:
                break
        return best_move, max_value
    else:
        min_value = math.inf
        best_move = 3
        for move in player_moves:
            new_env = copy.deepcopy(env)
            new_env.step(move)
            new_env.change_player()

            _, value = min_max(new_env, depth - 1, alpha, beta, True)
            if min_value > value:
                min_value = value
                best_move = move
            beta = min(beta, value)
            if beta <= alpha:
                break
        return best_move, min_value


def play_game(vs_server=False):
    """
   The reward for a game is as follows. You get a
   botaction = random.choice(list(avmoves)) reward from the
   server after each move, but it is 0 while the game is running
   loss = -1
   win = +1
   draw = +0.5
   error = -10 (you get this if you try to play in a full column)
   Currently the player always makes the first move
   """

    # default state
    state = np.zeros((6, 7), dtype=int)

    # setup new game
    if vs_server:
        # Start a new game
        res = call_server(-1)  # -1 signals the system to start a new game. any running game is counted as a loss

        # This should tell you if you or the bot starts
        print(res.json()['msg'])
        botmove = res.json()['botmove']
        state = np.array(res.json()['state'])
        # reset env to state from the server (if you want to use it to keep track)
        env.reset(board=state)
    else:
        # reset game to starting state
        env.reset(board=None)
        # determine first player
        student_gets_move = random.choice([True, False])
        if student_gets_move:
            print('You start!')
            print()
        else:
            print('Bot starts!')
            print()

    # Print current gamestate
    print("Current state (1 are student discs, -1 are servers, 0 is empty): ")
    print(state)
    print()

    done = False
    while not done:
        # Select your move
        # Pass
        stmove = student_move()

        # make both student and bot/server moves
        if vs_server:
            # Send your move to server and get response
            res = call_server(stmove)
            print(res.json()['msg'])

            # Extract response values
            result = res.json()['result']
            botmove = res.json()['botmove']
            state = np.array(res.json()['state'])
            # reset env to state from the server (if you want to use it to keep track)
            env.reset(board=state)
        else:
            if student_gets_move:
                # Execute your move
                avmoves = env.available_moves()
                if stmove not in avmoves:
                    print("You tied to make an illegal move! You have lost the game.")
                    break
                state, result, done, _ = env.step(stmove)

            student_gets_move = True  # student only skips move first turn if bot starts

            # print or render state here if you like

            # select and make a move for the opponent, returned reward from students view
            if not done:
                state, result, done = opponents_move(env)

        # Check if the game is over
        if result != 0:
            done = True
            if not vs_server:
                print("Game over. ", end="")
            if result == 1:
                print("You won!")
            elif result == 0.5:
                print("It's a draw!")
            elif result == -1:
                print("You lost!")
            elif result == -10:
                print("You made an illegal move and have lost!")
            else:
                print("Unexpected result result={}".format(result))
            if not vs_server:
                print("Final state (1 are student discs, -1 are servers, 0 is empty): ")
        else:
            print("Current state (1 are student discs, -1 are servers, 0 is empty): ")

        # Print current gamestate
        print(state)
        print()


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-l", "--local", help="Play locally", action="store_true")
    group.add_argument("-o", "--online", help="Play online vs server", action="store_true")
    parser.add_argument("-s", "--stats", help="Show your current online stats", action="store_true")
    args = parser.parse_args()

    # Print usage info if no arguments are given
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    if args.local:
        play_game(vs_server=False)
    elif args.online:
        play_game(vs_server=True)

    if args.stats:
        stats = check_stats()
        print(stats)

    # TODO: Run program with "--online" when you are ready to play against the server
    # the results of your games there will be logged
    # you can check your stats bu running the program with "--stats"


if __name__ == "__main__":
    main()
