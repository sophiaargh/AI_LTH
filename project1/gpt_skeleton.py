import gym
import random
import requests
import numpy as np
import argparse
import sys
import time
from gym_connect_four import ConnectFourEnv

env: ConnectFourEnv = gym.make("ConnectFour-v0")

SERVER_ADDRESS = "https://vilde.cs.lth.se/edap01-4inarow/"
API_KEY = 'nyckel'
STIL_ID = ["so4816ko-s", "ChatGPT"] # fill this list with your stil-id's

def call_server(move):
   res = requests.post(SERVER_ADDRESS + "move",
                       data={
                           "stil_id": STIL_ID,
                           "move": move, # -1 signals the system to start a new game. any running game is counted as a loss
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
   env.change_player() # change to oppoent
   avmoves = env.available_moves()
   if not avmoves:
      env.change_player() # change back to student before returning
      return -1

   # TODO: Optional? change this to select actions with your policy too
   # that way you get way more interesting games, and you can see if starting
   # is enough to guarrantee a win
   action = random.choice(list(avmoves))

   state, reward, done, _ = env.step(action)
   if done:
      if reward == 1: # reward is always in current players view
         reward = -1
   env.change_player() # change back to student before returning
   return state, reward, done


def student_move():
    """
    TODO: Implement your min-max alpha-beta pruning algorithm here.
    Give it whatever input arguments you think are necessary
    (and change where it is called).
    The function should return a move from 0-6
    """
    max_depth = 3  # You can adjust the depth based on available computation time
    start_time = time.time()
    def max_value(state, alpha, beta, depth):
        if depth == 0 or env.is_win_state() or len(env.available_moves()) == 0:
            return evaluate_state(state)
        
        value = float("-inf")
        for action in env.available_moves():
            state, reward, done, _ = env.step(action) #changed here
            value = max(value, min_value(env.board, alpha, beta, depth - 1))
            env.reset(board = state)
            if done:
                return value
            alpha = max(alpha, value)
            if alpha >= beta:
                return value
        return value
    
    def min_value(state, alpha, beta, depth):
        if depth == 0 or env.is_win_state() or len(env.available_moves()) == 0:
            return evaluate_state(state)
        
        value = float("inf")
        for action in env.available_moves():
            state, reward, done, _ = env.step(action) #changed here
            value = min(value, max_value(env.board, alpha, beta, depth - 1))
            env.reset(board = state)
            if done:
                return value
            beta = min(beta, value)
            if alpha >= beta:
                return value
        return value
    
    def evaluate_state(state):
        #first attempt:
        # Simple evaluation function
        # You may want to improve this based on the current state
        #return np.sum(state) / 10.0  # Adjust the scale based on the importance of the evaluation
      score = 0

      # Check horizontally
      for row in state:
         score += evaluate_line(row)

      # Check vertically
      for col in state.T:
         score += evaluate_line(col)

      # Check diagonally
      for i in range(-2, 4):
         diagonal = np.diagonal(state, offset=i)
         score += evaluate_line(diagonal)

      for i in range(1, 4):
         diagonal = np.diagonal(np.fliplr(state), offset=i)
         score += evaluate_line(diagonal)

      return score

    def evaluate_line(line):
         score = 0
         count_player = 0
         count_opponent = 0

         for cell in line:
            if cell == 1:
                  count_player += 1
                  count_opponent = 0
            elif cell == -1:
                  count_opponent += 1
                  count_player = 0
            else:
                  count_player = 0
                  count_opponent = 0

            # Evaluate based on connected discs
            if count_player == 4:
                  score += 1000
            elif count_opponent == 4:
                  score -= 1000
            elif count_player == 3:
                  score += 100
            elif count_opponent == 3:
                  score -= 100
            elif count_player == 2:
                  score += 10
            elif count_opponent == 2:
                  score -= 10

         return score

       
        
    best_move = -1
    best_value = float("-inf")
    for action in env.available_moves():
        state_copy = np.copy(env.board)
        state, reward, done, _ = env.step(action) #changed here
        value = min_value(env.board, float("-inf"), float("inf"), max_depth - 1)
        env.reset(board = state_copy)
        if value > best_value:
            best_value = value
            best_move = action
    
    print("time for action: ", time.time() - start_time)
    return best_move


def play_game(vs_server = False):
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
      res = call_server(-1) # -1 signals the system to start a new game. any running game is counted as a loss

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
      stmove = student_move() # TODO: change input here

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

         student_gets_move = True # student only skips move first turn if bot starts

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
   group.add_argument("-l", "--local", help = "Play locally", action="store_true")
   group.add_argument("-o", "--online", help = "Play online vs server", action="store_true")
   parser.add_argument("-s", "--stats", help = "Show your current online stats", action="store_true")
   args = parser.parse_args()

   # Print usage info if no arguments are given
   if len(sys.argv)==1:
      parser.print_help(sys.stderr)
      sys.exit(1)

   if args.local:
      play_game(vs_server = False)
   elif args.online:
      play_game(vs_server = True)

   if args.stats:
      stats = check_stats()
      print(stats)

   # TODO: Run program with "--online" when you are ready to play against the server
   # the results of your games there will be logged
   # you can check your stats bu running the program with "--stats"

if __name__ == "__main__":
    main()
