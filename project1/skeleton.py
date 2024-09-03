import random
import gym
import requests
import numpy as np
import argparse
import sys
from gym_connect_four import ConnectFourEnv
from math import inf
from copy import deepcopy
import time
env: ConnectFourEnv = gym.make("ConnectFour-v0")

SERVER_ADDRESS = "https://vilde.cs.lth.se/edap01-4inarow/"
API_KEY = 'nyckel'
STIL_ID = ["so4816ko-s", "vi0713ba-s"] 
BIG_NUMBER = -inf 
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
   env.change_player() # change to oppoment
   avmoves = env.available_moves()
   if not avmoves:
      env.change_player() # change back to student before returning
      return -1

   #so that the user can play against the agent
   print(env.board)
   action = int(input("Your move (between 0 and 6): ")) 
   #action = random.choice(list(avmoves))

   state, reward, done, _ = env.step(action)
   if done:
      if reward == 1: # reward is always in current players view
         reward = -1
   env.change_player() # change back to student before returning
   return state, reward, done

def student_move(env):
   """
   Implement your min-max alpha-beta pruning algorithm here.
   Give it whatever input arguments you think are necessary
   (and change where it is called).
   The function should return a move from 0-6
   """
   #definition of adversarial_search from the lecture
   start_time = time.time()
   best_action = None
   best_score = -inf   
   for a in env.available_moves():
      env_copy = deepcopy(env)
      new_s = env_copy.step(a)
      score = SCORE(new_s, env_copy, best_score)
      if score > best_score:
         best_score = score
         best_action = a
   print("time for action: ", time.time() - start_time)
   return best_action 

MAX_DEPTH = 4

#using the minimax algorithm implementation from the lecture notes, with the alpha beta pruning
#doing deep copies at every step to preserve the real state of the board
def SCORE(state, env_c, alpha):
   return min_player(state, env_c, alpha, inf, depth = 1)

def min_player(state, env_c, alpha, beta, depth = 1):
   env_c.change_player()
   done = state[2] #done return value in step
   if done: return state[1] #reward return value in step

   elif depth == MAX_DEPTH: return eval(env_c) 

   best_score = inf
   for a in env_c.available_moves():
      env_copy = deepcopy(env_c)
      best_score = min(best_score, max_player(env_copy.step(a), env_copy, alpha, beta, depth = depth + 1))
      if best_score <= alpha: break
   return best_score

def max_player(state, env_c, alpha, beta, depth = 1):
   env_c.change_player()
   done = state[2]
   if done: return state[1]

   elif depth == MAX_DEPTH: return eval(env_c) 

   
   best_score = -inf
   for a in env_c.available_moves():
      env_copy = deepcopy(env_c)
      best_score = max(best_score, min_player(env_copy.step(a), env_copy, alpha, beta, depth = depth + 1)) 
      alpha = max(alpha, best_score)
      if best_score >= beta: break
   return best_score

def evaluate_score(block):
   minimum = min(block)
   maximum = max(block)
   streak = sum(block)
   #if i have 4 in a row in that block
   if streak == 4: return 100000
   #if the opponent has 4 in a row
   elif streak == -4: return -1000000
   #if i have 3 in a row and if there is space to have 1 more 
   elif streak == 3 and minimum >= 0: return 100
   #if the opponent has 3 in a row
   elif streak == -3 and maximum <= 0: return -200
   #if i block the opponent = nice
   elif streak == -3 and minimum >= 0: return 50
   #if i have 2 i a row
   elif streak == 2 and np.count_nonzero(block == 0) == 2: return 2
   #if the opponent has 2 in a row
   elif streak == -2 and np.count_nonzero(block == 0) == 2: return -2
   else:
      return 0
    
SHAPE_0 = 6
SHAPE_1 = 7
def eval(env_c):
   score = 0
   env_copy = deepcopy(env_c)
   board = env_copy.board

	#for each row, column and diagonal, check if there are some discs aligned

   for row in range(SHAPE_0):
      row_array = np.array(board[row])
      for spot in range(4):
         block = np.array([row_array[spot + i] for i in range(4)])
         score += evaluate_score(block)

	#columns
   for col in range(SHAPE_1):
      col_array = np.array([board[row][col] for row in range(SHAPE_0 - 1, -1, -1)])

      for spot in range(3):
         block = np.array([col_array[spot + i] for i in range(4)])
         score += evaluate_score(block)

	#diagonals
   for row in range(3):
      for col in range(4):
         diag_array = np.array([board[5-row-i][col+i] for i in range(4)])
         score += evaluate_score(diag_array)
   
   for row in range(3):
      for col in range(4):
       four = np.array([board[row+i][col+i] for i in range(4)])
       score += evaluate_score(four)

   #if i have the middle row, it's good (allows more combinations)
   if board[5][3] == 1: score += 2
   if board[5][3] == -1: score += -2
   return score


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
         print('Bot (agent) start!')
         print()
      else:
         print('You (user) start!')
         print()

   # Print current gamestate
   print("Current state (1 are student discs, -1 are servers, 0 is empty): ")
   print(state)
   print()

   done = False
   while not done:
      # Select your move
      stmove = student_move(env) 

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
               print("You tried to make an illegal move! You have lost the game.")
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
            print("You (bot) won!")
         elif result == 0.5:
            print("It's a draw!")
         elif result == -1:
            print("You (bot) lost!")
         elif result == -10:
            print("You (bot) made an illegal move and have lost!")
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
