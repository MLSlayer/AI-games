# This project has games you can play against where the games utilize AI in many forms

# Currently implemented are

___

## 1. Tic-Tac-Toe
#### Difficulty: easy
* [x] Minimax
  * [x] Pruning 
    * Unbeatable, this is a zero sum game is why.
* [x] Q-learning (reinforcement)
  * [ ] Experience replay(probably unnecessary because tic tac toe states are so small our net most likely just memorized...)
    * Very good, but maybe beatable, haven't wrote a test script
  
### Basic usage
    Load this files up in an IDE and mark AI-games as sources root. 
    
    Run mini_max_prune.py or q_reinforcement_learning_play.py
___
## 2. Gridworld
#### Difficulty: easy
* [x] Policy Iteration
* [x] Value Iteration
* [x] On Policy Monte Carlo Control
* [x] SARSA / on policy td control
* [ ] Q-learning (reinforcement)
* [ ] etc
    * [ ] a* style kind of search alg. May not do because will be tedious so..
  
### Basic usage
    Load this files up in an IDE and mark AI-games as sources root. 
    
    Run policy_iteration.py
___

