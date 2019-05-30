# blackjack
In [the main notebook of this repository](https://github.com/slmwest/blackjack/blob/master/learn_blackjack.ipynb) I'll show how Q-learning can be used by an agent with no prior knowledge to teach itself to play Blackjack. While I don't expect the agent to find a way to make money from gambling, I am interested in how artifical intelligence trained over hundreds of thousands of games fares against top human performance. Indeed, by the end of this notebook we will show that a model-free agent can learn to play at near-human levels.

To make the comparison we will simulate games under a strategy recommended by a human expert, using the simulation's mean rewards over last 25,000 games as the benchmark. Since mean rewards are expected to be negative in all cases (human or otherwise), we can also reframe this standard RL evaluation metric as something more easily understood to wider audiences: "expected value of funds after 100 games". Using the human strategy, we find that the mean expected value of £100 after 100 games would be £96.65. We will treat this figure as a benchmark for what a "100% accurate" solution looks like.

We show that with the help of a grid search and some additional fine-tuning, we can produce an agent whose expected value of £100 after 100 games is £96.39. In other words, **our AI reaches 99.7% of human performance in under an hour**.

In terms of what the human strategy and best agent's strategy looks like, we can show this by making some "advice charts". S stands for Stick, H for Hit and D for Double, plus the 1 in Dealer's initial hand stands for an Ace.

![Human strategy](https://github.com/slmwest/blackjack/blob/master/additional_analysis/agent_strategy_chart.png)
![AI strategy](https://github.com/slmwest/blackjack/edit/master/additional_analysis/agent_strategy_chart.png)

### Special thanks:
My implementation relies heavily on the following key sources, which I highly recommend taking a closer look at:
 - https://github.com/openai/gym/blob/master/gym/envs/toy_text/blackjack.py
 - https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0\
-  https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-7-action-selection-strategies-for-exploration-d3a97b7cceaf\

   
