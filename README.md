Environement:


Need Update (/pygame\)

Python3
Numpy
Tensorflow 1.12

Algorythm:

1 Convolution layer of size 32.
A padding of 1, a kernel of 2 and a jump of 1. Activation with Relu.

1 Convolution layer of size 64.
A padding of 1, a kernel of 2 and a jump of 1. Activation with Relu.

1 Hidden layer of size 512.
Activation with Relu.

Output of size 4, with Sigmoid.

Reinforcement with Policy gradient.

Environement:

The player is in a 20x20 map.
He can make 4 actions: Up, Down, Right, Left
He has 20 life, and 100 food.
Map have 2 elements, earth and water.
When the player stand on earth, he suffer 1 damage.
Fish and Sea Weed can be find in water.
When the player stand on a Fish or a Sea Weed, he eat it and win 1 life (max 20) and 20 food (max 100).
When the player eat, he has a reward.
The Player lose 1 food for each move, if he has 0 or less food, he lose 1 life instead.
Each time the player lose 1 life, he lose reward.
The player has constantly a reward equivalent to the distance between him and the nearest food.
The player can see 4 cell around him (a 9x9 matrix where he is in [4, 4])

