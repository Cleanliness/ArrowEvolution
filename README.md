# ArrowEvolution
Genetic algorithm, and NEAT test on a simple 2d game. Slingshots try to shoot enemies, enemies try to survive.

# Requirements
- Python 3
- numpy
- pygame
- neat-python

# Explanation
Enemies and slingshots are controlled by a neural network initially with a fully connected input and output layer. 
Enemy NN inputs are slingshot draw strength in x and y , position, and if a projectile is in one of its quadrants, outputs up, down , left, right movement.
Slingshot inputs are its own draw strength, enemy position, enemy's dy and dx, outputs xy position of 'mouse' and fires if last node's activation is above a certain threshold.

Enemy fitness is a function of time and number of arrows dodged, slingshot fitness is a function of average distance of projectiles from enemy, and if the enemy was hit.
Enemies and slingshots are selected by highest fitness. NEAT crosses over selected individuals and can mutate the structure of their NNs to get more complex behaviour.

# Screenshots
![No Input](https://i.postimg.cc/267rnMy5/midgame.png)

# Videos
[![Youtube](https://img.youtube.com/vi/sRS6lCvJe9U/0.jpg)](https://www.youtube.com/watch?v=sRS6lCvJe9U)
