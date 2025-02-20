# Neural Network Library

Written for recreational programming purposes. 

### Example Output of simple Transformer Model
The model in this test case had 3 transformer layers with 1 attention head
In this basic test, the model was given 40 "sentences" of 8 float values in an increasing order.
The model correctly picked up on the increasing pattern.
``` bash
Transformer Output = [
       -1.608024 -0.711159 -0.725896 -0.770303 0.646998 0.964390 1.028893 1.175101 
       -1.607983 -0.711125 -0.725947 -0.770385 0.647141 0.964435 1.028831 1.175032 
       -1.607942 -0.711091 -0.725997 -0.770467 0.647285 0.964479 1.028770 1.174962 
       ...
       -1.606504 -0.709886 -0.727756 -0.773323 0.652294 0.966027 1.026622 1.172526 
       -1.606463 -0.709851 -0.727806 -0.773405 0.652436 0.966073 1.026560 1.172457 
       -1.606421 -0.709818 -0.727858 -0.773486 0.652582 0.966116 1.026498 1.172387 
     ]
```

### Example Output of MNIST classification
A take on the classic MNIST dataset classification problem
Achieving approximately 90% accuracy with only a
Feed Forward style approach and C. No external libraries for the neural network or linear algebra needed.
``` bash
...
Predicted: 2, Actual: 2
Predicted: 4, Actual: 4
Predicted: 9, Actual: 9
Predicted: 4, Actual: 4
Predicted: 2, Actual: 3
Predicted: 6, Actual: 6
Predicted: 4, Actual: 4
Predicted: 1, Actual: 1
Predicted: 7, Actual: 7
Test Accuracy: 89.54%
```

### Example output of XOR
A simple Neural Network model learning to replicate XOR
``` bash
Backprop complete
Batch cost: 19.133190
Updated batch begin: 4
Batch processing finished. Final average cost: 9.566595

Testing XOR:
0 XOR 0 = 0.004199
0 XOR 1 = 0.996128
1 XOR 0 = 0.994307
1 XOR 1 = 0.003592
```