# Machine Learning Interview Notes

## Fundamentals

### Q: [easy] What is overfitting and how can you prevent it?
Overfitting happens when a model learns the noise and quirks of the training data instead of the underlying pattern, so it scores very well on training data but poorly on unseen data. Ways to prevent it include collecting more training data, simplifying the model, applying regularization such as L1, L2 or dropout, using early stopping, performing data augmentation, pruning decision trees, and using cross-validation to detect the problem. A large gap between training and validation error is the typical warning sign.

### Q: [medium] Explain the bias-variance tradeoff.
Bias is the error from overly simple assumptions: a high-bias model underfits and misses real patterns. Variance is the error from sensitivity to the particular training set: a high-variance model overfits and changes a lot when the data changes. Total expected error is bias squared plus variance plus irreducible noise. Making a model more complex lowers bias but raises variance. The goal is the sweet spot with the lowest validation error, which you find with cross-validation, regularization and ensembling methods such as bagging, which reduces variance, and boosting, which reduces bias.

### Q: [medium] When would you use precision, recall or the F1 score?
Precision is the fraction of predicted positives that are truly positive; recall is the fraction of actual positives that were found. Optimise for precision when false positives are costly, such as flagging legitimate emails as spam. Optimise for recall when false negatives are costly, such as missing a disease or fraud. The F1 score is the harmonic mean of precision and recall and is useful for imbalanced classes where accuracy is misleading. For example, a model that always predicts the majority class in a dataset with 1 percent positives is 99 percent accurate but useless.

### Q: [medium] What is the difference between L1 and L2 regularization?
Both add a penalty on the size of the weights to the loss function to reduce overfitting. L1 regularization, used in lasso, adds the sum of absolute values of the weights and tends to drive some weights exactly to zero, which performs automatic feature selection and produces sparse models. L2 regularization, used in ridge regression and weight decay, adds the sum of squared weights and shrinks all weights smoothly toward zero without eliminating them, which handles correlated features well. Elastic net combines both.

### Q: [medium] Why use cross-validation?
Cross-validation estimates how well a model generalises by training and evaluating it on several different splits of the data. In k-fold cross-validation the data is divided into k folds; each fold takes a turn as the validation set while the others train the model, and the scores are averaged. It gives a more reliable estimate than a single train and test split, helps with hyperparameter tuning, and makes efficient use of small datasets. Use stratified folds for imbalanced classes and time-based splits for time series to avoid leaking future information.

## Deep learning

### Q: [hard] How does backpropagation work?
Backpropagation computes the gradient of the loss with respect to every weight in a neural network so gradient descent can update them. In the forward pass the input flows through the layers to produce a prediction and a loss. In the backward pass the chain rule is applied layer by layer from the output back to the input, reusing intermediate results, so each weight receives its gradient efficiently. The optimizer, such as SGD or Adam, then moves the weights a small step against the gradient, scaled by the learning rate. Problems include vanishing and exploding gradients, mitigated by ReLU activations, careful initialisation, normalisation layers, residual connections and gradient clipping.

### Q: [hard] Explain self-attention in transformers.
Self-attention lets every token in a sequence look at every other token and build a context-aware representation. Each token embedding is projected into a query, a key and a value vector. The attention weights are computed by taking the dot product of a token's query with all keys, scaling by the square root of the key dimension, and applying softmax; the output is the weighted sum of the value vectors. Multi-head attention runs several of these in parallel so the model can capture different relationships. Unlike recurrent networks, attention processes all positions in parallel and handles long-range dependencies well, but its cost grows quadratically with sequence length, and positional encodings are needed because attention itself has no notion of order.
