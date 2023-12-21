## My proposition is to split our dataset with many features into multiple datasets with fewer features.
## It will speed up the training process and may improve quality in certain conditions.

Let's denote the original dataset as $\mathcal{D}$ with $N$ examples and $M$ features. We split $\mathcal{D}$ into $K$ smaller datasets ${\mathcal{D}_1, \mathcal{D}_2, \ldots, \mathcal{D}_K}$, where each $\mathcal{D}_i$ contains $N$ examples and $M/k$ features.

For each $\mathcal{D}_k$, we train a logistic regression model. After training, we obtain the weight vector $\mathbf{w}_k$ and bias vector $\mathbf{b}_k$.

Let $\mathbf{W} = [\mathbf{w}_1, \mathbf{w}_2, \ldots, \mathbf{w}_K]$ be the matrix as a result of concatenating the weight matrices of all logistic regression models.
And $\mathbf{b} = \mathbf{b}_1 + \mathbf{b}_2 + ... + \mathbf{b}_k$ be the bias vector as a result of summing up the biases of all logistic regression models.

### Now we can make predictions with this combined model.

# Open Presentation.ipynb for demonstation of usage. 

# Code for class can be found at .py file.
