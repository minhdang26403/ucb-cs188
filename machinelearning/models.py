import nn


class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        return nn.DotProduct(self.w, x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        return 1 if nn.as_scalar(self.run(x)) >= 0 else -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        converged = False
        while not converged:
            converged = True
            for x, y in dataset.iterate_once(1):
                if self.get_prediction(x) != nn.as_scalar(y):
                    self.w.update(x, nn.as_scalar(y))
                    converged = False


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """

    def __init__(self):
        # Initialize your model parameters here
        self.W1 = nn.Parameter(1, 128)
        self.b1 = nn.Parameter(1, 128)
        self.W2 = nn.Parameter(128, 64)
        self.b2 = nn.Parameter(1, 64)
        self.W3 = nn.Parameter(64, 1)
        self.b3 = nn.Parameter(1, 1)
        self.params = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        first_layer = nn.ReLU(nn.AddBias(nn.Linear(x, self.W1), self.b1))
        second_layer = nn.ReLU(nn.AddBias(nn.Linear(first_layer, self.W2), self.b2))
        output_layer = nn.AddBias(nn.Linear(second_layer, self.W3), self.b3)

        return output_layer

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        batch_size = 50
        learning_rate = 0.01
        for x, y in dataset.iterate_forever(batch_size):
            loss = self.get_loss(x, y)
            if nn.as_scalar(loss) < 0.01:
                return

            gradients = nn.gradients(loss, self.params)
            for param, grad in zip(self.params, gradients):
                param.update(grad, -learning_rate)


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        # Initialize your model parameters here
        self.W1 = nn.Parameter(784, 256)
        self.b1 = nn.Parameter(1, 256)
        self.W2 = nn.Parameter(256, 128)
        self.b2 = nn.Parameter(1, 128)
        self.W3 = nn.Parameter(128, 10)
        self.b3 = nn.Parameter(1, 10)
        self.params = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        first_layer = nn.ReLU(nn.AddBias(nn.Linear(x, self.W1), self.b1))
        second_layer = nn.ReLU(nn.AddBias(nn.Linear(first_layer, self.W2), self.b2))
        output_layer = nn.AddBias(nn.Linear(second_layer, self.W3), self.b3)

        return output_layer

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        batch_size = 100
        learning_rate = 0.1

        for x, y in dataset.iterate_forever(batch_size):
            loss = self.get_loss(x, y)
            gradients = nn.gradients(loss, self.params)
            for param, grad in zip(self.params, gradients):
                param.update(grad, -learning_rate)

            if dataset.get_validation_accuracy() > 0.98:
                return


class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]
        self.hidden_layer_size = 100

        # Initialize your model parameters here
        self.W_x = nn.Parameter(self.num_chars, self.hidden_layer_size)
        self.W_hidden = nn.Parameter(self.hidden_layer_size, self.hidden_layer_size)
        self.b_hidden = nn.Parameter(1, self.hidden_layer_size)
        self.W_output = nn.Parameter(self.hidden_layer_size, len(self.languages))
        self.b_output = nn.Parameter(1, len(self.languages))
        self.params = [
            self.W_x,
            self.W_hidden,
            self.b_hidden,
            self.W_output,
            self.b_output,
        ]

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        h_i = nn.ReLU(nn.AddBias(nn.Linear(xs[0], self.W_x), self.b_hidden))
        for x in xs[1:]:
            h_i = nn.ReLU(
                nn.AddBias(
                    nn.Add(nn.Linear(x, self.W_x), nn.Linear(h_i, self.W_hidden)),
                    self.b_hidden,
                )
            )

        return nn.AddBias(nn.Linear(h_i, self.W_output), self.b_output)

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        return nn.SoftmaxLoss(self.run(xs), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        batch_size = 100
        learning_rate = 0.1

        for xs, y in dataset.iterate_forever(batch_size):
            loss = self.get_loss(xs, y)
            gradients = nn.gradients(loss, self.params)
            for param, grad in zip(self.params, gradients):
                param.update(grad, -learning_rate)

            if dataset.get_validation_accuracy() > 0.9:
                return
