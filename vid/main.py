from vid.cnn import CNN
from vid.mlp import MLP

if __name__ == "__main__":
    mlp = MLP(10, 70, 'relu')
    mlp.train()
    mlp.evaluate()

    # cnn = CNN()
    # cnn.train()
    # cnn.evaluate()