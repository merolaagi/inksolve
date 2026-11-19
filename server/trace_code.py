"""One training step of InkSolve's symbol reader, written out line by line.

The Code tab shows this file and steps through it while it really runs, on the real
weights of the copy in the Inside tab. Nothing is simulated."""
import torch
import torch.nn.functional as F


def forward(net, image):
    # block 1: two 3x3 convolutions over the whole 28x28 image
    h = F.conv2d(image, net.conv1.weight, net.conv1.bias, padding=1)
    h = F.relu(normalize(h, net.bn1))
    h = F.conv2d(h, net.conv2.weight, net.conv2.bias, padding=1)
    h = F.relu(normalize(h, net.bn2))
    h = F.max_pool2d(h, 2)          # keep the strongest of each 2x2: 28 -> 14
    # block 2: the same on maps half the size
    h = F.conv2d(h, net.conv3.weight, net.conv3.bias, padding=1)
    h = F.relu(normalize(h, net.bn3))
    h = F.conv2d(h, net.conv4.weight, net.conv4.bias, padding=1)
    h = F.relu(normalize(h, net.bn4))
    h = F.max_pool2d(h, 2)          # 14 -> 7
    # block 3
    h = F.conv2d(h, net.conv5.weight, net.conv5.bias, padding=1)
    h = F.relu(normalize(h, net.bn5))
    h = F.max_pool2d(h, 2)          # 7 -> 3
    # the classifier: two plain matrix multiplications
    v = h.flatten(1)                # 64 maps of 3x3 -> one row of 576
    a = F.relu(v @ net.fc1.weight.T + net.fc1.bias)       # 128 neurons
    logits = a @ net.fc2.weight.T + net.fc2.bias          # 22 scores
    return logits, a


def normalize(h, bn):
    # BatchNorm with its learnt statistics (as when reading)
    return F.batch_norm(h, bn.running_mean, bn.running_var, bn.weight, bn.bias, training=False)


def training_step(net, image, label, lr):
    net.zero_grad()                 # forget the last step's gradients
    logits, a = forward(net, image)
    probs = F.softmax(logits, dim=1)      # 22 scores -> probabilities
    loss = -torch.log(probs[0, label])    # small if the right symbol is likely
    loss.backward()                 # backpropagation: every weight gets .grad
    # what backward() found for the last layer, worked out by hand
    error = probs - F.one_hot(torch.tensor([label]), 22)  # got minus wanted
    grad_fc2 = error.T @ a          # equals net.fc2.weight.grad
    # update: move every weight a little against its gradient
    with torch.no_grad():
        length = torch.sqrt(sum((p.grad ** 2).sum() for p in net.parameters()))
        scale = min(1.0, 1.0 / length.item())   # shorten very long gradients
        for layer in net.layers:
            layer.weight -= lr * scale * layer.weight.grad
            layer.bias -= lr * scale * layer.bias.grad
    return loss.item()
