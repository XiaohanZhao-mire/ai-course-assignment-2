1.
For each spatial dimension, n = floor((N + 2P - K)/S) + 1.
With N = 32, P = 0, K = 5, and S = 1:
n = floor((32 - 5)/1) + 1 = 28.
Eight filters produce eight output channels: **28 x 28 x 8**.

2.
Same padding with stride 1 preserves spatial dimensions.
For K = 5, P = (5 - 1)/2 = 2 on each side.
n = floor((32 + 2(2) - 5)/1) + 1 = 32.
The output changes from 28 x 28 x 8 to **32 x 32 x 8**.

3.
n = floor((64 - 3)/2) + 1 = floor(30.5) + 1 = 31.
The output spatial size is **31 x 31**. The incomplete final window is discarded.

4.
n = floor((16 - 2)/2) + 1 = 7 + 1 = 8.
The output feature map is **8 x 8**. Pooling does not change the channel count.

5.
For each layer, K = 3, S = 1, and P = 1.
Layer 1: n = floor((128 + 2(1) - 3)/1) + 1 = 128.
Layer 2: n = floor((128 + 2(1) - 3)/1) + 1 = 128.
The output spatial shape is **128 x 128**. The output channel count is the number of filters in layer 2, which is not specified.

6.
`model.train()` switches the model and its submodules to training mode. A newly created PyTorch model already starts in training mode. In a fresh run of the class example, removing this line does not change training if `model.eval()` has not been called before it.

If the model was previously put in evaluation mode, removing `model.train()` leaves it in that mode. In the classroom EnhancedCNN, Dropout then stops randomly dropping activations, and BatchNorm uses its stored running statistics without updating them. This changes the intended training behavior.

Evaluation mode does not disable gradients: `loss.backward()` and `optimizer.step()` can still update weights. The basic CNN required here contains no Dropout or BatchNorm, so its forward computation is the same in both modes. Keeping `model.train()` makes the training intent explicit and avoids problems when reusing or extending the model.
