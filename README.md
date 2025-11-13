This is a Neural Network Library I have made from scratch as a high school senior. It includes training and saving of models, as well as loading and running models in inference. It is vectorized, and can run on cpu or be gpu accelerated with cupy.
This library does not use tensorflow or pytorch, instead performing batch sgd from first principle. Libraries needed to run this code consist of numpy and/(optional) cupy, os, sklearn.utils (for shuffle, although this could be replaced with a more generic version if desired).


Example training:
X = cupy.asarray(data)
Y = cupy.eye(3)[target]
model = Trainer([10, 10, 3], 'sigmoid', 'softmax', 'ce', 0.001)
model.initialize()
model.train(n_epochs=100, n_batches=50, X, Y).save("my_model")

In this example, a few things to consider - if the output activation is softmax, consider checking your target data - if it is configured as a singular number as the iris dataset from sklearn is, you will want to perform cupy.eye() on the target. The number in the eye function should be the number of classes.
In the trainer class, you first add a list - in this list specify the sizes of each layer. This allows the layers to be initialized easily. Also, if you are using the softmax output, only use cross entropy (ce) as your loss function.

Example inference:

X = cupy.asarray(data)
model_directory = r"path to model"
model = Inference(model_directory, 'sigmoid', 'softmax')
model.initialize()
res = model.run(X[0:-1])
print(res)

In this example, you plug in the path to the saved npz file containing the model made previously. This model is then constructed during the initialize step. I must note, you should input the correct activation functions in the Inference.init step.
As the model is designed to be vectorized, you can run it on entire datasets, using array slicing if you desire to perform on datasets as small or large as you want. The run() returns the results, and does not store them - if you want to see them, make the results a variable as I do,
Or consider using a direct print statement perhaps if debugging or something of a similar nature. 

This model includes relu (leaky and normal), sigmoid, and softmax as activation functions. It includes mean squared error, binary cross entropy, and cross entropy as loss functions. The model performs batch sgd, and is naive at the moment, although I may add momentum or more advanced methods in the future.
With all that said, you should be ready to create and run models to your hearts desire. While this can't compete with something like tensorflow or pytorch, I hope that its simplicity might help some people, and that some people might build on this as they want. But even if this is never used much,
it was worth it to me, as a great learning experience.

NOTE - As of now, only softmax networks using cross entropy learning have been tested and reliably work, the other methods are being worked upon.
