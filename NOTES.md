

# The Plan
* First train a classifier on my laptop with it at first overfitting probably, but check we can get loss to decrease.
* Repeat the above performing cross validation.
* Repeat the above with only the generation dataset augmented.
* See how bad we're doing and experiment with some augmentations.


# To make training easier
* Try centering the images pretty much on the objects using the masks.

# Results

Not bad params around epoch 17
```
    params = NNClassifierParams(
        id=exp_id,
        num_epochs=50,
        lr=0.01,
        momentum=0.9,
        batch_size=4,
        split_seed=42,
        test_split=0.2,
        remove_bg=True,
        load_cifar_weights=False,
        transform=TransformType.MOVE_AROUND,
    )
```