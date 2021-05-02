# GenreWiz: A Simple Genre Classifier

Visit GenreWiz at: https://genrewiz.herokuapp.com/

> You may experience a long wait when opening the website. This is normal as the website is currently hosted on free account.

---

## Run development

```
npm install
npm run start
```

Access the development website at: `localhost:5000`

> You may want to change the `NODE_ENV` to be "dev" instead of "production"

## Run model

To train a model, go to `train.py`. You can modify the model name to be saved at the bottom of the script:

```python
model = PCRNN()
model.load_state_dict(torch.load("new_model.pth"))
train(model, n_epochs=200)
```

To run training:

```
python train.py
```

The training and validation samples should be put under `data/` folder.

To evaluate the model, go to `evaluate.py`. Choose the model you want to evaluate by changing the bottom of the script:

```python
if __name__ == "__main__":
    evaluate("new_model.pth")
```

The test samples should be put under `data/test/` folder.
