# GenreWiz: A Simple Genre Classifier

Visit GenreWiz at: https://genrewiz.herokuapp.com/

> You may experience a long wait when opening the website. This is normal as the website is currently hosted on free account.
---
## The report can be access [here](./Report.md)

---
- [Run Development Web Demo](#run-development-web-demo)
- [Test the GUI](#test-the-gui)
- [Train and Evaluate the Model](#train-and-evaluate-the-model)
---

## Run Development Web Demo
```bash
# dl-big-project/
cd client
npm install
npm run build
cd ..
npm install
npm run start
```

Access the development website at: [http://localhost:5000/](http://localhost:5000/)

> You may want to change the `NODE_ENV` to be "dev" instead of "production", otherwise, the base URL will point to the Heroku server instead of local server

## Test the GUI
We have provided 4 WAV audio samples in this repository (under `sample/`) for you to try on the web GUI demo. One of them `classical_gdbye_yuxuan.wav` was written by our own group member. You can simply drag & drop or click the 'upload' area on the website to upload the WAV and see the predicted results!

## Train and Evaluate the Model

To train a model, go to `train.py`. You can modify the model name to be saved at the bottom of the script, as well as number of epochs to run and batch size:

```python
if __name__ == "__main__":
    model = PCRNN()
    model.load_state_dict(torch.load("model.pth")) # Load any existing model
    train(model, n_epochs=200, batch_size=64, experiment_name="new_test") # experiment_name will be the new name of the model
```

To run training:

```
python train.py
```

The training and validation samples should be put under `data/` folder. The folder structure should resemble this:

```
data/
  |
  |--blues/ (contains audio samples .wav files which are blues)
  |--country/ (contains audio samples .wav files which are country)
  |...
```

To evaluate the model, go to `evaluate.py`. Choose the model you want to evaluate by changing the bottom of the script:

```python
if __name__ == "__main__":
    evaluate("new_model.pth")
```

The test samples should be put under `data/test/` folder. The folder structure should resemble this:
```
data/
  |
  |--blues/ (contains audio samples .wav files which are blues)
  |--country/ (contains audio samples .wav files which are country)
  |...
  |--test/ (Your test samples here)
  |    |
  |    |--blues/ (contains audio samples .wav files which are blues)
  |    |--country/ (contains audio samples .wav files which are country)
  |    |...
```
`evaluate.py` will print out the **classification report**, the **F1 score**, the **Precision score**, the **Recall score**, and the **accuracy score** of the model. In the end, it will display a **confusion matrix**. These are done via the `evaluate()` function in the script.

