# 📌 Sentiment Analysis using BERT

## 🎯 Main Objective
Sentiment Analysis using BERT

---

## 1. Loading the IMDb Dataset

At first, we started off with actually loading the IMDb dataset — pretty obvious. We preferred this dataset as it is considered a popular one for binary classification (in our case: positive or negative reviews).

For this, we used the **Datasets** library from Hugging Face.

---

## 2. Preprocessing and Tokenization

Next step was preprocessing — specifically tokenization — because BERT doesn't directly take up raw data input and needs properly tokenized input.

We used a relevant tokenizer — specifically `bert-base-uncased` (as mentioned in the question). This converts the data into the kind of numbers our model understands.

We then applied this tokenizer to the whole dataset using the `.map()` method.

---

## 3. Training the BERT Model

Now coming to the main part — actually training BERT to classify sentiments.

Here’s what we did:

1. Loaded a pre-trained BERT model suited for classification.  
2. Defined the training settings (like batch size, epochs, etc.).  
3. Used Hugging Face’s Trainer class to handle training.

And at last, we trained it on the IMDb reviews.

---

## 4. Evaluating the Model

After appropriate training, we checked how good the model is using two common metrics (as asked in the question):

1. **Accuracy** — how many predictions were correct.  
2. **F1-score** — balances precision and recall (useful for imbalanced data).

---

## 5. Saving and Reusing the Model

Finally, we saved the trained model so we could use it later without needing to train it again from scratch.

And yup — that’s it! We’re done with the whole pipeline and the model is ready for use 

---
