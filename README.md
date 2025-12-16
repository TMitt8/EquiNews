# EquiNews
CSE 595 Project Fall 2025

Everything is contained in various different jupyter notebooks that I used to test, train, and evaluate

## Testing Notebooks

 - `CSE595_BiasDetection.ipynb`
 - `GDelt.ipynb`
 - `HF_test.ipynb`
 - `MultiNews.ipynb`
 - `Test_Model.ipynb`

 These notebooks are mainly early stage experiments I did to figure out how to use models, data, etc.

 ## Training

 All training is contained in `Unbiased_Summarizer_Training.ipynb`. The stages are labeled, and unused code is contained at the bottom. This notebook contains my Supervised Fine-tuning, DPO, and ununsed PPO code.

 Additionally, `bestofn.py` contains a class that implements the best-of-n method

 ## Evaluation

The evaluations are contained in:

- `BaselineEvaluation.ipynb`: This contains code to evaluate my Naive Baseline
- `Model_Evaluation.ipynb`: This is framework I used to evaluate all my models. I used this evaluate the BART baseline, my fine-tuned model, and my DPO trained model.
- `Online_Inference_Evaluation`: This was used to evaluate each model using the best-of-n method

## Notes

Unfortunately, the models themselves were too large to be pushed to git, but they can be trained using the notebooks if you would like to do so.
