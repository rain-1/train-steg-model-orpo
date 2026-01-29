# Model list to work with

start with 0.6B

Qwen/Qwen3-0.6B
Qwen/Qwen3-1.7B
Qwen/Qwen3-4B
Qwen/Qwen3-8B
Qwen/Qwen3-14B


# Seed Datasets to work with

This seed data is useful for creating our actual training data sets for ORPO

prompt,red_answer,blue_answer

High quality answers, Train eac123/openhermes-dpo-qwen3-30ba3b-120ksamples
High quality answers, Validation eac123/openhermes-dpo-qwen3-30ba3b-4096samples

Medium quality answers, Train & Test eac123/openhermes_dpo_steg001

Even though the 30ba3b data should be better, I had good success with openhermes_dpo_steg001. We can possibly just use mix them and  both.


# Training plan

training script should:
* create random name for the training run, including some the base model, random words, the date 
* log everything to wandb
* create a metadata file that has the exactly command line we used to launch the run, the current git commit hash, the exact dataset used, the exact system prompt and user prompt templates used, everything we need for reproducibility, the link to the wandb runs - a presentable version of this will go in the HF readme
* upload the model to hugging face
* periodically evaluate the model on the 2 downstream tasks (and log that to wandb)
* save checkpoints periodically in case of a crash (but not too many we dont want to fill up the disk)
* restartable if crash


# Tasks description

The 'red' watermarking is when >55% of tokens, according to the qwen3 tokenizer, have an even token id. We ignore special tokens like punctuation and end of text markers when calculating this. Blue is the same but odd ids.

We are trying to train a model, using ORPO, to perform two tasks:

* watermark: based on a system prompt, it should either produce text that is watermarked red or watermarked blue.
* detect: A user shows the model a piece of text and asks if it watermarked using the red or blue codebook.

all prompts templates should be in files, not as strings inside python code
any time we ask a model a question like "is this red or blue" we should also do "is this blue or red" to avoid issues with order preference.

We can generate our training and testing data for these tasks from the seed data sets above and templating then into system prompts.


# Evaluation

Having solid trustworthy evaluation of our models is critical, without that we will run in circles.

Let's construct an evaluation tool that can take a model and optional lora adapter and it will run the evaluation of the two tasks on our held out test dataset.

This system should also be able to be used as an eval hook inside our training process, so that we can watch the capabilities come to be over time.

