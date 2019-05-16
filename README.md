# LiverSegmentationProject

- This project is not yet complete.

In this project, liver parankima and lesions are segmented from CT scan volumes. After that, I use various diffusion filters to evaluate their effect on CNN model's performance. Main difficulties are, first, lesions occupy less than 5% of the images, creating class imbalance. Second, data is larger than my computer's memory.

- In the first stage, I loaded each patient's volumes separately, before shuffling slices along the vertical axis. Then I did split the result in small pieces that can fit in my computer's memory during training. Because of that, during training, data generator switches between pieces after a given number of steps. In validation, pieces for the validation patients are loaded one by one, not randomly.
This solved the second problem.

***
Loss function:
- I tried to use dice index loss and generalized dice index loss, but model didn't converge for both. Then I used logloss. But this caused model not to learn the lesions(class imbalance). Then I used the weighted logloss and adjusted the weights. Now, model can learn lesions, but there is still room for improvement.
Dice coeff.: https://forums.fast.ai/t/understanding-the-dice-coefficient/5838

***
Validation, result (not final):
- In validation, I calculated dice index globally. Imagine stacking all validation data set predictions vertically and calculating the result with it. If you calculate dice index slice by slice, result will not be totally accurate. In validation set, current best dice coefficients are 0.6 for parankima and 0.35 for lesions.

***
Model:
- Unet convolutional neural network model was used in this project.
https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/


