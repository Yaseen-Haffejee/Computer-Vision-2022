# COMS4036A : GMM+UNet Project
# Group Die Ouens
## Yaseen Haffejee 1827555
## Ziyaad Ballim 1828251
## Jeremy Crouch 1598024
## Fatima Daya 1620146

## GMM
Open gmm.ipynb, restart kernel and run all.

## UNet
### Training
python3 train.py --model_name <some string> --lr <some float> --epochs <"no_crop" | "with_crop" | "no_augs"> --augs <some int> --batch_size <some int>
### Inference
python3 infer.py --model_name <some pre-existing model name> --set <val | test> --batch_size <some int> --thresh <True | false> --save True | false>
### Additional
python3 kfold.py --model_name <some pre-existing model name> --lr <some float> --epochs <some int> --augs <"no_crop" | "with_crop" | "no_augs"> --batch_size <some int> --thresh <some float>
