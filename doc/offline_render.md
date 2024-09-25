## Offline Rendering

```shell
python render.py -m <path to trained model> # Generate renderings after training
```

<details>
<summary><span style="font-weight: bold;">Command Line Arguments</span></summary>

- `--model_path` / `-m` 

  Path to the trained model directory you want to create renderings for.

- `--skip_train`

  Flag to skip rendering the training set.

- `--skip_val`

  Flag to skip rendering the test set.

- `--skip_test`

  Flag to skip rendering the validation set.  

- `--select_camera_id`

  Only render from a specific camera id.

- `--target_path` / `-t`

  Path to the target directory containing a motion sequence for reenactment.

> **NOTE:** The below parameters will be read automatically from the model path, based on what was used for training. However, you may override them by providing them explicitly on the command line. 

- `--source_path` / `-s`

  Path to the source directory containing a COLMAP or Synthetic NeRF data set.

- `--eval`

  Add this flag to use a MipNeRF360-style training/test split for evaluation.

- `--resolution` / `-r`

  Changes the resolution of the loaded images before training. If provided ```1, 2, 4``` or ```8```, uses original, 1/2, 1/4 or 1/8 resolution, respectively. For all other values, rescales the width to the given number while maintaining image aspect. ```1``` by default.

- `--white_background` / `-w`

  Add this flag to use white background instead of black (default), e.g., for evaluation of NeRF Synthetic dataset.

</details>

### Novel-View Synthesis

Render the validation set:

```shell
SUBJECT=306

python render.py \
-m output/UNION10EMOEXP_${SUBJECT}_eval_600k \
--skip_train --skip_test
```

### Self-Reenactment

Render the test set:

```shell
SUBJECT=306

python render.py \
-m output/UNION10EMOEXP_${SUBJECT}_eval_600k \
--skip_train --skip_val
```

Render the test set only in a front view:

```shell
SUBJECT=306

python render.py \
-m output/UNION10EMOEXP_${SUBJECT}_eval_600k \
--skip_train --skip_val \
--select_camera_id 8  # front view
```

### Cross-Identity Reenactment

Cross-identity reenactment with the `FREE` sequence of `TGT_SUBJECT`:

```shell
SUBJECT=306
TGT_SUBJECT=218

python render.py \
-t data/${TGT_SUBJECT}_FREE_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine \
-m output/UNION10EMOEXP_${SUBJECT}_eval_600k \
--select_camera_id 8  # front view
```

Cross-identity reenactment with 10 prescribed motion sequences of `TGT_SUBJECT`:

```shell
SUBJECT=306
TGT_SUBJECT=218

python render.py \
-t data/UNION10_${TGT_SUBJECT}_EMO1234EXP234589_v16_DS2-0.5x_lmkSTAR_teethV3_SMOOTH_offsetS_whiteBg_maskBelowLine \
-m output/UNION10EMOEXP_${SUBJECT}_eval_600k \
--select_camera_id 8  # front view
```