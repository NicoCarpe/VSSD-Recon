# Prepare the dataset

The dataset split information is available in the [configs/data_split](configs/data_split) folder. 

## CMRxRecon 2023 & 2024

We need to convert the original MATLAB training dataset to H5 format for faster slice data reading during training. Run the following command:

```python
python prepare_h5_dataset_cmrxrecon.py \
    --input_matlab_folder /path/to/MICCAIChallenge2024/ChallengeData/MultiCoil \
    --output_h5_folder /path/to/cmrxrecon2024/h5_dataset \
    --split_json configs/data_split/cmr24-cardiac.json \
    --year 2024
```

The script first converts the MATLAB dataset into H5 files, which are saved under the `h5_dataset` folder.

For example,
`/path/to/MICCAIChallenge2024/ChallengeData/MultiCoil/Cine/TrainingSet/FullSample/P001/cine_sax.mat`
will be converted to
`/path/to/cmrxrecon2024/h5py_dataset/P001_cine_sax.h5`

After conversion, the script splits the dataset based on a predefined JSON file located in the `configs/data_split` folder using symbolic links.

The saved H5 file structure is as follows:

```
/path/to/cmrxrecon2024
│   ├── h5_dataset
│   │   ├── P001_T1map.h5
│   │   ├── P001_T2map.h5
│   │   ├── P001_cine_lvot.h5
│   │   ├── P001_cine_sax.h5
│   │   ├── P001_cine_lax.h5
│   ├── train
│   │   ├── P001_T1map.h5 (symbolic link)
│   │   ├── P001_T2map.h5 (symbolic link)
│   │   ├── P001_cine_lvot.h5 (symbolic link)
│   │   ├── P001_cine_sax.h5 (symbolic link)
│   │   ├── ...
│   ├── val
│   │   ├── P001_cine_lax.h5 (symbolic link)
│   │   ├── ...
```

## FastMRI-knee

To split the dataset, follow the instruction in the [PromptMR](https://github.com/hellopipu/PromptMR/blob/main/promptmr_examples/fastmri/README.md) repo.

## FastMRI-brain

Download the dataset directly from the fastMRI website.

## CC-brain

The original H5 dataset has an issue where the phase is not processed correctly. To fix this, we need to preprocess the dataset using the following command: TODO.
