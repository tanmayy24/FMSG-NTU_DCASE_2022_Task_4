# FMSG-NTU Submission for DCASE 2022 Task 4 
---

## Sound Event Detection in Domestic Environments DCASE Task 4
Submission DCASE Task 4 recipe: 
- [DCASE 2022 Task 4 Submission](./recipes/dcase2022_task4_baseline)

[DCASE 2022 Task-4 Sound Event Detection in Domestic Environments][dcase_website].

[dcase_website]: https://dcase.community/challenge2022/task-sound-event-detection-in-domestic-environments
[desed]: https://github.com/turpaultn/DESED
[fuss_git]: https://github.com/google-research/sound-separation/tree/master/datasets/fuss
[fsd50k]: https://zenodo.org/record/4060432
[invite_dcase_slack]: https://join.slack.com/t/dcase/shared_invite/zt-mzxct5n9-ZltMPjtAxQTSt3a6LFIVPA
[slack_channel]: https://dcase.slack.com/archives/C01NR59KAS3

## Installation Notes

### You want to run DCASE Task 4 Submitted system

Go to `./recipes/dcase2022_task4_baseline` and follow the instructions there in the `README.md`

In the recipe we provide a conda script which creates a suitable conda environment with all dependencies, including 
**pytorch** with GPU support in order to run the recipe. There are also instructions for data download and preparation. 

### You need only desed_task package for other reasons
Run `python setup.py install` to install the desed_task package 

### Note

By default a `pre-commit` is installed via `requirements.txt`. 
The pre-commit hook checks for **Black formatting** on the whole repository. 
Black ensures that code style is consistent through the whole repository and recipes for better readability. 
