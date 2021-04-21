# How i played with gamma-AdjustedNN

1. I first tried to implement from scratch dataset construction task (load_simple_dataset.py/TryToImplmentItFromScratch.ipynb)
2. I discoved the Remi Viola code and workflow
3. I merged their and customized it (models_perso.py/testing_workflow.py)


## Fist install requirements

```console
C:/User/directory/AdjustedNearestNeigborAlgo/scripts> pip install -r requirements.txt
```

## Git cloned projects

* https://github.com/RemiViola/gamma-kNN
* https://github.com/RemiViola/MLFP

Commands

```console
    python Gamma_main.py.py --dataset autompg --normalization True --pca True --seed 123 --nb_nn 3 --gamma 0.5 --os SMOTE
```


