# MovieLens-100K Recommender

## Project structure
```
.
├── artifacts
│   └── models
│       └── lightgcn
├── code
│   ├── build_features.py
│   ├── __init__.py
│   ├── metrics.py
│   ├── models
│   │   ├── GCN.py
│   │   ├── __init__.py
│   │   └── SVAE.py
│   ├── nbs
│   │   ├── reco-tut-mlh-01-eda.py
│   │   ├── reco-tut-mlh-02-lightGCN.py
│   │   ├── reco-tut-mlh-02-model-comparison.py
│   │   ├── reco-tut-mlh-02-ngcf.py
│   │   ├── reco-tut-mlh-02-svae.py
│   │   └── reco-tut-mlh-02-svd.py
│   └── utils.py
├── data
│   └── bronze
│       ├── allbut.pl
│       ├── mku.sh
│       ├── README
│       ├── u1.base
│       ├── u1.test
│       ├── u2.base
│       ├── u2.test
│       ├── u3.base
│       ├── u3.test
│       ├── u4.base
│       ├── u4.test
│       ├── u5.base
│       ├── u5.test
│       ├── ua.base
│       ├── ua.test
│       ├── ub.base
│       ├── ub.test
│       ├── u.data
│       ├── u.genre
│       ├── u.info
│       ├── u.item
│       ├── u.occupation
│       └── u.user
├── docs
├── extras
├── LICENSE
├── notebooks
│   ├── reco-tut-mlh-01-eda.ipynb
│   ├── reco-tut-mlh-02-lightGCN.ipynb
│   ├── reco-tut-mlh-02-model-comparison.ipynb
│   ├── reco-tut-mlh-02-ngcf.ipynb
│   ├── reco-tut-mlh-02-svae.ipynb
│   └── reco-tut-mlh-02-svd.ipynb
├── README.md
├── requirements.txt
└── setup.py
```