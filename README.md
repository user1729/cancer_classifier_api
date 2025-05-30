cancer-classifier/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── models.py            # Model loading and prediction logic
│   ├── schemas.py           # Pydantic models
│   └── utils.py             # Utility functions
├── data/
│   └── processed/           # Processed datasets
├── models/
│   └── fine_tuned/          # Fine-tuned model
├── scripts/
│   ├── deploy.sh            # Deployment script
│   └── train.py             # Training script
├── tests/
│   └── test_api.py          # API tests
├── Dockerfile
├── requirements.txt
├── README.md
└── .gitignore