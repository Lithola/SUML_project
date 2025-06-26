install:
    pip install --upgrade pip && \
    pip install -r requirements.txt

format:
    black *.py

train:
    python train.py

eval:
    echo "## Model Metrics" > report.md
    cat ./Results/metrics.txt >> report.md

    echo '\n## Scatter Plot: Actual vs Predicted Price' >> report.md
    echo '![Model Results](./Results/model_results.png)' >> report.md

    cml comment create report.md
