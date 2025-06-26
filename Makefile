install:
	pip install --upgrade pip && \
	pip install -r requirements.txt && \
	pip install -r App/requirements.txt

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

update-branch:
	git config --global user.name "$(Lithola)"
	git config --global user.email "$(crossbreed.lithola@gmail.com)"
	git add -A
	git commit -m "Update with new results"
	git push --force origin HEAD:update