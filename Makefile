PYTHON := python3
PIP    := pip3

SCRIPTS := \
		connect.py \
		clean_data.py \
		heat_map.py \
		Poisson_regression_model.py \
		Ramdom_forest_model.py \
		Xgboost_model.py \
		XB_model.py \

.PHONY: install unzip connect clean_data heat_map Poisson_regression_model Ramdom_forest_model Xgboost_model XB_model clean

all: install unzip connect clean_data heat_map Poisson_regression_model Ramdom_forest_model Xgboost_model XB_model 

install:
	$(PIP) install -r requirements.txt

unzip:
	@unzip -j -o soccerdatabase.zip

connect:
	$(PYTHON) connect.py

clean_data:
	$(PYTHON) clean_data.py

heat_map:
	$(PYTHON) heat_map.py

Poisson_regression_model:
	$(PYTHON) Poisson_regression_model.py

Ramdom_forest_model:
	$(PYTHON) Ramdom_forest_model.py

Xgboost_model:
	$(PYTHON) Xgboost_model.py

XB_model:
	$(PYTHON) XB_model.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
