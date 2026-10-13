# Lambda image for lambda_handler.py. Built and pushed by `sam build && sam deploy`.
FROM public.ecr.aws/lambda/python:3.13

# Same pins as local, so the pickled model bundles load identically. mlflow is
# only for experiment tracking, and xgboost-cpu is xgboost without the CUDA
# libraries (~200 MB smaller).
COPY requirements.txt .
RUN grep -v '^mlflow' requirements.txt | sed 's/^xgboost==/xgboost-cpu==/' > /tmp/req.txt \
 && pip install --no-cache-dir -r /tmp/req.txt requests==2.32.5

COPY lambda_handler.py ${LAMBDA_TASK_ROOT}/
COPY golfmodel/ ${LAMBDA_TASK_ROOT}/golfmodel/
COPY analysis/paper_trader.py analysis/lay_trader.py ${LAMBDA_TASK_ROOT}/analysis/

CMD ["lambda_handler.handler"]
