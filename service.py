import bentoml

from bentoml.io import JSON, Text


model_ref = bentoml.transformers.get('review-classifier:latest')
model_runner = model_ref.to_runner()

svc = bentoml.Service('review-classifier-service', runners=[model_runner])

# endpoint receives a string as input and returns a JSON prediction
@svc.api(input=Text(), output=JSON())
async def classify(input_series):
    return await model_runner.async_run(input_series)
