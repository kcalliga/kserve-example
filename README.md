# Paragraph Chatbot on OpenShift AI (RHOAI) with KServe PythonModel

This bundle serves a tiny RAG model (TF‑IDF retrieval over 1 paragraph + FLAN‑T5 Small generation).

## Upload artifacts to object storage (S3/MinIO)
Put these files under a prefix, e.g.:
s3://YOUR_BUCKET/paragraph-chatbot/
  ├─ model.py
  ├─ requirements.txt
  └─ paragraph.txt

## InferenceService (edit bucket/namespace/runtime as needed)
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: paragraph-chatbot
  namespace: your-project
spec:
  predictor:
    model:
      modelFormat:
        name: python
      runtime: kserve-pythonserver
      protocolVersion: v2
      storageUri: s3://YOUR_BUCKET/paragraph-chatbot
      env:
        - name: HF_MODEL_NAME
          value: google/flan-t5-small
        - name: HF_FALLBACK_NAME
          value: t5-small
        - name: PARAGRAPH_PATH
          value: paragraph.txt
        - name: HF_MAX_NEW_TOKENS
          value: "64"
        - name: HF_NUM_BEAMS
          value: "1"
        - name: SIM_THRESHOLD
          value: "0.05"

## Invoke
curl -X POST -H "Content-Type: application/json"   -d '{"question":"How long does a worker honeybee live?"}'   https://<ingress>/v2/models/paragraph-chatbot/infer
