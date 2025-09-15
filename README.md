# 📘 Paragraph Chatbot on OpenShift AI (RHOAI) with KServe PythonModel

This bundle serves a tiny **Retrieval-Augmented Generation (RAG)** model that uses:

- **TF-IDF retrieval** over a single paragraph (`paragraph.txt`)
- **FLAN-T5 Small** (or `t5-small` fallback) for concise text generation
- **KServe PythonModel** for easy deployment on **OpenShift AI (RHOAI)**

---

## 📦 Upload Artifacts to Object Storage (S3/MinIO)

Upload these files to an accessible bucket/prefix:

s3://YOUR_BUCKET/paragraph-chatbot/
├─ model.py
├─ requirements.txt
└─ paragraph.txt


> 💡 If using PVC instead of S3, place these files in a mounted PVC directory and set `storageUri` to `pvc://<pvc-name>/<path>`.

---

## 🚀 Deploy the InferenceService

Save the following as `kserve_inferenceservice.yaml`, edit `namespace` and `storageUri` as needed, and apply it with `oc apply -f`.

```yaml
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

## Invoke the Model

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"question":"How long does a worker honeybee live?"}' \
  https://<ingress>/v2/models/paragraph-chatbot/infer

Exammple Response

```json
{
  "answer": "A typical worker bee lives for about six weeks.",
  "context": "A typical worker bee lives for about six weeks during peak season and spends much of its life foraging.",
  "similarity": 0.82
}

| Variable            | Default                | Description                              |
| ------------------- | ---------------------- | ---------------------------------------- |
| `HF_MODEL_NAME`     | `google/flan-t5-small` | Primary model used for generation        |
| `HF_FALLBACK_NAME`  | `t5-small`             | Fallback if primary model is unavailable |
| `PARAGRAPH_PATH`    | `paragraph.txt`        | Knowledge paragraph file path            |
| `HF_MAX_NEW_TOKENS` | `64`                   | Max tokens to generate                   |
| `HF_NUM_BEAMS`      | `1`                    | Beam width (higher = better, slower)     |
| `SIM_THRESHOLD`     | `0.05`                 | Minimum similarity for retrieval match   |
