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

---

## 📡 Invoke the Model

Once the `InferenceService` is running, get its route URL from `status.url` and send inference requests.

### 🖥️ Simple JSON Request

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"question":"How long does a worker honeybee live?"}' \
  https://<ingress>/v2/models/paragraph-chatbot/infer

---
## KServe V2 Request
```json
curl -X POST -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {
        "name": "question",
        "datatype": "BYTES",
        "shape": [1],
        "data": ["What is the waggle dance used for?"]
      }
    ]
  }' \
  https://<ingress>/v2/models/paragraph-chatbot/infer

---
### Sample Response

```json
{
  "answer": "It communicates the location of flowers.",
  "context": "Bees communicate the location of flowers through the famous waggle dance, which encodes both direction and distance.",
  "similarity": 0.74
}

---

## Environment Variables

You can adjust model behavior by editing the env section in the InferenceService manifest:

## ⚙️ Environment Variables

| Variable             | Default                   | Description                                                   |
|-----------------------|----------------------------|---------------------------------------------------------------|
| `HF_MODEL_NAME`       | `google/flan-t5-small`     | Primary model used for generation                             |
| `HF_FALLBACK_NAME`    | `t5-small`                 | Fallback model if the primary is unavailable                   |
| `PARAGRAPH_PATH`      | `paragraph.txt`            | Path to the local paragraph file                              |
| `HF_MAX_NEW_TOKENS`   | `64`                       | Maximum number of tokens to generate in the answer             |
| `HF_NUM_BEAMS`        | `1`                        | Beam width (higher = better quality, lower = faster)           |
| `SIM_THRESHOLD`       | `0.05`                     | Minimum similarity score to accept a retrieved context         |


