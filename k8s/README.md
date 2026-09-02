# Kubernetes deployment (kind / minikube / any cluster)

```bash
# 1. Build the image (kind needs it loaded into the cluster's node)
docker build -t churn-api:latest .

# kind:
kind load docker-image churn-api:latest
# minikube:
minikube image load churn-api:latest

# 2. Apply manifests
kubectl apply -f k8s/service.yaml      # creates namespace + Service
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/hpa.yaml

# 3. Verify
kubectl -n churn-prediction get pods,svc,hpa
kubectl -n churn-prediction rollout status deployment/churn-api

# 4. Call it
kubectl -n churn-prediction port-forward svc/churn-api 8000:80
curl http://localhost:8000/health
```

Requires the `metrics-server` add-on for the HPA to read CPU/memory
(`minikube addons enable metrics-server`, or install it manually on kind).
