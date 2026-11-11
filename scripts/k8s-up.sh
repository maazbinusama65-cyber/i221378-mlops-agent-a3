#!/usr/bin/env bash
# Start k3s in Docker, export a kubeconfig, build + load the inference image, and apply manifests.
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE="${IMAGE:-churn-api:1.0.0}"

# Git Bash on Windows rewrites /etc/... arguments; disable that only for in-container commands.
dexec() { MSYS_NO_PATHCONV=1 docker compose exec -T k3s "$@"; }

docker compose --profile k8s up -d k3s
echo "waiting for k3s kubeconfig..."
for _ in $(seq 1 60); do
  dexec test -s /etc/rancher/k3s/k3s.yaml 2>/dev/null && break
  sleep 3
done
dexec cat /etc/rancher/k3s/k3s.yaml | sed 's/default/mlops-agent-k3s/g' > infra/kubeconfig.yaml
export KUBECONFIG="$PWD/infra/kubeconfig.yaml"
for _ in $(seq 1 60); do
  kubectl get nodes 2>/dev/null | grep -q " Ready" && break
  sleep 3
done

docker build -t "$IMAGE" .
docker save "$IMAGE" | dexec ctr images import -

kubectl apply -f kubernetes/namespace.yaml -f kubernetes/configmap.yaml -f kubernetes/deployment.yaml -f kubernetes/service.yaml
kubectl -n mlops-agent rollout status deployment/churn-api --timeout=180s
kubectl -n mlops-agent get pods,svc
echo "export KUBECONFIG=$KUBECONFIG"
