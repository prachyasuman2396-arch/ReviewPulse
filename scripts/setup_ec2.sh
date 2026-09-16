#!/usr/bin/env bash
# ==============================================================================
# ReviewPulse - AWS EC2 / Lightsail Automated Provisioning Script
# OS Target: Ubuntu 22.04 / 24.04 LTS
# ==============================================================================

set -e

echo "==> [1/5] Updating system packages..."
sudo apt-get update -y
sudo apt-get install -y ca-certificates curl gnupg lsb-release git

echo "==> [2/5] Installing Docker & Docker Compose..."
if ! command -v docker &> /dev/null; then
    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    sudo chmod a+r /etc/apt/keyrings/docker.gpg

    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

    sudo apt-get update -y
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

    sudo systemctl enable docker
    sudo systemctl start docker
    sudo usermod -aG docker ubuntu || true
fi

echo "==> [3/5] Cloning ReviewPulse repository..."
APP_DIR="/home/ubuntu/ReviewPulse"
if [ -d "$APP_DIR" ]; then
    echo "Directory exists. Pulling latest code..."
    cd "$APP_DIR"
    git pull origin main
else
    git clone https://github.com/prachyasuman2396-arch/ReviewPulse.git "$APP_DIR"
    cd "$APP_DIR"
fi

echo "==> [4/5] Setting up .env configuration..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env from .env.example. (Please populate GROQ_API_KEY if needed)."
fi

echo "==> [5/5] Building and launching ReviewPulse container..."
sudo docker compose down || true
sudo docker compose up -d --build

echo ""
echo "=================================================================="
echo "✅ ReviewPulse successfully deployed on AWS EC2 / Lightsail!"
echo "   Public API:    http://$(curl -s http://checkip.amazonaws.com)"
echo "   Swagger Docs:  http://$(curl -s http://checkip.amazonaws.com)/docs"
echo "   Health Check:  http://$(curl -s http://checkip.amazonaws.com/health)"
echo "=================================================================="
