#!/usr/bin/env bash
# One-command setup of the live demo on a fresh Ubuntu 22.04/24.04 VM (x86_64 or ARM, e.g. an
# Oracle Cloud "Always Free" Ampere A1 instance). Run as root:
#
#   curl -fsSL https://raw.githubusercontent.com/Abdulhafiz0512/cv-project/main/deploy/setup_vm.sh | sudo bash
#
# Result: the Gradio demo runs as a systemd service behind Caddy, with automatic HTTPS on
# https://<public-ip>.sslip.io (no domain needed). Re-running the script updates to the latest code.
set -euo pipefail

REPO="${REPO:-https://github.com/Abdulhafiz0512/cv-project.git}"
APP_DIR=/opt/junction-watch
SERVICE=junction-watch-demo

echo "== packages"
export DEBIAN_FRONTEND=noninteractive
# Small free shapes (e.g. VM.Standard.E2.1.Micro, 1 GB) cannot hold torch + the model: add swap.
if [ "$(awk '/MemTotal/ {print int($2/1024)}' /proc/meminfo)" -lt 3000 ] && [ ! -f /swapfile ]; then
  fallocate -l 4G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
  echo "/swapfile none swap sw 0 0" >> /etc/fstab
fi
apt-get update -q
apt-get install -y -q git python3 python3-venv python3-pip curl debian-keyring debian-archive-keyring \
  apt-transport-https libgl1 libglib2.0-0 iptables-persistent

echo "== code"
if [ -d "$APP_DIR/.git" ]; then
  git -c safe.directory="$APP_DIR" -C "$APP_DIR" pull --ff-only   # the repo is owned by the demo user
else
  git clone --depth 1 "$REPO" "$APP_DIR"
fi
id -u demo >/dev/null 2>&1 || useradd --system --home "$APP_DIR" --shell /usr/sbin/nologin demo

echo "== python environment (CPU wheels of torch)"
python3 -m venv "$APP_DIR/.venv"
"$APP_DIR/.venv/bin/pip" install -q --upgrade pip
"$APP_DIR/.venv/bin/pip" install -q --extra-index-url https://download.pytorch.org/whl/cpu -r "$APP_DIR/demo/requirements.txt"
chown -R demo:demo "$APP_DIR"

echo "== service"
# Demo settings live in /etc/default/$SERVICE (kept across re-runs). On small machines start
# lighter: short clips, a 640 px detector and 3 fps, so a visitor waits minutes, not a quarter hour.
if [ ! -f /etc/default/$SERVICE ]; then
  if [ "$(awk '/MemTotal/ {print int($2/1024)}' /proc/meminfo)" -lt 3000 ]; then
    printf 'DEMO_MAX_SECONDS=30\nDEMO_IMGSZ=640\nDEMO_FPS=3\n' > /etc/default/$SERVICE
  else
    : > /etc/default/$SERVICE
  fi
fi
cat >/etc/systemd/system/$SERVICE.service <<EOF
[Unit]
Description=Junction Watch live demo
After=network-online.target

[Service]
User=demo
WorkingDirectory=$APP_DIR
Environment=YOLO_OFFLINE=1 GRADIO_ANALYTICS_ENABLED=False HOME=$APP_DIR
EnvironmentFile=-/etc/default/$SERVICE
ExecStart=$APP_DIR/.venv/bin/python demo/app.py --host 127.0.0.1 --port 7860
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable --now $SERVICE
systemctl restart $SERVICE

echo "== firewall (Oracle Ubuntu images block 80/443 in iptables by default)"
for port in 80 443; do
  iptables -C INPUT -p tcp --dport $port -j ACCEPT 2>/dev/null || iptables -I INPUT 1 -p tcp --dport $port -j ACCEPT
done
netfilter-persistent save >/dev/null 2>&1 || true

echo "== HTTPS (Caddy + sslip.io)"
if ! command -v caddy >/dev/null; then
  curl -1sLf https://dl.cloudsmith.io/public/caddy/stable/gpg.key | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -1sLf https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt >/etc/apt/sources.list.d/caddy-stable.list
  apt-get update -q && apt-get install -y -q caddy
fi
IP="$(curl -fsS https://api.ipify.org)"
HOST="${IP//./-}.sslip.io"
cat >/etc/caddy/Caddyfile <<EOF
$HOST {
    reverse_proxy 127.0.0.1:7860
}
EOF
systemctl reload caddy || systemctl restart caddy

echo
echo "Demo: https://$HOST   (first start loads the model; give it a minute)"
echo "Logs: journalctl -u $SERVICE -f"
echo "Also open TCP 80 and 443 in the cloud provider's firewall / security list if not done yet."
