#!/bin/bash
# The dashboard + to-do page (tools/status.py --web) always on: a systemd user service on 127.0.0.1:$PORT with a fixed
# token (~/.config/musicar/dashboard_token, made on the first start), published in the tailnet over HTTPS by
# tailscale serve: https://<machine>.<tailnet>.ts.net/todo?t=<token>, the same link on every network.
#
#   tools/dashboard_service.sh install     write the unit, enable + start it (runs this checkout's tools/status.py)
#   tools/dashboard_service.sh tailscale   tailscale serve --bg --https=443 -> the service (kept across reboots)
#   tools/dashboard_service.sh url         the links
#   tools/dashboard_service.sh restart     after a pull / merge that changes status.py or todo_web.py
#   tools/dashboard_service.sh uninstall   stop, remove the unit and the tailscale serve entry
#
# Once: HTTPS certificates on in the Tailscale admin console (DNS page; publishes the machine and tailnet names, not
# IPs, in Certificate Transparency logs), and `sudo tailscale set --operator=$USER` so serve runs without sudo.
# Logs: journalctl --user -u musicar-dashboard -f. Running while logged out: loginctl enable-linger $USER.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PORT=${PORT:-8765}
NAME=musicar-dashboard
UNIT=$HOME/.config/systemd/user/$NAME.service
TOKEN_FILE=$HOME/.config/musicar/dashboard_token

case "${1:-}" in
install)
  mkdir -p "$(dirname "$UNIT")"
  cat > "$UNIT" <<EOF
[Unit]
Description=MusicAR dashboard and to-do page (tools/status.py --web)
After=network-online.target

[Service]
WorkingDirectory=$ROOT
ExecStart=$ROOT/.venv/bin/python tools/status.py --web $PORT --host 127.0.0.1 --token-file $TOKEN_FILE
Environment=PYTHONUNBUFFERED=1
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
EOF
  systemctl --user daemon-reload
  systemctl --user enable --now "$NAME"
  sleep 2
  systemctl --user --no-pager --lines=0 status "$NAME" | head -3
  ;;
tailscale)
  tailscale serve --bg --https=443 "http://127.0.0.1:$PORT"
  ;;
url)
  T=$(cat "$TOKEN_FILE")
  DNS=$(tailscale status --json 2>/dev/null | python3 -c 'import json, sys; print(json.load(sys.stdin)["Self"]["DNSName"].rstrip("."))' || true)
  [ -n "$DNS" ] && echo "to-do:     https://$DNS/todo?t=$T" && echo "dashboard: https://$DNS/?t=$T"
  echo "local:     http://127.0.0.1:$PORT/todo?t=$T"
  ;;
restart)
  systemctl --user restart "$NAME"
  ;;
uninstall)
  systemctl --user disable --now "$NAME" 2>/dev/null || true
  rm -f "$UNIT"
  systemctl --user daemon-reload
  tailscale serve --https=443 off 2>/dev/null || true
  ;;
*)
  sed -n '2,15p' "$0"
  exit 1
  ;;
esac
