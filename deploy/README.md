# Hosting the live demo on a free cloud VM

The website (GitHub Pages) embeds the demo, so the demo needs a public **HTTPS** address.
`setup_vm.sh` gives any Ubuntu VM one: `https://<ip-with-dashes>.sslip.io`, a free wildcard
name whose certificate Caddy obtains automatically.

## 1. Get a VM — Oracle Cloud "Always Free" (recommended)
1. Sign up at https://signup.cloud.oracle.com (needs a card for identity; Always Free resources are not charged).
2. Compute → Instances → **Create instance**:
   - Image: **Ubuntu 22.04** (or 24.04)
   - Shape: **Ampere A1 Flex**, 4 OCPU, 24 GB RAM (all within Always Free). If it says "out of
     capacity", try another availability domain or retry later.
   - Add your SSH public key (or let Oracle generate one and download it).
3. Networking → the instance's subnet → **Security List** → add ingress rules for TCP **80** and **443**
   from `0.0.0.0/0`.

Students: **Azure for Students** (https://azure.microsoft.com/free/students, $100 credit, no card)
works the same way with an Ubuntu B2s VM; open ports 80/443 in its network security group.

## 2. Install the demo (one command)
```bash
ssh ubuntu@<public-ip>
curl -fsSL https://raw.githubusercontent.com/Abdulhafiz0512/cv-project/main/deploy/setup_vm.sh | sudo bash
```
It prints the address, e.g. `https://140-238-1-2.sslip.io`. The first start loads the model (about a minute).
Logs: `journalctl -u junction-watch-demo -f`. Re-run the same command to update after a `git push`.

## 3. Put it on the website
```bash
python tools/set_site_links.py --demo-url https://<ip-with-dashes>.sslip.io
git commit -am "demo link" && git push
```
