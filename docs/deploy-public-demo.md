# Publishing the demo without a domain

Two pieces, deployed separately, because they have opposite cost profiles.

| | What it is | Where it lives | Cost |
|---|---|---|---|
| **The page** | static HTML with five recorded episodes | Cloudflare Pages, `*.pages.dev` | free, always on |
| **The live backend** | the agent on a GPU | EC2, reached through a Cloudflare tunnel | ~$1/h, only while you run it |

The page works on its own. When the backend is up, the same page notices and turns on
the upload panel. Neither needs a domain name.

---

## 1. The page — Cloudflare Pages

### One-time: authenticate

Wrangler needs your Cloudflare account. Run this yourself (it opens a browser):

```bash
npx wrangler login
```

There is no need to create a Cloudflare account in advance — the login flow will make
one. The free tier covers this entirely.

### Build and deploy

```bash
python scripts/make_replay.py \
  --out site/index.html \
  --repo https://github.com/<you>/roadsense-agent

npx wrangler pages deploy site --project-name roadsense
```

The first deploy creates the project and prints a URL like
`https://roadsense.pages.dev`. Every later deploy of the same project name updates
that same URL.

**What is in it:** five real episodes, images embedded as data URIs, no network calls
except Google Fonts. No API keys, no backend, nothing to attack, nothing to bill.

---

## 2. The live backend — Cloudflare tunnel

A page served over HTTPS cannot call `http://1.2.3.4:8080`: browsers block mixed
content, and the EC2 box has no certificate. A tunnel solves both and needs no
domain, no DNS and no Cloudflare account.

### On the GPU instance

```bash
# once
curl -L -o cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x cloudflared && sudo mv cloudflared /usr/local/bin/

# each session: start the agent with limits, then the tunnel
setsid /home/ubuntu/start_web.sh > /home/ubuntu/web.log 2>&1 &
cloudflared tunnel --url http://localhost:8080
```

`cloudflared` prints a URL like `https://odd-word-pair-1234.trycloudflare.com`. It is
HTTPS, it is free, it needs no account, and it disappears when you stop the process —
which is exactly the lifetime you want for something that spends money.

### Point the page at it

No redeploy needed. Append the tunnel URL as a query parameter:

```
https://roadsense.pages.dev/?api=https://odd-word-pair-1234.trycloudflare.com
```

The page probes `/api/status`, sees a server, and enables uploads. Share the bare
`pages.dev` link for the permanent demo and the `?api=` link only while you are
watching the bill.

### Start the server with limits when you do this

```bash
# in start_web.sh, or directly
python -m roadsense.web.server --host 127.0.0.1 --port 8080 --public
```

`--public` means 200 runs/day, 10 per client per hour, and samples only. Add
`--token "$SECRET"` if the link will be somewhere public; the page forwards it from
`?token=`.

Bind to `127.0.0.1`, not `0.0.0.0`: the tunnel reaches it locally, so there is no
reason to open the port to the internet at all. You can then remove the 8080 rule
from the security group entirely.

---

## Why not host the agent itself on Cloudflare

Pages and Workers run JavaScript at the edge with no GPU and a short execution budget.
This agent needs 4.5 GB of model weights resident on a CUDA device, plus AWS
credentials for Rekognition and Bedrock. None of that fits, and the parts that could
be moved would stop the demo being a demo of this system.

So the split above is not a compromise. The static page is genuinely the right artifact
for a permanent public link, and the live backend is genuinely something you want off
by default.

---

## Checklist before sharing a link

- [ ] `site/index.html` rebuilt from the current `eval/results/results_gpu.json`
- [ ] repo URL in the footer is the real one, not a placeholder
- [ ] if sharing `?api=`, the server was started with `--public`
- [ ] GPU instance stopped afterwards: `aws ec2 stop-instances --instance-ids <id>`
