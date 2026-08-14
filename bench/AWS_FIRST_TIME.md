# AWS_FIRST_TIME — launching a Graviton box, start to finish

For someone who has never used AWS. Covers account setup through termination.
Once the instance is up and you have a shell,
[`GRAVITON_SESSION.md`](GRAVITON_SESSION.md) is what you execute — this file
only gets you to that shell and safely back off it.

**Budget:** a `c8g.16xlarge` is roughly **$2.30–3.00/hour** on-demand. A 3-hour
session is **≈$9**.

> **The one number that matters:** you are billed by the second while the
> instance *exists*, whether or not you are using it. Every warning in this file
> about terminating is about that.

---

# Phase 0 — Do this NOW, days before the session

## 0.1 Pick a region and never change it

Top-right of the console. **`us-east-1` (N. Virginia)** is the default choice —
cheapest, best capacity for newer instance types.

**Key pairs, security groups, and instances are all per-region.** The classic
first-timer hour is spent hunting for a key pair created in a region you are no
longer looking at. Pick one. Write it down.

## 0.2 Set a billing alarm before anything else

Even on credits. Billing → **Budgets** → Create budget → Cost budget → set
**$25** with an email alert at 80%.

This is the difference between "I left it running overnight, that was $70" and
an email at 2am telling you so.

## 0.3 Check your vCPU quota — THIS IS THE BLOCKER

**Do this first, because it is the only step that can take days.**

Console → search **Service Quotas** → AWS services → **Amazon EC2** → search for:

> **Running On-Demand Standard (A, C, D, H, I, M, R, T, Z) instances**

**That number is in vCPUs, not instances.**

| You want | vCPUs needed |
|---|---|
| `c8g.16xlarge` | **64** |
| `c8g.8xlarge` | **32** |

New accounts are frequently capped at **5–32**. If yours is under what you need,
click **Request increase at account level**, ask for 64, and expect **hours to
days**. Requesting costs nothing.

> **If the quota fight drags on, take `c8g.8xlarge` (32 vCPU).** A 1→32 core
> scaling sweep is six points and tells the story nearly as well. Do not lose
> days to the last doubling.

## 0.4 Know which student account you have

They are not equivalent, and one of them will not work:

| Account | Verdict |
|---|---|
| **GitHub Student Pack** credits on a normal AWS account | ✅ Fine |
| **AWS Educate** | ⚠️ Often restricts instance types |
| **AWS Academy Learner Lab** | ❌ **Blocks larger instance types and auto-terminates sessions (~4 h).** A 3-hour benchmark run is at real risk |

If you are on Learner Lab, check whether `c8g` is even launchable before
planning the session around it.

## 0.5 Create a key pair

EC2 → **Key Pairs** (left sidebar, under Network & Security) → Create key pair.

- Name: `graviton-key`
- Type: **RSA**
- Format: **`.pem`**

The file downloads **once**. If you lose it you cannot get into the instance —
you make a new key pair and relaunch. Put it somewhere you will find it.

### Lock down its permissions, or SSH will refuse it

SSH rejects a private key that other users can read. This step differs by OS.

**Windows (PowerShell):**

```powershell
icacls "$env:USERPROFILE\Downloads\graviton-key.pem" /inheritance:r /grant:r "$($env:USERNAME):(R)"
```

**Linux / macOS:**

```bash
chmod 400 ~/Downloads/graviton-key.pem
```

---

# Phase 1 — Launch the instance

EC2 → **Instances** → **Launch instances**.

## 1.1 Name

`graviton-bench`

## 1.2 AMI — **the most common expensive mistake**

Select **Ubuntu**, then **Ubuntu Server 24.04 LTS**.

> ### ⚠️ Set Architecture to **64-bit (Arm)**
>
> There is an architecture dropdown directly under the AMI selector. It defaults
> to **64-bit (x86)**.
>
> **If you leave it on x86, the instance will not launch with a `c8g` type — and
> if you also switch to an x86 instance type to make the error go away, you will
> spend three hours benchmarking an Arm kernel on an x86 machine.** That is the
> whole project, silently voided.
>
> The AMI description must say **arm64**.

## 1.3 Instance type

Search `c8g.16xlarge` (Graviton4, Neoverse-V2, 64 vCPU).

Fallbacks, in order: `c8g.8xlarge` (32 vCPU) → `c7g.16xlarge` (Graviton3).

## 1.4 Key pair

Select **`graviton-key`** from 0.5.

## 1.5 Network settings → Edit

- ✅ **Allow SSH traffic from** → change the dropdown to **My IP**

Not "Anywhere (0.0.0.0/0)". Your IP is enough, and an open SSH port on a public
box gets scanned within minutes.

*(If your IP changes — different wifi, VPN toggling — you will lose access and
need to edit the security group's inbound rule to your new IP. Not fatal, just
confusing the first time.)*

## 1.6 Storage

Change the root volume from the default 8 GiB to **40 GiB**, gp3.

You need room for torch, transformers, mamba-130m, **both 187M Mamba-3
checkpoints (~357 MB each)** and the 53 MB of goldens in the repo. Running out
of disk 40 minutes in is a painful way to learn this.

## 1.7 Launch

Click **Launch instance**, then **View all instances**. Wait for:

- **Instance state:** Running
- **Status checks:** 2/2 passed *(takes ~1 min; SSH will refuse before this)*

---

# Phase 2 — Connect

Copy the **Public IPv4 address** from the instance row.

```bash
ssh -i ~/Downloads/graviton-key.pem ubuntu@<PUBLIC-IP>
```

On Windows, PowerShell has `ssh` built in — same command, but the path looks
like `$env:USERPROFILE\Downloads\graviton-key.pem`.

First connection asks about host authenticity — type `yes`.

**The username is `ubuntu`.** Not `root`, not your name. A different AMI would
use a different user, which is a common source of "Permission denied".

### Confirm you got what you paid for — before spending an hour on it

```bash
nproc && uname -m && lscpu | grep -i "model name"
```

- [ ] `nproc` → **64** (or 32 on the 8xlarge)
- [ ] `uname -m` → **`aarch64`** ← if this says `x86_64`, you booted the wrong
      AMI. Terminate and relaunch. Do not proceed.
- [ ] `uptime` load ≈ 0 — a contended benchmark is void

---

# Phase 3 — Run the session

```bash
git clone https://github.com/AdityaP9116/Arm-Scan && cd Arm-Scan
bash bench/setup_ampere.sh
```

Then follow **[`GRAVITON_SESSION.md`](GRAVITON_SESSION.md)** from §2. In short:

| § | What | Time |
|---|---|---|
| 2 | Correctness gate — **abort point** | 10 min |
| 3 | Mamba-1 baseline + core-scaling curve | 90 min |
| **3b** | **Mamba-3 — the headline work** | 35 min |
| 4 | Profiling | 20 min |

If you are short on time, §6 of that file is the reduced plan — it puts Mamba-3
first, because that is the part with no Arm numbers of any kind.

> **Use `tmux`** so a dropped connection does not kill a 90-minute run:
> ```bash
> tmux new -s bench
> ```
> Detach with `Ctrl-b` then `d`; reattach with `tmux attach -t bench`.
> Without this, closing your laptop lid can end a benchmark you are paying for.

---

# Phase 4 — Retrieve results, then TERMINATE

## 4.1 Pack up (on the instance)

```bash
cd ~/Arm-Scan && tar czf results-graviton.tgz bench/results session.log \
    ct.log mamba3.log m3-model.log bench/profile/out
```

## 4.2 Copy to your laptop (run this on your laptop, not the instance)

```bash
scp -i ~/Downloads/graviton-key.pem ubuntu@<PUBLIC-IP>:~/Arm-Scan/results-graviton.tgz .
```

- [ ] The archive is on your laptop and opens

## 4.3 Terminate — **not stop**

EC2 → Instances → select it → **Instance state** → **Terminate instance**.

| Action | Compute billing | Storage billing | Recoverable |
|---|---|---|---|
| **Stop** | stops | **continues** | yes |
| **Terminate** | stops | stops | **no** |

**Stopping is not enough.** The EBS volume keeps billing. Since everything you
need is in the tarball and the repo is on GitHub, terminate.

- [ ] Instance state reads **terminated**
- [ ] Refresh the Instances list and confirm nothing else is running
- [ ] Check **Billing → Bills** the next day

---

# The five ways this goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| `uname -m` says `x86_64` | Architecture dropdown left on x86 | Terminate, relaunch with **64-bit (Arm)** |
| Instance type not selectable | vCPU quota, or a restricted student account | §0.3 / §0.4, or drop to `c8g.8xlarge` |
| `Permission denied (publickey)` | Wrong username, or `.pem` permissions | User is **`ubuntu`**; re-run the `icacls`/`chmod` in §0.5 |
| SSH hangs with no response | Security group has no SSH rule from your current IP | Edit inbound rules → SSH → My IP |
| A surprise bill | Stopped instead of terminated | §4.3, and set the budget alarm in §0.2 |
