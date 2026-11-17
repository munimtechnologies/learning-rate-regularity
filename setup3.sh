#!/bin/sh
# learning-rate-regularity phase 3 (munimtechnologies/learning-rate-regularity, PLAN-3.md).
# 1) Wait until PLAN-3.md, phase3.py, fit3.R, test_phase3.py and realdata.py are on main, then run exactly those
#    public copies.
# 2) Install R and lme4 from Debian packages (fallback: CRAN source build).
set -eu
python - <<'EOF'
import hashlib, time, urllib.request
REPO = "https://raw.githubusercontent.com/munimtechnologies/learning-rate-regularity/main/"
NAMES = ("PLAN-3.md", "phase3.py", "fit3.R", "test_phase3.py", "realdata.py")


def norm(data):
    return data.replace(b"\r\n", b"\n")


for attempt in range(600):
    got, missing = {}, []
    for name in NAMES:
        try:
            got[name] = urllib.request.urlopen(REPO + name + "?t=%d" % attempt, timeout=60).read()
        except Exception as exc:
            missing.append("%s (%s)" % (name, exc))
    if not missing:
        break
    print("waiting for main:", "; ".join(missing), flush=True)
    time.sleep(60)
else:
    raise SystemExit("PLAN-3 files never appeared on main; not running")
for name, data in got.items():
    try:
        same = norm(open(name, "rb").read()) == norm(data)
    except OSError:
        same = False
    open(name, "wb").write(data)
    print(name, "sha256 (LF)", hashlib.sha256(norm(data)).hexdigest(),
          "same as filed copy" if same else "DIFFERS from filed copy (public copy used)", flush=True)
print("preregistered files fetched from main", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), flush=True)
EOF
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
if ! apt-get install -y -qq --no-install-recommends r-base-core r-cran-lme4; then
  apt-get install -y -qq --no-install-recommends r-base-core r-base-dev
  Rscript -e 'install.packages("lme4", repos = "https://cloud.r-project.org")'
fi
Rscript -e 'suppressPackageStartupMessages(library(lme4)); cat(R.version.string, "lme4", as.character(packageVersion("lme4")), "\n")'
