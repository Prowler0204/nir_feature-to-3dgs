# patches/

## gsplat_msvc_fix.patch

**Problem:** `gsplat`'s JIT build passes `-Wno-attributes` (a GCC flag) to `extra_cflags`,
which is routed to the host C++ compiler. On Windows, this is MSVC's `cl.exe`, which
rejects it with `D8021: invalid numeric argument '/Wno-attributes'`, crashing the build.

**Root cause:** `gsplat/cuda/_backend.py` line 177 (v1.5.3):
```python
extra_cflags = [opt_level, "-Wno-attributes"]
```

**Fix:** Guard the flag behind `sys.platform == "win32"` so it is only omitted for MSVC.
Linux/GCC builds are unaffected.

**Applying the patch after a fresh gsplat install:**
```bash
# From the repo root (after conda activate gaussian_splatting):
python -c "
import site, os
sp = site.getsitepackages()[0]
src = os.path.join(sp, 'gsplat', 'cuda', '_backend.py')
with open(src) as f: txt = f.read()
txt = txt.replace(
    \"import shutil\\nimport time\",
    \"import shutil\\nimport sys\\nimport time\"
).replace(
    'extra_cflags = [opt_level, \"-Wno-attributes\"]',
    '_msvc = sys.platform == \"win32\"\\n        extra_cflags = [opt_level] if _msvc else [opt_level, \"-Wno-attributes\"]'
)
with open(src, 'w') as f: f.write(txt)
print('Patch applied to', src)
"
```

Or apply manually with `git apply patches/gsplat_msvc_fix.patch` if working from a
cloned gsplat source tree.
