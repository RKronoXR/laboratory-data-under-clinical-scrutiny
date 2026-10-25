"""Read-only environment and safe GPU admission checks.

GPU compute PID listings on DGX Spark also include GNOME remote desktop and
other graphical services. Only verified OS graphical clients are ignored;
unknown/actual compute clients continue to block and are reported by name.
"""
from __future__ import annotations
import importlib.util,importlib.metadata,subprocess,shutil,json,os,time
from pathlib import Path
from .config import counts,discover_plans


def check(c):
    if not Path(c['datasets_zip']).is_file():raise FileNotFoundError(c['datasets_zip'])
    packages={'ultralytics':c['ultralytics_version'],'nnunetv2':c['nnunet_version']}
    for package,expected in packages.items():
        got=importlib.metadata.version(package)
        if got!=expected:raise RuntimeError(f'{package} {got} is installed; the supplied code targets {expected}. No package has been upgraded. Obtain a matching code/environment audit before proceeding.')
    for module in ('torch','numpy','pandas','scipy','sklearn','PIL','yaml','matplotlib','dynamic_network_architectures','batchgenerators'):
        if importlib.util.find_spec(module) is None:raise RuntimeError(f'Missing Python dependency: {module}. Use the existing Spark .venv.')
    import torch
    if c['training']['device']!='cpu' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable in the selected Python environment. No automatic CPU fallback.')
    if c['training']['device']!='cpu':print('GPU:',torch.cuda.get_device_name(0))
    from .reuse import legacy_split_path
    if not legacy_split_path(c).is_file():
        raise FileNotFoundError(f'Original folds are required: {legacy_split_path(c)}. No new splits will be generated.')
    print('Original split file:',legacy_split_path(c))
    print('Teacher plan:',discover_plans(c))
    parent=Path(c['output_dir'])
    while not parent.exists():parent=parent.parent
    free=shutil.disk_usage(parent).free/1024**3
    if free<c['reporting'].get('min_free_gb',25):raise RuntimeError(f'Only {free:.1f} GiB free. Below configured minimum. No training started.')
    print(f'Disk free: {free:.1f} GiB. Actual required space depends on candidate retention and dataset caches.')
    if c['statistics']['run_glmm']:
        if shutil.which('Rscript'):
            r=subprocess.run(['Rscript','-e','quit(status=if(all(vapply(c("lme4","emmeans","jsonlite"),requireNamespace,logical(1),quietly=TRUE)))0 else 1)'],capture_output=True,text=True)
            ready=r.returncode==0
        else:ready=False
        if not ready:
            message='R/lme4/emmeans/jsonlite not ready. Run bash INSTALL_STATISTICS.sh before the final statistical report.'
            if c['statistics']['required_before_training']:raise RuntimeError(message)
            print('WARNING:',message)
    print(json.dumps(counts(c),indent=2))
    print(f'Clinical inputs: {c["full_image_size"]}x{c["full_image_size"]}, aspect ratio preserved. ACTA inputs: {c["crop_image_size"]}x{c["crop_image_size"]}.')
    print(f'Teacher full-bitewing input: {c["teacher_image_size"]}x{c["teacher_image_size"]}, original preprocessing preserved.')
    print('Read-only environment checks passed. Source images, masks and split integrity are checked during preparation before training.')


# Strict graphical-service allowlist. Check BOTH executable name and trusted
# system path. Python, custom CUDA, nnU-Net, YOLO and unknown apps are NEVER
# permitted just because utilization is momentarily zero.
_SYSTEM_GUI_EXECUTABLES = frozenset({
    'gnome-remote-desktop-daemon', 'gnome-remote-desktop',
    'gnome-remote-desktop-headless', 'gnome-shell', 'Xorg', 'Xwayland',
    'mutter', 'kwin_wayland', 'kwin_x11', 'weston', 'sway', 'wayfire',
    'gnome-remote-desktop-daemon-headless',
})
_BROWSER_EXE = frozenset({'chrome','chromium','chromium-browser'})


def inspect_gpu_process(pid:int, proc_root='/proc') -> dict:
    """Classify process reported by nvidia-smi; fail closed on ambiguity."""
    proc = Path(proc_root)/str(pid)
    entry = dict(pid=pid, classification='BLOCK', reason='unrecognised_gpu_process', exe=None, command='')
    if not proc.exists():
        entry.update(classification='GONE',reason='stale_nvidia_smi_pid')
        return entry
    try:
        exe = os.readlink(proc/'exe').removesuffix(' (deleted)')
        entry['exe'] = exe
    except OSError:
        exe = None
        entry['reason'] = 'cannot_read_executable'
    try:
        raw = (proc/'cmdline').read_bytes()
        entry['command'] = ' '.join(x.decode('utf-8','replace') for x in raw.split(b'\0') if x)[:600]
    except OSError:
        pass
    try:
        status = (proc/'status').read_text(errors='replace')
        state_line = next((line for line in status.splitlines() if line.startswith('State:')), '')
        if '\tZ' in state_line or '\tX' in state_line:
            entry.update(classification='GONE',reason='exited_or_zombie_process')
            return entry
    except OSError:
        pass
    if exe and exe.startswith('/usr/') and Path(exe).name in _SYSTEM_GUI_EXECUTABLES:
        entry.update(classification='DISPLAY',reason='verified_system_graphics_service')
    elif exe and (exe.startswith('/usr/') or exe.startswith('/opt/google/chrome/')) and Path(exe).name in _BROWSER_EXE and '--type=gpu-process' in entry['command']:
        entry.update(classification='DISPLAY',reason='verified_browser_gpu_helper')
    return entry


def gpu_process_status(*, own_pid=None, proc_root='/proc') -> dict:
    """Return a diagnostic snapshot, independent of CUDA initialization."""
    r = subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],
                       capture_output=True,text=True,timeout=20)
    if r.returncode != 0:
        raise RuntimeError('nvidia-smi GPU process query failed: '+r.stderr.strip()[:300])
    pids=set()
    for value in r.stdout.splitlines():
        val=value.strip()
        if val.isdigit() and int(val) != (own_pid or os.getpid()):pids.add(int(val))
        elif val and val.lower() not in ('no running processes found','[not supported]','n/a'):
            raise RuntimeError(f'Unexpected nvidia-smi process listing: {val[:100]}')
    processes=[inspect_gpu_process(pid,proc_root) for pid in sorted(pids)]
    return dict(display=[p for p in processes if p['classification']=='DISPLAY'],
                blockers=[p for p in processes if p['classification']=='BLOCK'],
                vanished=[p for p in processes if p['classification']=='GONE'])


def gpu_status_text(status):
    lines=['ACTA KD GPU GATE',f"System display/desktop processes ignored: {len(status['display'])}",
           f"Other GPU processes blocking training: {len(status['blockers'])}"]
    for cls,label in (('display','IGNORED (desktop)'),('blockers','BLOCKING (compute/unknown)'),('vanished','IGNORED (exited)')):
        for p in status[cls]:
            lines += [f"  {label}: PID={p['pid']} executable={p.get('exe') or 'unknown'}",
                      f"    reason={p['reason']} command={p.get('command','')}"]
    return '\n'.join(lines)+'\n'


def wait_for_gpu(c):
    if c['training']['device']=='cpu':return
    from .io import check_pause,event,atomic_bytes
    root=Path(c['output_dir'])
    alert=root/'GPU_BLOCKERS.txt'
    last_display=None
    while True:
        check_pause(root)
        status=gpu_process_status()
        if status['blockers']:
            text=gpu_status_text(status)
            atomic_bytes(alert,(f"GPU admission blocked; no training started.\n{time.strftime('%Y-%m-%d %H:%M:%S')}\n"+text).encode())
            event(root,'waiting_for_other_gpu_jobs',pids=[x['pid'] for x in status['blockers']],
                  blockers=[dict(pid=p['pid'],executable=p['exe'],command=p['command'],reason=p['reason']) for p in status['blockers']],
                  ignored_display_pids=[x['pid'] for x in status['display']],alert_file=str(alert))
            time.sleep(30)
            continue
        alert.unlink(missing_ok=True)
        display_ids=tuple(p['pid'] for p in status['display'])
        if display_ids != last_display:
            event(root,'gpu_gate_ready',ignored_display_pids=list(display_ids),
                  note='Known OS desktop/remote-graphics processes are not GPU training blockers.')
        return


if __name__=='__main__':
    print(gpu_status_text(gpu_process_status()))
