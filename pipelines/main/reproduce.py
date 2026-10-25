#!/usr/bin/env python3
"""Portable entry point for the eight main manuscript training strategies.

The archived numerical engines are vendored under yolo/ and nnunet/; bounded
portable changes are recorded in UPSTREAM_MANIFEST.json.
This entry point supplies portable paths, a main-strategy-only dependency graph,
separate training/assessment, and friendly result names.
"""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONFIG = ROOT / "configs" / "main"
STRATEGIES = {
    "nnunet_clinical": ("yolo", "T"),
    "nnunet_laboratory_pretraining": ("nnunet", "E"),
    "nnunet_laboratory_distillation": ("nnunet", "D"),
    "nnunet_pretraining_and_distillation": ("nnunet", "F"),
    "yolo_clinical": ("yolo", "A"),
    "yolo_laboratory_pretraining": ("yolo", "B"),
    "yolo_clinical_distillation": ("yolo", "C"),
    "yolo_pretraining_and_distillation": ("yolo", "D"),
}
SEEDS = [17, 42, 2026]
FOLDS = [0, 1, 2, 3, 4]

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    payload=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as f:
        f.write(payload);tmp=Path(f.name)
    os.replace(tmp,path)

def cells(args):
    return [(seed,fold) for fold in (args.fold if args.fold is not None else FOLDS)
            for seed in (args.seed if args.seed is not None else SEEDS)]

def paths(args):
    return Path(args.output).expanduser().resolve(),Path(args.data_zip).expanduser().resolve()

def dataset_names(path):
    source=str(ROOT/"src")
    if source not in sys.path:sys.path.insert(0,source)
    from carieslab.datasets import detect_raw_names
    return detect_raw_names(path)

def configure_data_aliases(path):
    names=dataset_names(path)
    for package in ["actakd","reversekd"]:
        source=importlib.import_module(package+"._source")
        source.RAW_NAMES.update(names)
    return names

def verify_scientific_inputs(path,annotation_version="recorded"):
    """Allow enclosing-folder changes while verifying the frozen input bytes."""
    manifest="data_fingerprints_updated.json" if annotation_version=="updated" else "data_fingerprints.json"
    names=dataset_names(path);expected=read(CONFIG/manifest)["files"]
    actual={};missing=[];changed=[]
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            parts=Path(name).parts
            for domain,component in names.items():
                if component in parts:
                    rel="/".join(parts[parts.index(component)+1:])
                    actual[(domain,rel)]=name
        for item in expected:
            name=actual.get((item["domain"],item["path"]))
            if name is None:missing.append({"domain":item["domain"],"path":item["path"]});continue
            h=hashlib.sha256()
            with z.open(name) as f:
                for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
            if h.hexdigest()!=item["sha256"]:changed.append({"domain":item["domain"],"path":item["path"],"expected_sha256":item["sha256"],"actual_sha256":h.hexdigest()})
    return {"verified":not missing and not changed,"files_checked":len(expected),"missing":missing,"changed":changed}

def plan(args):
    selected=cells(args);out,data=paths(args)
    return {"study":"main manuscript strategies","requested_cells":[{"seed":s,"fold":f} for s,f in selected],
        "complete_manuscript_matrix":len(selected)==15,"models_per_strategy":len(selected),
        "clinical_endpoint_models":8*len(selected),"shared_laboratory_pretraining_models":2*len(selected),
        "training_stages":10*len(selected),"strategies":list(STRATEGIES),
        "data_zip":str(data),"output":str(out),"hyperparameter_search":False,"annotation_version":getattr(args,"annotation_version","recorded"),
        "assessment_requires":"All requested endpoint models and development thresholds must be frozen first."}

def configs(args):
    out,data=paths(args)
    y=read(CONFIG/"yolo.json");n=read(CONFIG/"nnunet.json")
    for c,sub,pkg,profile in [(y,"yolo","actakd","NO_HPO"),(n,"nnunet","reversekd","CONTROLLED_V3_NO_HPO")]:
        c.update(project_root=str(ROOT),package_root=str(HERE/sub),datasets_zip=str(data),
            config_path=str(CONFIG/(sub+".json")),output_base=str(out/sub),output_dir=str(out/sub/profile),
            profile=profile,run_id=profile,run_name=None,hpo_enabled=[])
        c["training"]["device"]=args.device
        c["annotation_version"]=getattr(args,"annotation_version","recorded")
        c["teacher_reuse"]["legacy_splits"]=str(CONFIG/"clinical_splits.json")
        c["teacher_reuse"]["legacy_root"]=str(out/"prior_teachers")
    y["teacher_plans"]=str(CONFIG/"clinical_plans.json")
    n.update(student_plans=str(CONFIG/"clinical_plans.json"),source_v3=str(HERE/"yolo"),
        source_output=y["output_dir"],source_profile=y["profile"])
    return y,n

def check(args):
    issues=[];manifest=read(HERE/"UPSTREAM_MANIFEST.json")
    for item in manifest["files"]:
        path=ROOT/item["path"]
        if not path.is_file():issues.append("Missing source file: "+item["path"])
        elif sha(path)!=item.get("bundled_sha256",item["sha256"]):issues.append("Changed source file: "+item["path"])
    for item in ["yolo.json","nnunet.json","clinical_plans.json","clinical_splits.json","native_recipe.json","data_fingerprints.json"]:
        if not (CONFIG/item).is_file():issues.append("Missing configuration: "+item)
    splits=read(CONFIG/"clinical_splits.json")
    if len(splits)!=5 or [len(x["val"]) for x in splits]!=[20,20,19,21,20]:issues.append("Original clinical splits changed.")
    for x in splits:
        if set(x["train"])&set(x["val"]) or len(x["train"])+len(x["val"])!=100:issues.append("Invalid clinical split.")
    p=read(CONFIG/"clinical_plans.json")["configurations"]["2d"]
    if p["patch_size"]!=[384,192] or p["batch_size"]!=44:issues.append("Clinical native plan changed.")
    _,data=paths(args)
    input_identity=None
    if not data.is_file():issues.append("Required study input not found: "+str(data)+". See docs/data.md; original public DCBR boxes alone do not supply the refined masks and anatomical regions.")
    elif not zipfile.is_zipfile(data):issues.append("The study data input is not a ZIP archive.")
    else:
        try:
            detected=dataset_names(data)
            with zipfile.ZipFile(data) as z:
                names=z.namelist()
                if not any(x.endswith("Surface_Review_085.csv") for x in names):issues.append("Required frozen DCBR region-review CSV is absent.")
                for domain in [detected["dcbr"],detected["bw25"]]:
                    if not any(domain in Path(x).parts and x.endswith("Proximal_Side_Regions.csv") for x in names):issues.append("Required anatomical region coordinates absent for "+domain)
            input_identity=verify_scientific_inputs(data,getattr(args,"annotation_version","recorded"))
            if not input_identity["verified"]:issues.append(f"Scientific inputs differ from the archived experiment: {len(input_identity['missing'])} missing and {len(input_identity['changed'])} changed files. No training has started.")
        except (ValueError,FileNotFoundError) as exc:issues.append(str(exc))
    modules=["numpy","pandas","PIL","yaml","scipy","sklearn","matplotlib","torch","ultralytics","nnunetv2","dynamic_network_architectures","batchgenerators","batchgeneratorsv2","blosc2"]
    installed={name:importlib.util.find_spec(name) is not None for name in modules}
    if args.runtime:
        for name,ok in installed.items():
            if not ok:issues.append("Missing dependency: "+name)
        pinned=read(CONFIG/"nnunet.json")["source_runtime_versions"]
        for name,want in pinned.items():
            try:got=importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:continue
            if got.split("+")[0]!=want.split("+")[0]:issues.append(f"Runtime mismatch: {name}={got}, required {want}")
        if installed["torch"] and args.device!="cpu":
            import torch
            if not torch.cuda.is_available():issues.append("CUDA is unavailable; no CPU fallback is selected.")
    report={"status":"PASS" if not issues else "INPUTS_REQUIRED","issues":issues,"installed_modules":installed,"input_identity":input_identity,
        "checked_source_files":len(manifest["files"]),"plan":plan(args)}
    print(json.dumps(report,indent=2));return 0 if not issues else 2

def engines():
    for sub in ["yolo","nnunet"]:
        path=str(HERE/sub)
        if path not in sys.path:sys.path.insert(0,path)
    from actakd import pipeline as forward
    from reversekd import pipeline as reverse
    from reversekd import report as reverse_report
    # Public summaries are built below; old exploratory report menus are not run.
    forward._report=lambda c,plots=False:None
    reverse_report.refresh=lambda c,plots=False:None
    return forward,reverse

def require_training_runtime(args):
    """Reject an unverified training engine before starting any study fit."""
    required=read(CONFIG/"nnunet.json")["source_runtime_versions"]
    for name,want in required.items():
        try:got=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:raise RuntimeError("Required training package is missing: "+name) from exc
        if got.split("+")[0]!=want.split("+")[0]:raise RuntimeError(f"Training runtime mismatch: {name}={got}, required {want}. No fit was started.")
    os.environ["CUDA_VISIBLE_DEVICES"]="" if args.device=="cpu" else args.device
    import torch
    if args.device!="cpu" and not torch.cuda.is_available():raise RuntimeError("CUDA is unavailable; no CPU fallback was selected.")

def identity(args,y,n):
    out,data=paths(args)
    if not data.is_file():raise FileNotFoundError("Supply the frozen study Datasets.zip with --data-zip; see docs/data.md.")
    version=getattr(args,"annotation_version","recorded")
    inputs=verify_scientific_inputs(data,version)
    if not inputs["verified"]:
        raise RuntimeError(f"Study inputs do not match the archived scientific data: {len(inputs['missing'])} missing, {len(inputs['changed'])} changed files. Run main check for the exact file list. No mask was altered and no model was trained.")
    fingerprint="data_fingerprints_updated.json" if version=="updated" else "data_fingerprints.json"
    current={"schema":1,"annotation_version":version,"data_sha256":sha(data),"configuration":{x:sha(CONFIG/x) for x in ["yolo.json","nnunet.json","clinical_plans.json","clinical_splits.json","native_recipe.json",fingerprint]},
        "backend_sha256":sha(__file__),"upstream_manifest_sha256":sha(HERE/"UPSTREAM_MANIFEST.json"),
        "device":args.device,"requested_cells":plan(args)["requested_cells"],"strategies":list(STRATEGIES)}
    target=out/"RUN_MANIFEST.json"
    if target.exists() and read(target)!=current:raise RuntimeError("This output folder belongs to a different data/configuration/run matrix. Use another --output folder.")
    write(target,current)
    for c in [y,n]:
        c["reporting"]["prune_unselected_checkpoints"]=False
        Path(c["output_dir"]).mkdir(parents=True,exist_ok=True)
        protocol={"signature":digest(current),"public_main_strategy_run":True,"configuration":c,"counts":plan(args)}
        write(Path(c["output_dir"])/"protocol.json",protocol)
    return digest(current)

def prepare(args):
    y,n=configs(args);identity(args,y,n)
    for sub in ["yolo","nnunet"]:
        path=str(HERE/sub)
        if path not in sys.path:sys.path.insert(0,path)
    from actakd.data import prepare as yp
    from reversekd.data import prepare as np
    configure_data_aliases(y["datasets_zip"])
    yp(y);np(n)
    print("Prepared development data and exact original folds. Assessment data remain locked.")
    return 0

def baseline_recipe(n,result):
    import torch
    from reversekd.source_baselines import derive_native_recipe
    payload=torch.load(result["checkpoint"],map_location="cpu",weights_only=False)
    recipe=derive_native_recipe(n,result,payload)
    expected=read(CONFIG/"native_recipe.json")
    if recipe["recipe_signature"]!=expected["recipe_signature"]:
        raise RuntimeError("Fresh baseline effective native recipe differs from the archived clinical recipe; no silent substitution.")
    return recipe

def train(args):
    y,n=configs(args);protocol=identity(args,y,n);require_training_runtime(args);f,r=engines();configure_data_aliases(y["datasets_zip"])
    from actakd.data import prepare as yp
    from reversekd.data import prepare as np
    from actakd.search import candidates
    from actakd.io import atomic_json
    from reversekd.check import freeze_plans
    from reversekd.professor import audit_one,snapshot,audit_source_data
    data=yp(y);freeze_plans(n);ndata=np(n);audit_source_data(n,ndata)
    grid=candidates(y)
    for seed,fold in cells(args):
        cell=f"seed_{seed}_fold_{fold}";split=next(s for s in data["splits"] if s["fold"]==fold)
        teacher=f._get_selected(y,cell,"T")
        if teacher is None:
            teacher=f._train(y,protocol,cell,"T/fixed","teacher","T",grid["teacher"][0],split)
            f._save_selected(y,cell,"T",teacher)
        train_images=[x for x in data["images"] if x["sample_id"] in split["train_ids"]]
        validation=[x for x in data["images"] if x["sample_id"] in split["validation_ids"]]
        soft=f._predict(y,cell,"T",teacher,train_images,"distillation_train_cache")
        from actakd.preflight import smoke_student
        smoke_student(y,protocol,data,split,soft)
        pretrained=f._pretrain_search(y,protocol,cell,grid["pretraining"],split)
        initial={k:pretrained[k] for k in ["checkpoint","checkpoint_sha256"]}
        for label in ["A","B","C","D"]:
            result=f._get_selected(y,cell,label)
            if result is None:
                result=f._train(y,protocol,cell,f"clinical/fixed/{label}","student",label,grid["clinical"][0],split,
                    initial if label in ["B","D"] else None,soft if label in ["C","D"] else None,
                    float(y["training"]["distillation_weight"]) if label in ["C","D"] else 0.)
                f._save_selected(y,cell,label,result)
            f._evaluate(y,cell,label,result,validation,data["surfaces"],"development_validation")
        f._evaluate(y,cell,"T",teacher,validation,data["surfaces"],"development_validation")
        native_recipe=baseline_recipe(n,teacher)
        baseline={**copy.deepcopy(teacher),"native_recipe":native_recipe}
        ntrain=[x for x in ndata["images"] if x["sample_id"] in split["train_ids"]]
        nval=[x for x in ndata["images"] if x["sample_id"] in split["validation_ids"]]
        professor=snapshot(n,audit_one(n,seed,fold))
        gate=Path(n["output_dir"])/"preflight"/cell
        fit={**n,"native_recipe":native_recipe,"native_bundle":baseline["native_bundle"],
            "source_native_bundle":baseline["native_bundle"],"native_gate_path":str(gate/"NATIVE_EQUIVALENCE.json")}
        targets=r.target_task(fit,cell,professor,ntrain,ndata["surfaces"],"training_only",protocol)
        if not (gate/"result.json").is_file():
            r.invoke(fit,{"operation":"preflight","task_id":cell+"/native_preflight","directory":str(gate),
                "teacher":professor,"teacher_maps":{ntrain[0]["sample_id"]:targets[ntrain[0]["sample_id"]]},
                "baseline_model":baseline,"native_recipe":native_recipe,"protocol":protocol})
        stage_results={"A":baseline}
        for stage,parent,label,kd in [("K","A","D",True),("V",None,None,False),("E_CLIN","V","E",False),("F_CLIN","V","F",True)]:
            result=r.train_task(fit,cell,stage,split,protocol,stage_results[parent] if parent else None,targets if kd else None)
            stage_results[stage]=result
            if label:
                atomic_json(Path(n["output_dir"])/"selections"/cell/(label+".json"),result)
                r.evaluate(fit,cell,label,result,nval,ndata["surfaces"],"development_validation",protocol)
        write(Path(args.output).resolve()/"cells"/(cell+".json"),{"seed":seed,"fold":fold,"trained":True})
    frozen={}
    for seed,fold in cells(args):
        cell=f"seed_{seed}_fold_{fold}"
        for friendly,(family,label) in STRATEGIES.items():
            c=y if family=="yolo" else n;folder=Path(c["output_dir"])
            selected=read(folder/"selections"/cell/(label+".json"))
            if sha(selected["checkpoint"])!=selected["checkpoint_sha256"]:raise RuntimeError("Selected weights changed.")
            threshold=folder/"thresholds"/cell/(label+".json")
            frozen[cell+"/"+friendly]={"checkpoint_sha256":selected["checkpoint_sha256"],"threshold_sha256":sha(threshold)}
    freeze={"protocol":protocol,"models":frozen,"requested_cells":plan(args)["requested_cells"]}
    for c in [y,n]:write(Path(c["output_dir"])/"DEVELOPMENT_FROZEN.json",freeze)
    write(Path(args.output).resolve()/"DEVELOPMENT_FROZEN.json",freeze)
    export_results(args,y,n)
    print("All requested models and development thresholds are frozen. Run assess next.")
    return 0

def assess(args):
    y,n=configs(args);protocol=identity(args,y,n);require_training_runtime(args);f,r=engines();configure_data_aliases(y["datasets_zip"]);out,_=paths(args)
    frozen=read(out/"DEVELOPMENT_FROZEN.json")
    for seed,fold in cells(args):
        cell=f"seed_{seed}_fold_{fold}"
        for friendly,(family,label) in STRATEGIES.items():
            c=y if family=="yolo" else n;folder=Path(c["output_dir"]);result=read(folder/"selections"/cell/(label+".json"))
            item=frozen["models"][cell+"/"+friendly]
            if sha(result["checkpoint"])!=item["checkpoint_sha256"] or sha(folder/"thresholds"/cell/(label+".json"))!=item["threshold_sha256"]:
                raise RuntimeError("Frozen model or threshold changed; assessment was not run.")
    from actakd.data import prepare as yp
    from reversekd.data import prepare as np
    yf=yp(y,reference=True);nf=np(n,reference=True)
    for seed,fold in cells(args):
        cell=f"seed_{seed}_fold_{fold}"
        for label in ["T","A","B","C","D"]:
            result=f._get_selected(y,cell,label)
            f._evaluate(y,cell,label,result,yf["images"],yf["surfaces"],"reference25")
        for label in ["D","E","F"]:
            result=read(Path(n["output_dir"])/"selections"/cell/(label+".json"))
            fit={**n,"native_recipe":result["native_recipe"]}
            r.evaluate(fit,cell,label,result,nf["images"],nf["surfaces"],"reference25",protocol)
    export_results(args,y,n);write(out/"ASSESSMENT_COMPLETE.json",{"protocol":protocol,"requested_cells":plan(args)["requested_cells"]})
    print(out/"reports");return 0

def write_csv(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for row in rows for k in row));path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)

def export_results(args,y,n):
    out,_=paths(args);metric_rows=[];surface_rows=[]
    for friendly,(family,label) in STRATEGIES.items():
        c=y if family=="yolo" else n
        for seed,fold in cells(args):
            cell=f"seed_{seed}_fold_{fold}"
            for scope in ["development_validation","reference25"]:
                folder=Path(c["output_dir"])/"evaluations"/scope/cell/label
                if not (folder/"DONE.json").is_file():continue
                with (folder/"metrics.csv").open(newline="",encoding="utf-8") as f:
                    for row in csv.DictReader(f):metric_rows.append({**row,"strategy":friendly,"architecture":"nnunet" if friendly.startswith("nnunet") else "yolo","model_id":friendly+"_"+cell})
                with (folder/"surface_results.csv").open(newline="",encoding="utf-8") as f:
                    for row in csv.DictReader(f):surface_rows.append({**row,"strategy":friendly,"architecture":"nnunet" if friendly.startswith("nnunet") else "yolo","model_id":friendly+"_"+cell})
    write_csv(out/"reports/per_model_metrics.csv",metric_rows);write_csv(out/"reports/surface_results.csv",surface_rows)
    write(out/"reports/strategy_mapping.json",{k:{"backend":v[0],"source_label":v[1]} for k,v in STRATEGIES.items()})
    clinical=[row for row in surface_rows if row.get("scope")=="reference25"]
    if clinical:
        source=str(ROOT/"src")
        if source not in sys.path:sys.path.insert(0,source)
        from carieslab.predictions import write_regional_predictions
        report=write_regional_predictions(clinical,out/"reports/regional_predictions.csv.gz")
        write(out/"reports/regional_predictions_manifest.json",{"annotation_version":getattr(args,"annotation_version","recorded"),"requested_cells":plan(args)["requested_cells"],**report})

def status(args):
    out,_=paths(args);p=plan(args);p["completed_training_cells"]=sum((out/"cells"/f"seed_{s}_fold_{f}.json").is_file() for s,f in cells(args))
    p["development_frozen"]=(out/"DEVELOPMENT_FROZEN.json").is_file();p["assessment_complete"]=(out/"ASSESSMENT_COMPLETE.json").is_file()
    print(json.dumps(p,indent=2));return 0

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["plan","check","prepare","train","assess","status"])
    parser.add_argument("--data-zip",default=str(ROOT/"data/Datasets.zip"),help="Frozen study data ZIP, including refined DCBR masks and anatomical proximal regions.")
    parser.add_argument("--output",default=str(ROOT/"runs/main"),help="New or matching run directory; existing results are protected.")
    parser.add_argument("--device",default="0",help="CUDA device index, or cpu explicitly.")
    parser.add_argument("--annotation-version",choices=["recorded","updated"],default="recorded",help="recorded reproduces the archived input masks; updated explicitly uses the revised study masks and audits small-lesion loss on the fixed training grid. Results can differ.")
    parser.add_argument("--seed",type=int,choices=SEEDS,nargs="+",help="Subset for a smoke reproduction. Default: all three seeds.")
    parser.add_argument("--fold",type=int,choices=FOLDS,nargs="+",help="Subset for a smoke reproduction. Default: all five folds.")
    parser.add_argument("--runtime",action="store_true",help="For check: verify installed training versions and CUDA as well as source/data files.")
    args=parser.parse_args()
    if args.command=="plan":print(json.dumps(plan(args),indent=2));return 0
    try:return {"check":check,"prepare":prepare,"train":train,"assess":assess,"status":status}[args.command](args)
    except (FileNotFoundError,RuntimeError,ValueError,ModuleNotFoundError) as exc:
        print("ERROR: "+str(exc),file=sys.stderr);return 2

if __name__=="__main__":raise SystemExit(main())
