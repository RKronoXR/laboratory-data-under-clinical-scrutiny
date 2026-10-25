#!/usr/bin/env python3
"""Reproduce the retained additional nnU-Net pipelines in Appendix B.

Native additional routes reuse the main clinical and laboratory-guided models.
Earlier full-frame implementations use their own archived numerical engine.
They are never pooled with the main strategy comparison.
"""
from __future__ import annotations
import argparse,copy,importlib.util,json,subprocess,sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location("main_reproduction_backend",HERE.parent/"main/reproduce.py")
main_backend=importlib.util.module_from_spec(spec);spec.loader.exec_module(main_backend)
read=main_backend.read;write=main_backend.write;sha=main_backend.sha;digest=main_backend.digest
NATIVE={"nnunet_clinical_then_laboratory":"B","nnunet_laboratory_guidance_then_laboratory":"C"}
EARLIER={"earlier_nnunet_clinical":"A","earlier_nnunet_clinical_then_laboratory":"B",
    "earlier_nnunet_laboratory_guidance_then_laboratory":"C","earlier_nnunet_laboratory_distillation":"D",
    "earlier_nnunet_laboratory_pretraining":"E","earlier_nnunet_pretraining_and_distillation":"F"}

def plan(args):
    cells=main_backend.cells(args);profiles=["native","earlier"] if args.profile=="all" else [args.profile]
    return {"scope":"retained Appendix B nnU-Net pipelines","requested_cells":[{"seed":s,"fold":f} for s,f in cells],
        "profiles":{p:{"strategies":list(NATIVE if p=="native" else EARLIER),
            "endpoint_models":len(cells)*(2 if p=="native" else 6),
            "new_training_stages":len(cells)*2 if p=="native" else len(cells)*6+len({s for s,f in cells})} for p in profiles},
        "main_source":str(Path(args.main_output).resolve()),"output":str(Path(args.output).resolve()),
        "hyperparameter_search":False,"annotation_version":getattr(args,"annotation_version","recorded"),"earlier_results_combined_with_main":False}

def cfg(args):
    source_args=copy.copy(args);source_args.output=args.main_output
    y,n=main_backend.configs(source_args);out=Path(args.output).resolve()/args.profile
    if args.profile=="native":
        c=n;package=HERE.parent/"main/nnunet"
    else:
        c=read(ROOT/"configs/additional/earlier_fullframe.json");package=HERE/"earlier"
        c.update(project_root=str(ROOT),package_root=str(package),datasets_zip=str(Path(args.data_zip).resolve()),
            config_path=str(ROOT/"configs/additional/earlier_fullframe.json"),student_plans=str(ROOT/"configs/main/clinical_plans.json"),
            source_v3=str(HERE.parent/"main/yolo"),source_output=y["output_dir"],source_profile=y["profile"],
            hpo_enabled=[],profile="EARLIER_FULLFRAME_NO_HPO",run_name=None)
        c["teacher_reuse"]["legacy_splits"]=str(ROOT/"configs/main/clinical_splits.json")
        c["teacher_reuse"]["legacy_root"]=str(Path(args.main_output).resolve()/"prior_teachers")
    c.update(output_base=str(out),output_dir=str(out),package_root=str(package))
    c["training"]["device"]=args.device
    c["annotation_version"]=getattr(args,"annotation_version","recorded")
    return y,c

def identity(args,c):
    inputs=main_backend.verify_scientific_inputs(args.data_zip,getattr(args,"annotation_version","recorded"))
    if not inputs["verified"]:raise RuntimeError(f"Study inputs changed: {len(inputs['missing'])} missing and {len(inputs['changed'])} different scientific files. See main check.")
    state={"profile":args.profile,"data_sha256":sha(args.data_zip),"backend_sha256":sha(__file__),
        "requested_cells":plan(args)["requested_cells"],"main_output":str(Path(args.main_output).resolve()),
        "configuration":c,"config_sha256":sha(ROOT/"configs/additional/earlier_fullframe.json"),
        "source_manifest_sha256":sha(HERE/"UPSTREAM_MANIFEST.json")}
    path=Path(c["output_dir"])/"RUN_MANIFEST.json"
    if path.is_file() and read(path)!=state:raise RuntimeError("Additional run identity changed. Select another --output folder.")
    write(path,state);protocol=digest(state)
    write(Path(c["output_dir"])/"protocol.json",{"signature":protocol,"configuration":c,"public_appendix_run":True,"counts":plan(args)})
    return protocol

def engine(args,c,training=False):
    if args.profile=="native":
        _,r=main_backend.engines()
    else:
        sys.path.insert(0,str(HERE/"earlier"))
        from reversekd import pipeline as r,report,cache
        report.refresh=lambda c,plots=False:None
        # Explicit sharing below replaces old-machine cache discovery only.
        cache.reuse_training=lambda c,spec:None
        cache.find_target=lambda c,signature:None
        cache.reuse_data=lambda c,target,reference:False
    from reversekd import data
    source=importlib.util.find_spec("reversekd._source")
    import reversekd._source
    reversekd._source.RAW_NAMES.update(main_backend.dataset_names(args.data_zip))
    if training:
        from reversekd.check import freeze_plans
        freeze_plans(c)
    return r,data

def check(args):
    y,c=cfg(args);issues=[]
    for item in read(HERE/"UPSTREAM_MANIFEST.json")["files"]:
        p=ROOT/item["path"]
        if not p.is_file() or sha(p)!=item.get("bundled_sha256",item["sha256"]):issues.append("Source missing or changed: "+item["path"])
    if not Path(args.data_zip).is_file():issues.append("Study data ZIP is missing.")
    else:
        data=main_backend.verify_scientific_inputs(args.data_zip,getattr(args,"annotation_version","recorded"))
        if not data["verified"]:issues.append(f"Scientific input mismatch: {len(data['missing'])} missing and {len(data['changed'])} changed files.")
    source=Path(args.main_output).resolve()/"DEVELOPMENT_FROZEN.json"
    if not source.is_file():issues.append("Main training must first produce DEVELOPMENT_FROZEN.json: "+str(source))
    print(json.dumps({"status":"PASS" if not issues else "INPUTS_REQUIRED","issues":issues,"plan":plan(args)},indent=2))
    return 0 if not issues else 2

def prepare(args):
    y,c=cfg(args);identity(args,c)
    # Preparation uses no CUDA and does not construct a training controller.
    package=HERE.parent/"main/nnunet" if args.profile=="native" else HERE/"earlier"
    sys.path.insert(0,str(package))
    import reversekd._source
    from reversekd.data import prepare as prepare_data
    reversekd._source.RAW_NAMES.update(main_backend.dataset_names(args.data_zip));prepare_data(c)
    print("Prepared "+args.profile+" development inputs.");return 0

def source_models(args,y,seed,fold):
    cell=f"seed_{seed}_fold_{fold}";base=Path(y["output_dir"])
    source_manifest=read(Path(args.main_output).resolve()/"RUN_MANIFEST.json")
    if source_manifest.get("annotation_version","recorded")!=args.annotation_version or source_manifest.get("data_sha256")!=sha(args.data_zip):
        raise RuntimeError("The main source models were fitted with a different annotation version or data archive. Reuse was refused.")
    model=read(base/"selections"/cell/"T.json")
    for label in ["T","P"]:
        selected=read(base/"selections"/cell/(label+".json"))
        if sha(selected["checkpoint"])!=selected["checkpoint_sha256"]:raise RuntimeError("Main source checkpoint changed.")
    frozen_path=Path(args.main_output).resolve()/"DEVELOPMENT_FROZEN.json"
    if not frozen_path.is_file():raise RuntimeError("Complete main development before additional training.")
    frozen=read(frozen_path)
    if model["checkpoint_sha256"]!=frozen["models"][cell+"/nnunet_clinical"]["checkpoint_sha256"]:
        raise RuntimeError("The main clinical starting weights differ from the frozen development record.")
    return model

def train(args):
    y,c=cfg(args);protocol=identity(args,c);main_backend.require_training_runtime(args);r,data_module=engine(args,c,training=True)
    data=data_module.prepare(c);selected_map=NATIVE if args.profile=="native" else EARLIER
    from reversekd.professor import audit_one,snapshot
    from reversekd.io import atomic_json
    shared={}
    for seed,fold in main_backend.cells(args):
        cell=f"seed_{seed}_fold_{fold}";teacher_source=source_models(args,y,seed,fold)
        split=next(s for s in data["splits"] if s["fold"]==fold)
        validation=[i for i in data["images"] if i["sample_id"] in split["validation_ids"]]
        if args.profile=="native":
            recipe=main_backend.baseline_recipe(c,teacher_source);baseline={**teacher_source,"native_recipe":recipe}
            main_args=copy.copy(args);main_args.output=args.main_output;_,nc=main_backend.configs(main_args)
            guided=read(Path(nc["output_dir"])/"selections"/cell/"D.json")
            if sha(guided["checkpoint"])!=guided["checkpoint_sha256"]:raise RuntimeError("Main guided checkpoint changed.")
            main_frozen=read(Path(args.main_output).resolve()/"DEVELOPMENT_FROZEN.json")
            if guided["checkpoint_sha256"]!=main_frozen["models"][cell+"/nnunet_laboratory_distillation"]["checkpoint_sha256"]:
                raise RuntimeError("The main laboratory-guided starting weights differ from the frozen development record.")
            fit={**c,"native_recipe":recipe,"native_bundle":baseline["native_bundle"],"source_native_bundle":baseline["native_bundle"],
                "native_gate_path":str(Path(nc["output_dir"])/"preflight"/cell/"NATIVE_EQUIVALENCE.json")}
            schedule=[("B_ACTA",baseline,"B"),("C_ACTA",guided,"C")]
            for stage,parent,label in schedule:
                result=r.train_task(fit,cell,stage,split,protocol,parent)
                atomic_json(Path(c["output_dir"])/"selections"/cell/(label+".json"),result)
                r.evaluate(fit,cell,label,result,validation,data["surfaces"],"development_validation",protocol)
        else:
            teacher=snapshot(c,audit_one(c,seed,fold));train_images=[i for i in data["images"] if i["sample_id"] in split["train_ids"]]
            targets=r.target_task(c,cell,teacher,train_images,data["surfaces"],"training_only",protocol)
            gate=Path(c["output_dir"])/"preflight"/cell
            if not (gate/"result.json").is_file():r.invoke(c,{"operation":"preflight","task_id":cell+"/preflight","directory":str(gate),"teacher":teacher,"teacher_maps":{train_images[0]["sample_id"]:targets[train_images[0]["sample_id"]]},"protocol":protocol})
            stages={}
            for stage,parent,label,kd in [("N",None,"A",False),("K",None,"D",True),("B_ACTA","N","B",False),("C_ACTA","K","C",False),("V",None,None,False),("E_CLIN","V","E",False),("F_CLIN","V","F",True)]:
                if stage=="V" and seed in shared:result=shared[seed]
                else:
                    kwargs={"folder":Path(c["output_dir"])/"shared_pretraining"/f"seed_{seed}","task_id":f"shared_seed_{seed}/V"} if stage=="V" else {}
                    result=r.train_task(c,cell,stage,split,protocol,stages[parent] if parent else None,targets if kd else None,**kwargs)
                    if stage=="V":shared[seed]=result
                stages[stage]=result
                if label:
                    atomic_json(Path(c["output_dir"])/"selections"/cell/(label+".json"),result)
                    r.evaluate(c,cell,label,result,validation,data["surfaces"],"development_validation",protocol)
        write(Path(c["output_dir"])/"cells"/(cell+".json"),{"trained":True,"seed":seed,"fold":fold})
    frozen={"protocol":protocol,"models":{},"requested_cells":plan(args)["requested_cells"]}
    for seed,fold in main_backend.cells(args):
        cell=f"seed_{seed}_fold_{fold}"
        for friendly,label in selected_map.items():
            result=read(Path(c["output_dir"])/"selections"/cell/(label+".json"))
            frozen["models"][cell+"/"+friendly]={"checkpoint_sha256":sha(result["checkpoint"]),"threshold_sha256":sha(Path(c["output_dir"])/"thresholds"/cell/(label+".json"))}
    write(Path(c["output_dir"])/"DEVELOPMENT_FROZEN.json",frozen);export(args,c);print("Additional development frozen: "+args.profile);return 0

def assess(args):
    y,c=cfg(args);protocol=identity(args,c);main_backend.require_training_runtime(args);r,data_module=engine(args,c,training=False)
    frozen=read(Path(c["output_dir"])/"DEVELOPMENT_FROZEN.json");mapping=NATIVE if args.profile=="native" else EARLIER
    for seed,fold in main_backend.cells(args):
        cell=f"seed_{seed}_fold_{fold}"
        for friendly,label in mapping.items():
            model=read(Path(c["output_dir"])/"selections"/cell/(label+".json"));item=frozen["models"][cell+"/"+friendly]
            if sha(model["checkpoint"])!=item["checkpoint_sha256"] or sha(Path(c["output_dir"])/"thresholds"/cell/(label+".json"))!=item["threshold_sha256"]:raise RuntimeError("Frozen additional model/threshold changed.")
    data=data_module.prepare(c,reference=True)
    for seed,fold in main_backend.cells(args):
        cell=f"seed_{seed}_fold_{fold}"
        for friendly,label in mapping.items():
            model=read(Path(c["output_dir"])/"selections"/cell/(label+".json"));fit={**c,"native_recipe":model["native_recipe"]} if args.profile=="native" else c
            r.evaluate(fit,cell,label,model,data["images"],data["surfaces"],"reference25",protocol)
    export(args,c);write(Path(c["output_dir"])/"ASSESSMENT_COMPLETE.json",{"protocol":protocol});return 0

def export(args,c):
    import csv
    metrics=[];surfaces=[];mapping=NATIVE if args.profile=="native" else EARLIER
    for friendly,label in mapping.items():
        for seed,fold in main_backend.cells(args):
            cell=f"seed_{seed}_fold_{fold}"
            for scope in ["development_validation","reference25"]:
                folder=Path(c["output_dir"])/"evaluations"/scope/cell/label
                if not (folder/"DONE.json").is_file():continue
                for name,dest in [("metrics.csv",metrics),("surface_results.csv",surfaces)]:
                    with (folder/name).open(newline="",encoding="utf-8") as f:
                        for row in csv.DictReader(f):dest.append({**row,"strategy":friendly,"implementation":args.profile,"model_id":friendly+"_"+cell})
    main_backend.write_csv(Path(c["output_dir"])/"reports/per_model_metrics.csv",metrics)
    main_backend.write_csv(Path(c["output_dir"])/"reports/surface_results.csv",surfaces)

def status(args):
    y,c=cfg(args);out=Path(c["output_dir"])
    print(json.dumps({"profile":args.profile,"training_cells":sum((out/"cells"/f"seed_{s}_fold_{f}.json").is_file() for s,f in main_backend.cells(args)),"development_frozen":(out/"DEVELOPMENT_FROZEN.json").is_file(),"assessment_complete":(out/"ASSESSMENT_COMPLETE.json").is_file()},indent=2));return 0

def run_all(args):
    for profile in ["native","earlier"]:
        command=[sys.executable,str(__file__),args.command,"--profile",profile,"--data-zip",args.data_zip,"--output",args.output,"--main-output",args.main_output,"--device",args.device,"--annotation-version",args.annotation_version]
        if args.seed:command+=["--seed",*map(str,args.seed)]
        if args.fold:command+=["--fold",*map(str,args.fold)]
        result=subprocess.run(command)
        if result.returncode:return result.returncode
    return 0

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=["plan","check","prepare","train","assess","status"])
    p.add_argument("--profile",choices=["all","native","earlier"],default="all")
    p.add_argument("--data-zip",default=str(ROOT/"data/Datasets.zip"));p.add_argument("--main-output",default=str(ROOT/"runs/main"));p.add_argument("--output",default=str(ROOT/"runs/additional"));p.add_argument("--device",default="0")
    p.add_argument("--annotation-version",choices=["recorded","updated"],default="recorded")
    p.add_argument("--seed",type=int,choices=main_backend.SEEDS,nargs="+");p.add_argument("--fold",type=int,choices=main_backend.FOLDS,nargs="+");a=p.parse_args()
    if a.command=="plan":print(json.dumps(plan(a),indent=2));return 0
    try:
        if a.profile=="all":return run_all(a)
        return {"check":check,"prepare":prepare,"train":train,"assess":assess,"status":status}[a.command](a)
    except (OSError,RuntimeError,ValueError,ModuleNotFoundError) as exc:print("ERROR: "+str(exc),file=sys.stderr);return 2

if __name__=="__main__":raise SystemExit(main())
