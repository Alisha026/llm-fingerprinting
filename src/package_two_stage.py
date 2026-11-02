"""Build a private Kaggle dataset upload and inference-only notebook; no training."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


def main():
    root=Path('artifacts/kaggle_two_stage_8b')
    if root.exists():
        raise FileExistsError(f'Preserve existing package: {root}')
    bundle=root/'bundle'; upload=root/'upload'; kernel=root/'kernel'
    for p in (bundle,upload,kernel): p.mkdir(parents=True)
    run=Path('artifacts/two_stage_8b_job_5895767')
    shutil.copytree(run/'official',bundle/'official')
    shutil.copyfile('src/two_stage_inference.py',bundle/'two_stage_inference.py')
    shutil.copyfile('data/processed/validation.csv',bundle/'validation.csv')
    shutil.copyfile(run/'official_centered_blend_probabilities.csv',bundle/'reference_probabilities.csv')
    manifest={'run':str(run),'base_repo':'Qwen/Qwen3-8B-Base',
              'base_revision':'49e3418fbbbca6ecbdf9608b4d22e5a407081db4',
              'scope':'Inference only; fold-0 checkpoint; no full-data refit',
              'sha256':{str(p.relative_to(bundle)):hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted(bundle.rglob('*')) if p.is_file()}}
    (bundle/'bundle_manifest.json').write_text(json.dumps(manifest,indent=2))
    with zipfile.ZipFile(upload/'two_stage_8b_bundle.zip','w',compression=zipfile.ZIP_STORED) as z:
        for p in sorted(bundle.rglob('*')):
            if p.is_file(): z.write(p,p.relative_to(bundle))
    (upload/'dataset-metadata.json').write_text(json.dumps({
        'title':'NTL Two Stage 8B Fold0 Checkpoint', 'id':'alishasaifi/ntl-two-stage-8b-fold0-checkpoint',
        'licenses':[{'name':'CC-BY-SA-4.0'}],
        'description':'Private inference checkpoint from job 5895767. Base Qwen3-8B-Base (Apache-2.0). Public training pool credited to nishantdahal/ntl-two-stage-lora and its 14 upstream dataset authors. Local fold-0 exploratory accuracy 0.6148 is not a Kaggle score. No base weights included.'},indent=2))
    code='''import hashlib, json, os, sys, zipfile
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
inputs=Path('/kaggle/input')
markers=list(inputs.rglob('bundle_manifest.json'))
if markers:
    bundle=markers[0].parent
else:
    archives=list(inputs.rglob('two_stage_8b_bundle.zip'))
    assert len(archives)==1, 'Attach the private checkpoint dataset'
    bundle=Path('/kaggle/working/two_stage_bundle')
    bundle.mkdir(exist_ok=True)
    with zipfile.ZipFile(archives[0]) as z:
        assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist())
        z.extractall(bundle)
manifest=json.loads((bundle/'bundle_manifest.json').read_text())
for name,digest in manifest['sha256'].items():
    assert hashlib.sha256((bundle/name).read_bytes()).hexdigest()==digest, name
base=None
for path in inputs.rglob('config.json'):
    try: cfg=json.loads(path.read_text())
    except (ValueError,OSError): continue
    if cfg.get('model_type')=='qwen3' and cfg.get('hidden_size')==4096 and cfg.get('num_hidden_layers')==36:
        base=path.parent
        break
assert base is not None, 'Attach Qwen3-8B-Base'
tests=[p for p in inputs.rglob('test.csv') if p.parent.name=='name-that-llm']
assert len(tests)==1, 'Attach the competition'
sys.path.insert(0,str(bundle))
from two_stage_inference import run
run(bundle,base,tests[0])
'''
    notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},
              'cells':[{'cell_type':'markdown','metadata':{},'source':'# NTL Two Stage 8B Fold0\nInference only. Reuses the fixed checkpoint with plain + 0.35 centered log probabilities. No new classifier fitting, no Sinkhorn, and no test-label access. Validates against saved local probabilities before writing submission.parquet.'},
                       {'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':code}]}
    name='ntl-two-stage-8b-fold0.ipynb'
    (kernel/name).write_text(json.dumps(notebook,indent=2))
    shutil.copyfile(kernel/name,Path('notebooks')/name)
    (kernel/'kernel-metadata.json').write_text(json.dumps({
        'id':'alishasaifi/ntl-two-stage-8b-fold0','title':'NTL Two Stage 8B Fold0','code_file':name,
        'language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_internet':False,
        'dataset_sources':['alishasaifi/ntl-two-stage-8b-fold0-checkpoint'],
        'competition_sources':['name-that-llm'],
        'model_sources':['qwen-lm/qwen-3/Transformers/8b-base/1']},indent=2))
    print('Prepared',root)
    print('ZIP bytes:',(upload/'two_stage_8b_bundle.zip').stat().st_size)


if __name__=='__main__': main()
