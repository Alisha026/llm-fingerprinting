"""Recreate the original fold-0 Kaggle upload bundle without training."""
from pathlib import Path
import zipfile


def main():
    target=Path('artifacts/kaggle_upload/llm-fingerprint-fold0.zip')
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_STORED) as archive:
        archive.write('src/kaggle_inference.py','kaggle_inference.py')
        archive.write('artifacts/hybrid_word/classifier.joblib','classifier.joblib')
        for source, destination in [('artifacts/encoder_v1/encoder','frozen_encoder'),
                                    ('artifacts/finetune_distilbert_fold0_retry/best_model','finetuned_model')]:
            for file in sorted(Path(source).iterdir()):
                if file.is_file():
                    archive.write(file,str(Path(destination)/file.name))
    print(target)


if __name__ == '__main__':
    main()
