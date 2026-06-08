import glob
import os
from contextlib import redirect_stdout
from keras.models import load_model

model_paths = sorted(glob.glob("./model_*.h5"))

with open("model_structures.txt", "w", encoding="utf-8") as f:
    for path in model_paths:
        name = os.path.basename(path)

        print("=" * 100)
        print(name)
        print("=" * 100)

        print("=" * 100, file=f)
        print(name, file=f)
        print("=" * 100, file=f)

        try:
            model = load_model(path, safe_mode=False, compile=False)

            model.summary()

            with redirect_stdout(f):
                model.summary()

        except Exception as e:
            print("로드 실패:", e)
            print("로드 실패:", e, file=f)

        print()
        print(file=f)

print("완료: model_structures.txt 생성됨")