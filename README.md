# 학습 실행
```
make train ARGS="--train_data data/train.csv --output_dir outputs_gemma"
```

# 추론 실행
```
make inference ARGS="--checkpoint_path outputs_gemma --test_data data/test.csv --output_path output.csv"
```
