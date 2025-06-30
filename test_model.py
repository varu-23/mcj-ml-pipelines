#Upload dataset

url = "https://raw.githubusercontent.com/eimearfoley/CyberBullyingDetection/refs/heads/master/data/dataset.txt"
test_df = pd.read_csv(url, sep="\t", header=None, names=["label", "text"])

# Binary label: 1 = cyberbullying, 0 = not
test_df['label'] = test_df['label'].apply(lambda x: 1 if str(x).lower() == 'cyberbullying' else 0)
test_df['text'] = test_df['text'].astype(str)

#Tokenize data
test_encodings = tokenizer(test_df['text'].tolist(), truncation=True, padding=True, max_length=128)

test_dataset = Dataset.from_dict({**test_encodings, 'label': test_df['label'].tolist()})

metrics = trainer.evaluate(eval_dataset=test_dataset)
print("🔎 Evaluation on CyberBullyingDetection dataset:")
for key, value in metrics.items():
    print(f"{key}: {value:.4f}")

