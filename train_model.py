# Load test data 
file_path = '/content/drive/My Drive/labeled_data.csv'
train_df = pd.read_csv(file_path)

train_df.head()

train_df['label'] = train_df['class'].apply(lambda x: 1 if x in [0, 1] else 0)
train_df['tweet'] = train_df['tweet'].astype(str)  # Ensure it's string

train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_df['tweet'].tolist(), 
    train_df['label'].tolist(), 
    test_size=0.2, 
    random_state=42
)

#Tokenize using BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

#Create HuggingFace datasets
train_dataset = Dataset.from_dict({**train_encodings, 'label': train_labels})
val_dataset = Dataset.from_dict({**val_encodings, 'label': val_labels})

#Load BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

#Define evaluation function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}

#Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    # Remove or comment out the following line:
    # evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
)

#Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

#Train the model
trainer.train()
